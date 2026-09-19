"""The System One-compatible HTTP API.

    POST /v1/systemone   evaluate a state against typed questions
    GET  /v1/models      models this server answers for
    GET  /healthz        liveness (checks the engine)
    GET  /metrics        Prometheus

Configuration is by environment (see `Settings`), so the same server runs on a laptop with an
in-process engine and on a lab box behind a shared vLLM.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from jeb import __version__
from jeb.decide import Calibration, Scored, build_answers
from jeb.engines import Engine, FakeEngine, ScoreItem, VllmHttpEngine
from jeb.engines.base import EngineError
from jeb.images import ImageTokenCounter
from jeb.prompt import TEMPLATE_VERSION, HFTokenizer, Labeler, LabelError, Tokenizer, build_plans, pad_text_for
from jeb.schema import (
    ListModelsResponse,
    ModelMetadata,
    StructuredRequest,
    StructuredResponse,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    error_body,
)
from jeb.structured import SchemaError, assemble, compile_schema

log = logging.getLogger("jeb")

REQUESTS = Counter("jeb_requests_total", "requests", ["status"])
QUESTIONS = Counter("jeb_questions_total", "questions answered", ["kind"])
LATENCY = Histogram("jeb_request_seconds", "end-to-end request latency", buckets=(0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5))
MISSING = Counter("jeb_missing_label_logprobs_total", "label tokens the engine did not report")


@dataclass
class Settings:
    engine: str = "vllm-http"  # vllm-http | fake
    engine_url: str = "http://127.0.0.1:8021"
    engine_key: str = "local"
    engine_model: str = "openjev-base"
    tokenizer: str = "Qwen/Qwen3.5-4B"
    model_id: str = "jeb-4b"
    model_aliases: list[str] = field(default_factory=lambda: ["jeb-latest", "openjev-latest", "jev-latest"])  # extra model ids the server answers to
    api_keys: list[str] = field(default_factory=list)  # empty = open
    calibration: str | None = None
    permutations: int = 0  # 0 = the calibration file's recommendation, else 1
    prefix_pad_block: int = 0  # pad the shared prefix to a multiple of this many tokens (528 = vLLM's hybrid-cache block)
    max_questions: int = 256

    @classmethod
    def from_env(cls) -> Settings:
        e = _env
        split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
        s = cls()
        s.engine = e("JEB_ENGINE", s.engine)
        s.engine_url = e("JEB_ENGINE_URL", s.engine_url)
        s.engine_key = e("JEB_ENGINE_KEY", s.engine_key)
        s.engine_model = e("JEB_ENGINE_MODEL", s.engine_model)
        s.tokenizer = e("JEB_TOKENIZER", s.tokenizer)
        s.model_id = e("JEB_MODEL_ID", s.model_id)
        s.model_aliases = split(e("JEB_MODEL_ALIASES", ",".join(s.model_aliases)))
        s.api_keys = split(e("JEB_API_KEYS", ""))
        s.calibration = e("JEB_CALIBRATION") or None
        s.permutations = int(e("JEB_PERMUTATIONS", s.permutations))
        s.prefix_pad_block = int(e("JEB_PREFIX_PAD_BLOCK", s.prefix_pad_block))
        s.max_questions = int(e("JEB_MAX_QUESTIONS", s.max_questions))
        return s


def _env(name: str, default: Any = None) -> Any:
    """`JEB_*` first, then the pre-rename `OPENJEV_*` spelling of the same variable."""
    val = os.environ.get(name)
    if val is None and name.startswith("JEB_"):
        val = os.environ.get("OPENJEV_" + name[len("JEB_"):])
    return default if val is None else val


def make_engine(s: Settings) -> Engine:
    if s.engine == "fake":
        return FakeEngine()
    if s.engine == "vllm-http":
        return VllmHttpEngine(s.engine_url, s.engine_model, api_key=s.engine_key)
    raise ValueError(f"unknown engine {s.engine!r}")


def create_app(settings: Settings | None = None, engine: Engine | None = None, tokenizer: Tokenizer | None = None) -> FastAPI:
    s = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = s
        app.state.engine = engine or make_engine(s)
        app.state.labeler = Labeler(tokenizer or HFTokenizer.load(s.tokenizer))
        app.state.pad_text = pad_text_for(app.state.labeler.tokenizer)
        app.state.image_tokens = ImageTokenCounter()
        app.state.calibration = Calibration.load(s.calibration)
        app.state.started = time.strftime("%Y-%m-%d")
        log.info("jeb %s: engine=%s model=%s tokenizer=%s calibration=%s", __version__, app.state.engine.name, s.model_id, s.tokenizer, app.state.calibration.source)
        try:
            yield
        finally:
            await app.state.engine.aclose()

    app = FastAPI(title="JEB", version=__version__, lifespan=lifespan)

    async def require_key(request: Request) -> None:
        if not s.api_keys:
            return
        auth = request.headers.get("authorization", "")
        key = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not key:
            raise _Http(403, "authentication_error", "Must supply an API key! Check your request and try again.")
        if key not in s.api_keys:
            raise _Http(401, "authentication_error", "Invalid API key.")

    @app.exception_handler(_Http)
    async def _http_handler(_: Request, exc: _Http) -> JSONResponse:
        REQUESTS.labels(str(exc.status)).inc()
        return JSONResponse(status_code=exc.status, content=error_body(exc.error_type, exc.message))

    async def evaluate(req: SystemOneRequest, request: Request) -> SystemOneResponse:
        """The whole pipeline for one request: plans -> engine -> answers (+ usage, debug)."""
        t0 = time.perf_counter()
        if req.model not in (s.model_id, *s.model_aliases):
            raise _Http(422, "invalid_request_error", f"unknown model {req.model!r}; this server answers for {s.model_id}")
        if len(req.questions) > s.max_questions:
            raise _Http(422, "invalid_request_error", f"at most {s.max_questions} questions per request")
        opts = req.options
        calib: Calibration = request.app.state.calibration
        permutations = (opts.permutations if opts and opts.permutations else 0) or s.permutations or calib.permutations or 1
        extra = 0
        if req.images:
            try:
                for url in req.images:
                    extra += await request.app.state.image_tokens.tokens_for(request.app.state.engine, url)
            except EngineError as e:
                raise _Http(529, "overloaded_error", f"engine unavailable: {e}") from e
        try:
            plans, prefix = build_plans(req, request.app.state.labeler, permutations, pad_block=s.prefix_pad_block, extra_prefix_tokens=extra, pad_text=request.app.state.pad_text)
        except (LabelError, ValueError) as e:
            raise _Http(422, "invalid_request_error", str(e)) from e

        try:
            chat = {"system": plans[0].system, "images": list(req.images)} if req.images else None
            results = await request.app.state.engine.score([ScoreItem(p.prompt_ids, p.label_ids, chat={**chat, "user": p.user_text} if chat else None) for p in plans])
        except EngineError as e:
            log.warning("engine error: %s", e)
            raise _Http(529, "overloaded_error", f"engine unavailable: {e}") from e

        cal = calib if (opts is None or opts.calibration == "calibrated") else None
        answers, dbg = build_answers(req, [Scored(p, r.logprobs) for p, r in zip(plans, results)], cal, debug=bool(opts and opts.debug))
        missing = sum(r.missing for r in results)
        if missing:
            MISSING.inc(missing)
        for p in plans:
            QUESTIONS.labels(p.kind).inc()
        # Usage mirrors the upstream economics: the shared prefix once (without cache-alignment padding),
        # then each question's own tokens.
        usage = Usage(input_tokens=prefix.tokens + prefix.extra + sum(len(p.prompt_ids) - p.prefix_len for p in plans), output_tokens=len(plans))
        if dbg is not None and (opts and opts.debug):
            dbg["_engine"] = {
                "name": request.app.state.engine.name,
                "prompts_sent": len(plans),
                "engine_prompt_tokens": sum(r.prompt_tokens for r in results),
                "prefix_tokens": prefix.tokens,
                "prefix_pad_tokens": prefix.pad,
                "prefix_pad_block": s.prefix_pad_block,
                "permutations": permutations,
                "missing_label_logprobs": missing,
                "template": TEMPLATE_VERSION,
                "images": len(req.images or []),
                "image_tokens": prefix.extra,
                "pad_text": request.app.state.pad_text,
                "calibration": calib.source if cal else "raw",
            }
        LATENCY.observe(time.perf_counter() - t0)
        REQUESTS.labels("200").inc()
        return SystemOneResponse(model=s.model_id, answers=answers, usage=usage, debug=dbg if (opts and opts.debug) else None)

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True, dependencies=[Depends(require_key)])
    async def systemone(req: SystemOneRequest, request: Request) -> SystemOneResponse:
        return await evaluate(req, request)

    @app.post("/v1/structured", response_model=StructuredResponse, response_model_exclude_none=True, dependencies=[Depends(require_key)])
    async def structured(req: StructuredRequest, request: Request) -> StructuredResponse:
        """JEB extension: a JSON-Schema subset -> parallel questions -> the answer in the schema's shape."""
        try:
            questions, plan = compile_schema(req.schema_)
        except SchemaError as e:
            raise _Http(422, "invalid_request_error", f"schema: {e}") from e
        inner = SystemOneRequest.model_validate({"state": req.state, "model": req.model, "questions": questions, "options": req.options.model_dump() if req.options else None, "images": req.images})
        resp = await evaluate(inner, request)
        return StructuredResponse(model=resp.model, value=assemble(plan, resp.answers), fields=resp.answers, usage=resp.usage, debug=resp.debug)

    @app.get("/v1/models", response_model=ListModelsResponse, dependencies=[Depends(require_key)])
    async def models(request: Request) -> ListModelsResponse:
        return ListModelsResponse(models=[ModelMetadata(name=s.model_id, description=f"JEB {__version__} on {request.app.state.engine.name} ({s.engine_model})", release_date=request.app.state.started)])

    @app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        return {"ok": True, "engine": request.app.state.engine.name, "model": s.model_id, "version": __version__}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


class _Http(Exception):
    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status, self.error_type, self.message = status, error_type, message


def app_factory() -> FastAPI:
    """`uvicorn jeb.server:app_factory --factory`."""
    logging.basicConfig(level=_env("JEB_LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return create_app()
