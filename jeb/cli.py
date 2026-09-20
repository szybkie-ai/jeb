"""`jeb serve | ask | eval | calibrate`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def cmd_serve(a: argparse.Namespace) -> None:
    import uvicorn

    for key, val in {
        "JEB_ENGINE": a.engine,
        "JEB_ENGINE_URL": a.engine_url,
        "JEB_ENGINE_KEY": a.engine_key,
        "JEB_ENGINE_MODEL": a.engine_model,
        "JEB_TOKENIZER": a.tokenizer,
        "JEB_MODEL_ID": a.model_id,
        "JEB_API_KEYS": a.api_keys,
        "JEB_CALIBRATION": a.calibration,
        "JEB_PERMUTATIONS": a.permutations,
        "JEB_PREFIX_PAD_BLOCK": a.prefix_pad_block,
    }.items():
        if val is not None:
            os.environ[key] = str(val)
    uvicorn.run("jeb.server:app_factory", factory=True, host=a.host, port=a.port, workers=1, log_level=a.log_level)


def cmd_ask(a: argparse.Namespace) -> None:
    import httpx

    body = json.load(open(a.file)) if a.file else json.load(sys.stdin)
    headers = {"Authorization": f"Bearer {a.key}"} if a.key else {}
    r = httpx.post(f"{a.url.rstrip('/')}/v1/systemone", json=body, headers=headers, timeout=120)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


def cmd_eval(a: argparse.Namespace) -> None:
    from jeb.evals.datasets import load
    from jeb.evals.run import run

    items = load(a.task, a.n, a.seed, a.offset)
    out = Path(a.out or f"results/{a.task}-n{a.n}-off{a.offset}-p{a.permutations}.jsonl")
    summary = asyncio.run(run(items, a.url, a.key, a.model, a.permutations, a.concurrency, out))
    print(json.dumps(summary, indent=2))


def cmd_calibrate(a: argparse.Namespace) -> None:
    from jeb.calibration_fit import Record, evaluate, fit, variants

    def read(paths: list[str]) -> list[Record]:
        rows: list[Record] = []
        for p in paths:
            rows += [Record.from_json(json.loads(line)) for line in Path(p).read_text().splitlines() if line.strip()]
        return rows

    fit_rows = read(a.fit)
    perms = max(len(r.permutations) for r in fit_rows)
    cal = fit(fit_rows, permutations=perms, meta={"fit_files": a.fit})
    print("fitted:", json.dumps({"temperature": cal.temperature, "log_prior": cal.log_prior, "permutations": cal.permutations}, indent=1))
    for path in a.eval or []:
        rows = read([path])
        print(f"\n== {path} (n={len(rows)}, kind={rows[0].kind}) ==")
        table = variants(rows, cal)
        keys = ["accuracy", "ece", "brier", "nll"] + (["score_mae"] if "score_mae" in table["+ temperature"] else [])
        print(f"{'variant':<26}" + "".join(f"{k:>10}" for k in keys))
        for name, m in table.items():
            print(f"{name:<26}" + "".join(f"{m[k]:>10.4f}" for k in keys))
        print("coverage (calibrated):", " ".join(f"{c['confidence>=']}:{c['coverage']}/{c['accuracy']}" for c in table["+ temperature"]["coverage"]))
        if a.reliability:
            for r in evaluate(rows, cal)["reliability"]:
                print(f"  {r['bin']:<12} n={r['n']:<4} mean_p={r['mean_p']:.3f} acc={r['accuracy']:.3f}")
    if a.out:
        Path(a.out).write_text(cal.to_json())
        print(f"\nwrote {a.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="jeb")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the System One-compatible API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8020)
    s.add_argument("--engine", choices=["vllm-http", "llama-server", "fake"], default=None, help="default: $JEB_ENGINE or vllm-http; llama-server = llama.cpp / GGUF builds")
    s.add_argument("--engine-url", default=None, help="OpenAI-compatible vLLM base URL (default http://127.0.0.1:8021)")
    s.add_argument("--engine-key", default=None)
    s.add_argument("--engine-model", default=None, help="served model name at the engine")
    s.add_argument("--tokenizer", default=None, help="HF id or local path (default Qwen/Qwen3.5-4B)")
    s.add_argument("--model-id", default=None, help="model id this server answers for (default jeb-4b)")
    s.add_argument("--api-keys", default=None, help="comma-separated bearer keys; unset = open")
    s.add_argument("--calibration", default=None, help="calibration.json path")
    s.add_argument("--permutations", type=int, default=None, help="default option orderings per question (0 = calibration file's)")
    s.add_argument("--prefix-pad-block", type=int, default=None, help="pad the shared prefix to a multiple of N tokens (528 for vLLM hybrid cache)")
    s.add_argument("--log-level", default="info")
    s.set_defaults(fn=cmd_serve)

    q = sub.add_parser("ask", help="POST a request JSON (file or stdin) to a server")
    q.add_argument("--url", default="http://127.0.0.1:8020")
    q.add_argument("--key", default=None)
    q.add_argument("file", nargs="?")
    q.set_defaults(fn=cmd_ask)

    e = sub.add_parser("eval", help="run a public labelled set through a server, recording raw label logprobs")
    e.add_argument("--task", required=True, choices=["sst2", "agnews", "stsb"])
    e.add_argument("--url", default="http://127.0.0.1:8020")
    e.add_argument("--key", default=None)
    e.add_argument("--model", default="jeb-latest")
    e.add_argument("-n", type=int, default=200)
    e.add_argument("--offset", type=int, default=0, help="skip this many items of the seeded shuffle (disjoint fit/eval splits)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--permutations", type=int, default=4)
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("calibrate", help="fit position priors + temperature from eval records; report the ablation table")
    c.add_argument("--fit", nargs="+", required=True, help="records to fit on (any mix of kinds)")
    c.add_argument("--eval", nargs="*", default=None, help="held-out records to report on, one file per task")
    c.add_argument("--out", default=None, help="write calibration.json here")
    c.add_argument("--reliability", action="store_true")
    c.set_defaults(fn=cmd_calibrate)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
