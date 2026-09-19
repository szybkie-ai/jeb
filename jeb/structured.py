"""Structured output on top of the primitives: a JSON-Schema *subset* compiled to one fan-out of typed questions,
answered in parallel, assembled back into the schema's shape.

Supported (anything else is rejected with a clear message — nothing is ever generated):

    type: object            properties are compiled recursively; ids are dotted paths
    type: boolean           -> noul        (instructions: `description`, or "Is <name> true?"; `x-criteria: {true, false}`)
    type: string + enum     -> choice      (option rubrics from `x-descriptions: {value: text}`)
    type: string + x-candidates: [...]
                            -> choice      over the candidates your code found (select, not generate)
    type: integer + minimum/maximum (span <= 26)
                            -> score       (levels minimum..maximum; texts from `x-levels: [...]`, else the numbers)
    type: array + items.enum (multi-label)
                            -> one noul per option, kept when p >= `x-threshold` (default 0.5)

The assembled `value` holds plain JSON; `fields` holds every underlying answer (probabilities, confidence).
"""

from __future__ import annotations

from typing import Any

MAX_LEVELS = 26


class SchemaError(ValueError):
    pass


def compile_schema(schema: dict[str, Any], prefix: str = "") -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (questions, plan) where plan maps question ids to how their answers assemble."""
    if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
        raise SchemaError("the root schema must be an object with properties")
    questions: dict[str, dict[str, Any]] = {}
    plan: dict[str, dict[str, Any]] = {}
    for name, sub in schema["properties"].items():
        path = f"{prefix}{name}"
        if not isinstance(sub, dict):
            raise SchemaError(f"{path}: property schemas must be objects")
        t = sub.get("type")
        desc = sub.get("description")
        if t == "object":
            q, p = compile_schema(sub, prefix=f"{path}.")
            questions.update(q)
            plan.update(p)
        elif t == "boolean":
            crit = sub.get("x-criteria")
            questions[path] = {"type": "noul", "instructions": desc or f"Is `{name}` true for the state?", **({"criteria": crit} if crit else {})}
            plan[path] = {"kind": "boolean"}
        elif t == "string" and isinstance(sub.get("enum"), list):
            opts = [str(v) for v in sub["enum"]]
            _check_options(path, opts)
            rub = sub.get("x-descriptions") or {}
            questions[path] = {"type": "choice", "instructions": desc or f"Which `{name}` applies?", "criteria": {o: rub.get(o) for o in opts}}
            plan[path] = {"kind": "enum"}
        elif t == "string" and isinstance(sub.get("x-candidates"), list):
            opts = [str(v) for v in sub["x-candidates"]]
            _check_options(path, opts)
            questions[path] = {"type": "choice", "instructions": desc or f"Which candidate is the correct `{name}`?", "criteria": {o: None for o in opts}}
            plan[path] = {"kind": "candidate"}
        elif t == "integer" and "minimum" in sub and "maximum" in sub:
            lo, hi = int(sub["minimum"]), int(sub["maximum"])
            if hi <= lo or hi - lo + 1 > MAX_LEVELS:
                raise SchemaError(f"{path}: integer ranges need 2..{MAX_LEVELS} values (got {lo}..{hi})")
            levels = sub.get("x-levels") or [str(i) for i in range(lo, hi + 1)]
            if len(levels) != hi - lo + 1:
                raise SchemaError(f"{path}: x-levels must have exactly {hi - lo + 1} entries")
            questions[path] = {"type": "score", "instructions": desc or f"What is `{name}`, from {lo} to {hi}?", "criteria": list(levels)}
            plan[path] = {"kind": "integer", "minimum": lo}
        elif t == "array" and isinstance(sub.get("items"), dict) and isinstance(sub["items"].get("enum"), list):
            opts = [str(v) for v in sub["items"]["enum"]]
            rub = sub.get("x-descriptions") or {}
            for o in opts:
                questions[f"{path}[{o}]"] = {"type": "noul", "instructions": (desc + " " if desc else "") + f"Does `{name}` include \"{o}\"?" + (f" ({rub[o]})" if rub.get(o) else "")}
            plan[path] = {"kind": "multilabel", "options": opts, "threshold": float(sub.get("x-threshold", 0.5))}
        else:
            raise SchemaError(f"{path}: unsupported schema (use boolean, string+enum, string+x-candidates, integer+minimum/maximum, array of enum, or object)")
    return questions, plan


def _check_options(path: str, opts: list[str]) -> None:
    if len(opts) < 2:
        raise SchemaError(f"{path}: need at least two options")
    if len(opts) > MAX_LEVELS:
        raise SchemaError(f"{path}: at most {MAX_LEVELS} options are supported (got {len(opts)})")
    if len(set(opts)) != len(opts):
        raise SchemaError(f"{path}: options must be unique")


def assemble(plan: dict[str, dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    """Plain JSON in the schema's shape, from the typed answers (pydantic models or dicts)."""
    value: dict[str, Any] = {}
    for path, spec in plan.items():
        kind = spec["kind"]
        if kind == "boolean":
            v = _get(answers[path], "noul") >= 0.5
        elif kind in ("enum", "candidate"):
            v = _get(answers[path], "choice")
        elif kind == "integer":
            v = spec["minimum"] + int(round(_get(answers[path], "score")))
        elif kind == "multilabel":
            v = [o for o in spec["options"] if _get(answers[f"{path}[{o}]"], "noul") >= spec["threshold"]]
        else:  # pragma: no cover
            raise SchemaError(f"unknown plan kind {kind!r}")
        _set(value, path, v)
    return value


def _get(answer: Any, field: str) -> Any:
    return answer[field] if isinstance(answer, dict) else getattr(answer, field)


def _set(obj: dict[str, Any], path: str, v: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        obj = obj.setdefault(p, {})
    obj[parts[-1]] = v
