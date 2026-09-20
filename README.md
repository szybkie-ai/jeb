# JEB

A System One-compatible decision API on top of open LLMs: send a *state* and a map of typed
*questions* (`noul` yes/no, `choice`, `score`), get back one typed answer per question with
calibrated probabilities — from a single forward pass, no text generation.

```bash
pip install git+https://github.com/szybkie-ai/jeb
jeb serve --engine vllm-http --engine-url http://127.0.0.1:8021 --engine-model openjev-base
curl -X POST localhost:8020/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for order A-104. Please refund the duplicate.",
  "model": "jeb-latest",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
    "anger": {"type": "score", "instructions": "How angry is the customer?",
              "criteria": ["Calm", "Annoyed", "Furious"]}
  }}'
```

The wire format is 1:1 with the System One API, so existing clients work by pointing their base
URL here (`typesafe-sdk`: `TYPESAFE_BASE_URL=http://localhost:8020`). The server answers to the
model ids `jeb-latest`, `openjev-latest` and `jev-latest` by default (`JEB_MODEL_ALIASES`).

Settings are read from `JEB_*` environment variables (or the `jeb serve` flags); the pre-rename
`OPENJEV_*` names are still honoured as a fallback.

## How it works

Each question becomes a prompt in the model's chat format (thinking disabled) that ends right
where the next token is an answer label (`A`…`Z`, `Yes`/`No`). The engine — stock vLLM over its
OpenAI-compatible API — returns the log-probabilities of exactly those label tokens; renormalising
them is the answer distribution. All questions of a request share the tokenized `[system][state]`
prefix and go to the engine as one batch, so extra questions cost only their own tokens.

Details: the technical report in `docs/report.md`. Weights: https://huggingface.co/szybkie-ai. Site: https://www.szybkie.ai.

**Research preview.** Models and numbers change between training rounds; the API is stable.

## Models

| model | base | weights | held-out accuracy (ECE) |
|---|---|---|---|
| JEB-35B-A3B | Qwen3.6-35B-A3B, MoE, ~3B active | `szybkie-ai/jeb-35b-a3b` | 0.878 (0.008) on 5,419 items from 20 sets; base 0.859 (0.054); hosted Jev on the same items 0.880 (0.014) |
| JEB-4B | Qwen3.5-4B, dense | `szybkie-ai/jeb-4b` | 0.825 (0.007) on the same items; base 0.804 (0.038) |
| GGUF (Q4_K_M, Q8_0) of the 35B | | `szybkie-ai/jeb-35b-a3b-gguf` | for llama.cpp / Ollama with the vision projector; not separately evaluated |

Raw probabilities, no calibration fit; both models trained on the same data; full tables in `docs/report.md`. Research models, no warranty: evaluate on your
own data and set your own thresholds before relying on them.

## Run it with Docker

`docker compose up` starts vLLM with the FP8 weights and the JEB server on port 8020 (one NVIDIA GPU, container toolkit installed).

## License

Code: MIT. Weights (`szybkie-ai/jeb-*` on Hugging Face): Apache-2.0, like the Qwen3.5/3.6 base models.

## Status

v0.1: prompt-only on `Qwen/Qwen3.5-4B`, served FP8 by stock vLLM, with a fitted `calibration.json`
(position priors + temperature, 2 rotated presentations per choice/score question) and cache-aligned
prefix padding. Measured on a DGX Spark sharing its GPU with another model, through an SSH tunnel:
54 ms per single question, 238 ms for a 16-question request. Held-out: SST-2 as a `noul` 92% /
ECE 0.024; AG News as a 4-way `choice` 82% / ECE 0.023; STS-B as a 6-level `score` ECE 0.062.
`jeb eval` and `jeb calibrate` reproduce these. Fine-tuned weights come next.

## Examples

`examples/doom/` — a System One agent for Doom (ViZDoom engine state → six typed questions per tick → controls),
with recordings that show every judgment. See its README.

## Evaluate and calibrate on your own data

```bash
# public sets (SST-2 -> noul, AG News -> choice, STS-B -> score), raw label logprobs per presentation
jeb eval --task agnews --url http://localhost:8020 -n 300 --permutations 4 --out results/agnews-fit.jsonl
jeb eval --task agnews --url http://localhost:8020 -n 300 --offset 300 --permutations 4 --out results/agnews-eval.jsonl
# fit position priors + temperature on one split, report the ablation on the other, write calibration.json
jeb calibrate --fit results/agnews-fit.jsonl --eval results/agnews-eval.jsonl --out calibration.json
jeb serve ... --calibration calibration.json
```

The records are plain JSONL (`kind`, `gold`, per-presentation `labels`/`targets`/`logprobs`), so any
labelled set of yours can be evaluated the same way by writing rows in that shape.

## Development

```bash
uv sync
uv run pytest
```
