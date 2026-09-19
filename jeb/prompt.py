"""Turn a System One request into label-scoring prompts.

Every question becomes one or more prompts (permutations) that end right where the model's
next token is the answer label. The engine returns log-probabilities for exactly the label
tokens at that position; `decide` turns them into typed answers.

Prompts are rendered client-side in the model's chat format and tokenized here, so the
engine receives token ids and never applies a template of its own. The prefix
`[system][state]` is tokenized once and shared by every question of a request, which is what
lets the engine's prefix cache work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jeb.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

TEMPLATE_VERSION = "qwen-chat-v1"

SYSTEM_PROMPT = (
    "You are a decision engine inside a software system. You will be shown a STATE and one "
    "QUESTION about it with labeled options. Judge the state carefully and answer with the "
    "single label of the best option. Output only the label."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_OPTIONS = len(LETTERS)
YES, NO = "Yes", "No"

# Qwen chat format with thinking disabled: the empty think block is exactly what
# `enable_thinking=False` renders, so the model is in its non-thinking regime.
_PREFIX = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n{state}\n\n"
_SUFFIX_END = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class HFTokenizer:
    """`tokenizers` wrapper; no torch, loads from the HF cache (offline-safe) or a local directory."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    @classmethod
    def load(cls, name_or_path: str) -> HFTokenizer:
        from tokenizers import Tokenizer as _T

        p = Path(name_or_path)
        if p.is_dir():
            return cls(_T.from_file(str(p / "tokenizer.json")))
        if p.is_file():
            return cls(_T.from_file(str(p)))
        from huggingface_hub import hf_hub_download

        return cls(_T.from_file(hf_hub_download(name_or_path, "tokenizer.json")))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


class LabelError(ValueError):
    pass


class Labeler:
    """Maps label strings to single token ids, refusing labels that are not one token."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._cache: dict[str, int] = {}

    def token_id(self, label: str) -> int:
        if label not in self._cache:
            ids = self.tokenizer.encode(label)
            if len(ids) != 1:
                raise LabelError(f"label {label!r} is {len(ids)} tokens, need exactly one")
            self._cache[label] = ids[0]
        return self._cache[label]


def render_text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


@dataclass
class Plan:
    """One prompt whose next token answers one question under one option ordering."""

    qid: str
    kind: str  # noul | choice | score
    prompt_ids: list[int]
    prefix_len: int
    label_ids: list[int]
    targets: list[str]  # what each label maps back to: option key, level index, or "yes"/"no"
    text: str = field(repr=False, default="")
    labels: list[str] = field(default_factory=list)  # the label strings, aligned with label_ids
    system: str = field(repr=False, default="")  # the same prompt as chat messages, for engines that take images
    user_text: str = field(repr=False, default="")


def _option_lines(labels: list[str], keys: list[str], descs: list[Any]) -> str:
    lines = []
    for lab, key, desc in zip(labels, keys, descs):
        lines.append(f"{lab}) {key} — {render_text(desc)}" if desc not in (None, "") else f"{lab}) {key}")
    return "\n".join(lines)


def choice_variants(n: int, k: int) -> list[list[int]]:
    """Cyclic rotations of option order: the label a given option gets changes each time."""
    k = max(1, min(k, n))
    return [[(i + r) % n for i in range(n)] for r in range(k)]


def score_variants(n: int, k: int) -> list[tuple[bool, int]]:
    """(reversed?, alphabet shift). Order carries meaning, so never shuffle; reverse and shift instead."""
    out: list[tuple[bool, int]] = [(False, 0)]
    if k >= 2:
        out.append((True, 0))
    if k >= 3 and n + 1 <= MAX_OPTIONS:
        out.append((False, 1))
    if k >= 4 and n + 1 <= MAX_OPTIONS:
        out.append((True, 1))
    return out


PAD_TOKEN_ID = 198  # "\n" in Qwen3.5's vocabulary


def prompt_count(req: SystemOneRequest, permutations: int) -> int:
    """How many prompts a request expands to (nouls 1, choices/scores one per presentation)."""
    n = 0
    for q in req.questions.values():
        if isinstance(q, NoulQuestion):
            n += 1
        elif isinstance(q, ChoiceQuestion):
            n += len(choice_variants(len(q.criteria), permutations))
        else:
            n += len(score_variants(len(q.criteria), permutations))
    return n


@dataclass
class PrefixInfo:
    tokens: int  # real tokens of the shared [system][state] prefix
    padded: int  # its length as sent, including image tokens and cache-alignment padding
    extra: int = 0  # image tokens the engine adds inside the prefix (learned per image size)

    @property
    def pad(self) -> int:
        return self.padded - self.tokens - self.extra


PAD_TEXT = " ."  # one token per repetition on Qwen tokenizers and safe at both boundaries (checked by `pad_text_for`)


def pad_text_for(tokenizer: Tokenizer) -> str | None:
    """A string that always tokenizes 1:1 with this tokenizer, for padding rendered text (the image path), else None."""
    for cand in (PAD_TEXT, " |", " ~"):
        if all(len(tokenizer.encode(cand * k)) == k for k in (1, 2, 5, 17, 40)) and len(tokenizer.encode("end\n\n" + cand * 3 + "QUESTION")) == len(tokenizer.encode("end\n\n")) + 3 + len(tokenizer.encode("QUESTION")):
            return cand
    return None


def build_plans(
    req: SystemOneRequest,
    labeler: Labeler,
    permutations: int,
    pad_block: int = 0,
    pad_token_id: int = PAD_TOKEN_ID,
    extra_prefix_tokens: int = 0,
    pad_text: str | None = None,
) -> tuple[list[Plan], PrefixInfo]:
    """Return (plans, prefix info). Raises LabelError / ValueError on unservable questions.

    `pad_block` > 0 pads the shared prefix with newline tokens up to a multiple of that many tokens,
    so an engine whose prefix cache works in fixed blocks (528 for hybrid Qwen3.5 in vLLM) can
    reuse the state across the questions of a request even when the state is short.
    """
    tok = labeler.tokenizer
    user_prefix = f"STATE:\n{render_state(req.state)}\n\n"
    prefix = _PREFIX.format(system=SYSTEM_PROMPT, state=render_state(req.state))
    prefix_ids = tok.encode(prefix)
    info = PrefixInfo(tokens=len(prefix_ids), padded=len(prefix_ids))
    # Padding costs `n_pad` extra prefill tokens once and saves the prefix for every further prompt of the request
    # (`(prompts - 1) * prefix` tokens), so it only pays when the second is larger than the first.
    n_prompts = prompt_count(req, permutations)
    info.extra = extra_prefix_tokens
    info.padded = len(prefix_ids) + extra_prefix_tokens
    shared = len(prefix_ids) + extra_prefix_tokens  # what the engine sees before the question: text (+ image tokens)
    if pad_block > 0 and shared % pad_block and n_prompts >= 2:
        n_pad = pad_block - shared % pad_block
        if n_pad < (n_prompts - 1) * shared:
            if extra_prefix_tokens and pad_text:
                # Image path: the engine renders text, so pad with a 1:1 string after the state (still inside the prefix).
                user_prefix = user_prefix + pad_text * n_pad
                info.padded = shared + n_pad
            elif not extra_prefix_tokens:
                prefix_ids = prefix_ids + [pad_token_id] * n_pad
                prefix = prefix + "\n" * n_pad
                info.padded = len(prefix_ids)
    plans: list[Plan] = []

    for qid, q in req.questions.items():
        instr = render_text(q.instructions)
        if isinstance(q, NoulQuestion):
            body = f"QUESTION: {instr}\n"
            if q.criteria and (q.criteria.true or q.criteria.false):
                body += f"{YES} means: {q.criteria.true or 'the statement holds'}\n{NO} means: {q.criteria.false or 'it does not'}\n"
            body += f"Answer {YES} or {NO} only."
            plans.append(_plan(qid, "noul", prefix, prefix_ids, body, [YES, NO], ["yes", "no"], labeler, user_prefix))

        elif isinstance(q, ChoiceQuestion):
            keys = list(q.criteria)
            if len(keys) > MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: at most {MAX_OPTIONS} options are supported, got {len(keys)}")
            for order in choice_variants(len(keys), permutations):
                okeys = [keys[i] for i in order]
                labels = list(LETTERS[: len(okeys)])
                body = (
                    f"QUESTION: {instr}\nOPTIONS:\n{_option_lines(labels, okeys, [q.criteria[k] for k in okeys])}\n"
                    f"Answer with the label of the best option only."
                )
                plans.append(_plan(qid, "choice", prefix, prefix_ids, body, labels, okeys, labeler, user_prefix))

        elif isinstance(q, ScoreQuestion):
            levels = q.criteria
            if len(levels) > MAX_OPTIONS:
                raise ValueError(f"question {qid!r}: at most {MAX_OPTIONS} levels are supported, got {len(levels)}")
            for reverse, shift in score_variants(len(levels), permutations):
                idx = list(range(len(levels)))
                if reverse:
                    idx.reverse()
                labels = list(LETTERS[shift : shift + len(levels)])
                lines = "\n".join(f"{lab}) {render_text(levels[i])}" for lab, i in zip(labels, idx))
                body = f"QUESTION: {instr}\nLEVELS (ordered):\n{lines}\nAnswer with the label of the level that fits best."
                plans.append(_plan(qid, "score", prefix, prefix_ids, body, labels, [str(i) for i in idx], labeler, user_prefix))
        else:  # pragma: no cover
            raise ValueError(f"unknown question type for {qid!r}")
    return plans, info


def _plan(qid: str, kind: str, prefix: str, prefix_ids: list[int], body: str, labels: list[str], targets: list[str], labeler: Labeler, user_prefix: str = "") -> Plan:
    suffix = body + _SUFFIX_END
    suffix_ids = labeler.tokenizer.encode(suffix)
    return Plan(
        system=SYSTEM_PROMPT,
        user_text=user_prefix + body,
        qid=qid,
        kind=kind,
        prompt_ids=prefix_ids + suffix_ids,
        prefix_len=len(prefix_ids),
        label_ids=[labeler.token_id(lab) for lab in labels],
        targets=targets,
        text=prefix + suffix,
        labels=list(labels),
    )
