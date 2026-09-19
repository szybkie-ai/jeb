"""Public labelled sets mapped onto the three primitives.

Each task yields items of (state, question, gold target) where the gold is expressed in the
answer's own target space: "yes"/"no" for nouls, the option key for choices, the level index
(as a string) for scores. Parquet files come from the Hugging Face Hub and are cached locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE = Path(os.environ.get("JEB_EVAL_CACHE", Path.home() / ".cache" / "jeb" / "evals"))

HF = "https://huggingface.co/datasets"
SOURCES = {
    "sst2": f"{HF}/stanfordnlp/sst2/resolve/main/data/validation-00000-of-00001.parquet",
    "agnews": f"{HF}/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet",
    "stsb": f"{HF}/nyu-mll/glue/resolve/main/stsb/validation-00000-of-00001.parquet",
}

AGNEWS = ["world", "sports", "business", "sci_tech"]
STSB_LEVELS = [
    "The two sentences are completely dissimilar",
    "The two sentences are not equivalent, but are on the same topic",
    "The two sentences are not equivalent, but share some details",
    "The two sentences are roughly equivalent, but some important information differs or is missing",
    "The two sentences are mostly equivalent, but some unimportant details differ",
    "The two sentences are completely equivalent, as they mean the same thing",
]


@dataclass
class Item:
    state: Any
    question: dict[str, Any]
    gold: str
    kind: str
    meta: dict[str, Any]


def _frame(task: str):
    import pandas as pd

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{task}.parquet"
    if not path.exists():
        pd.read_parquet(SOURCES[task]).to_parquet(path)
    return pd.read_parquet(path)


def load(task: str, n: int, seed: int = 0, offset: int = 0) -> list[Item]:
    """`n` items after skipping `offset` of a seeded shuffle, so fit and eval splits are disjoint."""
    df = _frame(task).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = df.iloc[offset : offset + n]
    items: list[Item] = []
    for _, r in rows.iterrows():
        if task == "sst2":
            items.append(
                Item(
                    state={"review": r["sentence"]},
                    question={"type": "noul", "instructions": "Is the sentiment of `review` positive?", "criteria": {"true": "The reviewer liked it", "false": "The reviewer disliked it"}},
                    gold="yes" if int(r["label"]) == 1 else "no",
                    kind="noul",
                    meta={},
                )
            )
        elif task == "agnews":
            items.append(
                Item(
                    state={"headline_and_lead": r["text"]},
                    question={
                        "type": "choice",
                        "instructions": "Which news category does `headline_and_lead` belong to?",
                        "criteria": {
                            "world": "International news, politics, conflict, government",
                            "sports": "Sports, athletes, matches, leagues",
                            "business": "Companies, markets, economy, finance",
                            "sci_tech": "Science, technology, software, research, space",
                        },
                    },
                    gold=AGNEWS[int(r["label"])],
                    kind="choice",
                    meta={},
                )
            )
        elif task == "stsb":
            items.append(
                Item(
                    state={"sentence_1": r["sentence1"], "sentence_2": r["sentence2"]},
                    question={"type": "score", "instructions": "How similar in meaning are `sentence_1` and `sentence_2`?", "criteria": STSB_LEVELS},
                    gold=str(int(round(float(r["label"])))),
                    kind="score",
                    meta={"similarity": float(r["label"])},
                )
            )
        else:
            raise ValueError(f"unknown task {task!r}; choose from {sorted(SOURCES)}")
    return items
