"""Wire format: 1:1 with the System One API (`POST /v1/systemone`).

Requests: a `state` (string, object or array), a `model`, and a map of typed `questions`.
Responses: one typed answer per question under the same key, plus token `usage`.
JEB extensions live under the optional `options` request field and the optional `debug`
response field; default responses are byte-compatible with the original API.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Text = str | dict[str, Any] | list[Any]
"""Instructions and criteria may be plain strings or JSON structure."""


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str | None = Field(default=None, description="What a yes (value near 1) means.")
    false: str | None = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: Text
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: Text
    criteria: dict[str, Text | None] = Field(description="option -> rubric description (or null)")

    @field_validator("criteria")
    @classmethod
    def _at_least_two(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(v) < 2:
            raise ValueError("a choice needs at least two options")
        if any(not k.strip() for k in v):
            raise ValueError("option keys must be non-empty")
        return v


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: Text
    criteria: list[Text] = Field(min_length=2, description="ordered level descriptions, lowest first")


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class Options(BaseModel):
    """JEB-only knobs. Absent = defaults; never required for compatibility."""

    model_config = ConfigDict(extra="forbid")

    permutations: int | None = Field(default=None, ge=1, le=8, description="option orderings averaged per choice/score question")
    calibration: Literal["raw", "calibrated"] = "calibrated"
    debug: bool = Field(default=False, description="return prompts, in-set mass and per-permutation distributions")


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: Text
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    options: Options | None = None
    images: list[str] | None = Field(default=None, max_length=4, description="JEB extension: images (data: or http(s): URLs) shown to a vision-language model alongside the state")


# --------------------------------------------------------------------------- answers


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Text]
    probabilities: dict[str, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    debug: dict[str, Any] | None = None


class StructuredRequest(BaseModel):
    """JEB extension: a JSON-Schema subset compiled to a fan-out of questions (see jeb.structured)."""

    model_config = ConfigDict(extra="ignore")

    state: Text
    model: str
    schema_: dict[str, Any] = Field(alias="schema")
    options: Options | None = None
    images: list[str] | None = Field(default=None, max_length=4)


class StructuredResponse(BaseModel):
    model: str
    value: dict[str, Any]
    fields: dict[str, Answer]
    usage: Usage
    debug: dict[str, Any] | None = None


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ListModelsResponse(BaseModel):
    models: list[ModelMetadata]


def error_body(error_type: str, message: str) -> dict[str, Any]:
    """The upstream API's error envelope: `{"detail": {"error_type", "message"}}`."""
    return {"detail": {"error_type": error_type, "message": message}}
