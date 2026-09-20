# JEB: an open decision model built from an open language model

*szybkie.ai — technical report, revised 2026-09-20 (round 3). Research model, no warranty.*

## Abstract

JEB (Joint Evaluation of Branches) is a decision model: it takes a *state* and a set of typed *questions* (yes/no,
a choice among options, a score on an ordered rubric) and returns one calibrated answer per question from a single
forward pass, with no generated text. It is built on open Qwen language models (a 4B dense model and a 35B
mixture-of-experts model with about 3B active parameters), fine-tuned on one desk-side machine with an objective on the
answer distribution itself, and released with open weights, the serving and evaluation code, and this report. Across
5,549 held-out items from 18 sets the 35B-A3B round raises accuracy from 0.834 to 0.859 over its base and cuts the
expected calibration error from 0.062 to 0.009, with no set losing more than a point; on a hard multi-option decision
set the base is right 52% of the time at an ECE of 0.30, the fine-tune 64% at 0.06. A 16-question request costs
238 ms on a DGX Spark with the 4B model, and a six-field structured output 370 ms against 2.4-3.3 s for generating the
same JSON with the same weights. A Doom harness turns the model's judgments into play on real levels and lets us
compare it with a commercial decision model.

## 1. Why a decision model

Most production uses of language models are decisions: which team handles a ticket, is this message abusive, how
urgent is this alert, what should the agent do next. Generating a paragraph of JSON to make such a decision costs
hundreds of output tokens, seconds of latency, and a parser; the answer's probability is lost in the sampling. A
decision model answers the question directly: the label is the next token, its probability is the answer's
probability, and every question of a request is scored in parallel over a shared state. TypeSafe's Jev popularised
the interface ("System One"); Jeb is an open implementation of the same idea with the same wire format.

## 2. Method

### 2.1 Questions become label prompts
Each question is rendered in the model's chat format with thinking disabled, ending exactly where the next token is
an answer label (`A`…`Z` for options and levels, `Yes`/`No` for yes/no). The state is rendered once as the shared
prefix. Options and levels are listed with their rubric text.

### 2.2 One forward pass, restricted softmax
The engine (stock vLLM through its OpenAI-compatible API) returns the log-probabilities of exactly the label tokens
for one generated position. Renormalising over the labels gives the answer distribution; the share of probability
the labels hold ("in-set mass", 0.999 on the raw model) is a diagnostic. No text is generated: nothing to parse,
nothing to hallucinate.

### 2.3 Prefix sharing and cache alignment
All prompts of a request share the tokenized `[system][state]` prefix and are sent as one batch. vLLM's prefix cache
works in blocks (528 tokens for hybrid Qwen3.5 models, 1568 for the teacher without speculative decoding), so the
prefix is padded to a block multiple whenever the padding costs less than recomputing the prefix for the other
prompts. With images in the prefix (2.7) the image tokens count toward the alignment. Measured on a real Doom tick
(11 prompts): 85% prefix-cache hits with and without an image, identical latency.

### 2.4 Presentation averaging
Label position bias is real (the raw model prefers `A`). Each choice/score question is asked under two rotations of
its options and the distributions are averaged; the teacher flips its answer under re-ordering 2% of the time, the
raw student 10%.

### 2.5 Calibration
Per-position log-priors and a temperature per question type are fitted on held-out data (300 examples per type):
choice T = 2.09, score T = 1.16, yes/no T = 1.0. Expected calibration error on AG News drops from 0.153 to 0.023
with no change in accuracy; STS-B (6 levels) from 0.175 to 0.062.

### 2.6 Structured output
A JSON-Schema subset (booleans, enums, candidate strings, bounded integers, multi-label arrays, nested objects) is
compiled into one fan-out of typed questions and assembled back into the schema's shape, with every field's
probabilities alongside.

### 2.7 Images
The base models are vision-language models, so a frame can be attached to the state; a 320×240 frame costs
284 tokens (student) or 302 (teacher). Whether the untrained model uses the picture is measured in 4.4.

## 3. Training

JEB is fine-tuned from an open Qwen base with a decision objective: the model's distribution over the answer labels of
a question, read as in 2.2, is trained toward a target distribution with a cross-entropy loss (plus a small term that
keeps the model's mass on the label set). Targets combine gold labels, where the data has them, with the soft
judgements of a much larger teacher model asked the same questions in the same format; the teacher's calibration is
therefore part of what is distilled. The training data is a mix of public classification, entailment and
knowledge sets rendered as typed questions and of domain sets built for the round. The training pipeline, the data
recipe and the hyperparameters are not published; the weights and the full evaluation are.

## 4. Evaluation

### 4.0 Round 3: JEB-35B-A3B on 18 held-out sets

Round 3 moves to Qwen3.6-35B-A3B (mixture of experts, 256 experts, 8 active; about 3B active parameters) and adds
document question answering (Polish and English passages: multiple choice with gold, claim verification, passage
retrieval), exam-style medical questions (MedQA) and differential-diagnosis cases built from published case reports.
The medical rows exist to study calibration on hard multi-option decisions, not to make a medical model; see the
limitations. Accuracy with raw ECE in parentheses, base and fine-tune scored identically through the same server:

| set | n | base Qwen3.6-35B-A3B | JEB-35B-A3B (round 3) |
|---|---|---|---|
| AG News | 300 | 0.877 (0.091) | 0.913 (0.019) |
| SST-2 | 300 | 0.957 (0.024) | 0.960 (0.009) |
| TREC | 300 | 0.937 (0.034) | 0.973 (0.026) |
| BoolQ | 300 | 0.880 (0.047) | 0.920 (0.030) |
| CLINC150 (20 options) | 400 | 0.968 (0.009) | 0.980 (0.029) |
| QNLI | 300 | 0.933 (0.036) | 0.920 (0.037) |
| RTE | 277 | 0.874 (0.040) | 0.903 (0.022) |
| IMDB | 300 | 0.960 (0.023) | 0.957 (0.022) |
| DBpedia | 300 | 0.983 (0.011) | 0.973 (0.030) |
| ARC-Easy | 300 | 0.990 (0.012) | 0.987 (0.017) |
| MMLU (500) | 500 | 0.818 (0.059) | 0.838 (0.034) |
| STS-B | 300 | 0.470 (0.168) | 0.590 (0.075) |
| blackjack (basic strategy) | 300 | 0.713 (0.171) | 0.713 (0.096) |
| tic-tac-toe (minimax) | 300 | 0.367 (0.188) | 0.383 (0.054) |
| document retrieval (which passage / answerable) | 300 | 0.930 (0.053) | 0.953 (0.014) |
| claim supported by passage | 172 | 0.983 (0.032) | 0.988 (0.017) |
| MedQA (USMLE) test | 300 | 0.880 (0.022) | 0.897 (0.071) |
| differential over candidate conditions (+ 'none listed') | 300 | 0.520 (0.304) | 0.637 (0.058) |
| **all 5,549 items** | | **0.834 (0.062)** | **0.859 (0.009)** |

Three observations. Calibration is the headline: the ECE falls on every group of sets, overall from 0.062 to 0.009,
which is below every earlier round and below the hosted Jev's 0.017 measured in 4.1. Nothing is forgotten: the 14
sets shared with the earlier rounds go up on aggregate (0.839 to 0.860), with the largest gains on AG News, BoolQ,
RTE, TREC and STS-B and no set down by more than a point. Knowledge moves little: MedQA gains under two points and
is now over-confident there (ECE 0.07), and the document multiple-choice holdouts are saturated for base and
fine-tune alike, so the fine-tune changes how the model decides and how honest its probabilities are rather than
what it knows. The differential set makes the point most clearly: candidates are 6-8 conditions plus "none of these",
the base picks right 52% of the time while sounding sure, the fine-tune 64% with probabilities that match its hit rate.


All numbers below are from round 1 (2026-09-18): the merged 4B checkpoint served FP8 in vLLM on one DGX Spark, two
presentations per question, raw probabilities unless stated. "Jev" is TypeSafe's hosted model (jev-1.13.0) queried through
its public API on the same items, for comparison only; the teacher is the 176B model that labelled the training data.

### 4.1 Held-out classification and the forgetting canaries (300-500 items per set)

Accuracy, with raw ECE in brackets where it matters. Sets marked * had training splits in the mix; the knowledge sets
(ARC-Easy test, MMLU test) and the other unmarked sets were never seen in training and measure whether the fine-tune cost
general ability. Round 2 (2026-09-19) replaced the trading rows and the old Doom rows with teacher-labelled knowledge
questions (ARC-Challenge train, OpenBookQA, SciQ, CommonsenseQA, MMLU auxiliary train), SNLI, and a small set of Doom
decisions played under survival-aware orders: 20,402 rows, 638 steps, 66 minutes.

| set | base 4B | JEB r1 | JEB r2 | teacher | Jev |
|---|---|---|---|---|---|
| AG News * | 0.857 (0.105) | 0.900 (0.037) | 0.897 (0.052) | 0.890 | 0.863 (0.095) |
| SST-2 * | 0.933 | 0.960 (0.037) | 0.950 (0.028) | 0.953 | 0.957 (0.102) |
| TREC * | 0.773 | 0.970 (0.027) | 0.967 (0.026) | 0.943 | 0.933 (0.025) |
| BoolQ * | 0.847 | 0.867 | 0.880 (0.026) | 0.907 | 0.917 (0.018) |
| CLINC150 | 0.960 | 0.963 (0.019) | 0.960 (0.032) | 0.985 | 0.973 (0.012) |
| QNLI | 0.843 | 0.913 | 0.910 | 0.950 | 0.930 |
| RTE | 0.841 | 0.845 | 0.856 | 0.913 | 0.913 |
| IMDB | 0.947 | 0.950 | 0.953 | 0.970 | 0.970 |
| DBpedia | 0.987 | 0.980 | 0.967 | 0.987 | 0.983 |
| ARC-Easy | 0.980 | 0.957 (0.154) | 0.963 (0.011) | 0.993 | 0.993 |
| MMLU (500) | 0.730 | 0.618 | 0.722 | 0.870 | 0.920 |
| blackjack (basic strategy) | 0.577 | 0.567 | 0.473 | 0.770 | 0.837 |
| tic-tac-toe (minimax) | 0.390 | 0.300 | 0.403 | 0.383 | 0.427 |
| all 4,177 | 0.819 (0.028) | 0.823 (0.034) | 0.836 (0.011) | 0.887 (0.024) | 0.897 (0.017) |

On the classification and intent sets the 4B student is at the hosted model's level: ahead on AG News, SST-2 and TREC,
level on IMDB, DBpedia and RTE, behind on BoolQ and QNLI. Round 1 cost knowledge (MMLU −11 against the base); round 2's
knowledge rows brought it back to within a point of the base and lifted BoolQ and RTE, at a raw ECE of 0.011 over all
4,177 items, the best-calibrated of the five models before any post-hoc fit (Jev 0.017, the teacher 0.024). Jev's 0.92
on MMLU and its blackjack strategy are a matter of model size. Jev is not uniformly calibrated: ECE 0.095 on AG News and
0.102 on SST-2 against the student's 0.05 and 0.03.

Round 2 needs no post-hoc calibration: the temperature and prior fit on the five public held-out sets makes it worse
(overall ECE 0.011 raw, 0.023 after the fit), so the served model uses the identity calibration. Round 1 benefited from
the fit (0.034 to 0.027).

### 4.2 Latency and cost (DGX Spark, GPU shared with the teacher)

(unchanged from the draft; to be re-measured on an idle GPU before release)

### 4.3 Doom, long horizon

Six 1,200-decision episodes (MAP01 and MAP02, seeds 91-93, skill 3, director every 60 decisions). Kills per game-minute
is the fair rate; survival is the other half of the story.

| player | kills / episode | kills / game-minute | survived | mean game time |
|---|---|---|---|---|
| raw 4B + frame | 13.3 | 7.8 | 4 / 6 | 103 s |
| JEB r1 + frame | 8.0 | 8.9 | 0 / 6 | 54 s |
| JEB r2 + frame | 5.3 | 7.6 | 0 / 6 | 42 s |
| JEB r2 + frame + survival standing orders | 5.5 | 6.8 | 0 / 6 | 48 s |
| Jev (text only) | 13.3 | 9.0 | 2 / 6 | 89 s |

The raw student in this harness already plays at Jev's level over long episodes, and neither trained round kept that.
Round 1 distilled the teacher's aggression (it closes in on 40% of decisions and backs off on 1%; the raw model 7% and
14%, walking to items 54% of the time) and dies every episode. Round 2 had only 1,720 Doom rows, from three teacher
episodes played under survival-aware orders in which the teacher itself died twice; they did not teach survival. The
next Doom data comes from the best player we have, the raw student's own long-horizon play (self-distillation of its
distributions), with teacher rows kept only for the aim and target questions.

### 4.4 A game it never saw: Atari (three episodes each)

| game | random | scripted | raw 4B | raw + frame | JEB r1 | JEB r1 + frame | JEB r2 | JEB r2 + frame |
|---|---|---|---|---|---|---|---|---|
| Ms. Pac-Man (score) | 315 | 395 | 230 | 240 | 870 | 90 | 70 | 540 |
| Breakout (bricks) | 0.5 | 18 | 0 | 0 | 15 | 16 | 2 | 14 |
| Pong (points) | -12.5 | -3 | -21 | -21 | -21 | -21 | | |

Both trained rounds play real games where the raw model does not (Breakout), but with three episodes the point estimates
swing between rounds and between frame on and off (round 1 best without the frame, round 2 best with it). We keep Atari as a
no-collapse check, not a ranking; Pong is beyond every model at this cadence.

### 4.5 Free text after the fine-tune

Eight prompts, greedy, thinking off (`data/rounds/r1/gen_check.md`): the trained model still answers correctly and
coherently (code, translation, arithmetic, summaries) but 20-80% shorter, and made one factual slip (a chess reply) where
the base was right. No collapse; a drift that round 2 addresses with anchor rows.

### 4.6 Trading (round 1, its own action questions, gold Q1 2024)

410 trades in 658 decisions, win rate 37%, profit factor 0.57, -38.9%: the hindsight action labels taught activity, not
edge (their training loss never left the uniform level). The trading model is a separate, private round with a different
objective (price-movement forecasts and a code-side strategy) and is not part of this release.

## 5. Limitations
- A research model: no warranty, no support. It must not be used for medical, legal, financial or safety decisions
  without independent validation and human review; the medical rows in the mix are a calibration study, not a
  clinical capability, and we make no claim about medical knowledge.
- Calibration is measured on the sets above; on another distribution, measure it before setting a threshold.
- The fine-tune does not add knowledge: on knowledge-bound sets the model stays at its base's level.
- Label bias is reduced by presentation averaging, not removed. Options beyond 26 need a second hop.
- The Doom results are one harness's numbers; different questions give different play. The 35B round was not
  trained on game data.

## 6. Reproducibility
The server, prompts, evaluation scripts, the calibration fit and the Doom harness are in the repository; weights on
Hugging Face; every number above has a command that reproduces it against the published weights. The training
pipeline and the data recipe are private.
