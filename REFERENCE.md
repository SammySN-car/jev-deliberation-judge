# Multi-Agent Deliberation Judge — Planning & Design Reference (Laya backend)

> **Purpose:** Pre-build technical reference covering System One decision models, calibration, ensemble aggregation, and local model serving. Material drawn from official TypeSafe docs and Laya/Hugging Face sources.
>
> **Positioning:** ML systems / evaluation project (calibration, abstention, ensemble measurement), not model training or architecture work. Backend choice (Laya) is an implementation detail; the `DecisionBackend` interface keeps the pattern portable (e.g. hosted Jev later).

> **Backend decision (locked):** **Laya** (convaiinnovations/laya) — local, free, Apache 2.0 — instead of paid hosted Jev. Theory sections on primitives/calibration/patterns still apply (same Choice/Score/Noul vocabulary); TypeSafe-specific pricing/bands become *reference baselines* we compare Laya against, not hard dependencies.

## Contents

1. [Project Vision & Architecture](#1-project-vision--architecture)
2. [Theory — System One & Jev](#2-theory--system-one--jev-reference-baseline)
   - 2b. [Backend — Laya](#2b-backend--laya-decided--api-verified-from-model-card)
3. [Theory — Three Primitives](#3-theory--the-three-primitives-deep-dive)
4. [Theory — State design](#4-theory--state-design-what-you-feed-every-juror)
5. [Theory — Jaggedness](#5-theory--jaggedness--failure-modes-jev-113)
6. [Theory — Official patterns](#6-theory--official-patterns-applied)
7. [Theory — Deliberation & aggregation](#7-theory--deliberation--aggregation)
8. [Reference — TypeSafe SDK](#8-reference-code--typesafe-python-sdk-official-shapes)
9. [Reference — Deliberation skeleton + worked example](#9-reference-code--deliberation-skeleton-design-sketch-not-final)
10. [Design decisions open](#10-design-decisions-still-open-decide-before-build)
11. [Milestones (detailed)](#11-milestones)
    - 11b. [Structure, deps, CLI, tests](#11b-project-structure-dependencies-cli--tests)
12. [Design rationale FAQ](#12-design-rationale-frequently-asked-questions)
    - 12b. [Gotchas](#12b-gotchas--common-mistakes)
13. [Source map](#13-source-map-official--fetched)

---



## 1. Project Vision & Architecture

### What we are building

A **no-frontend** Python library + CLI where multiple independent **Laya "jurors"** (local System-One decision model) evaluate the same input state, and code aggregates their answers into one final verdict. Runs locally; no API cost.

```
                    ┌─────────────────────────────┐
                    │         INPUT STATE         │
                    │  (text / JSON — same for     │
                    │        every juror)          │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │  JUROR A    │         │  JUROR B    │         │  JUROR C    │
   │ request #1  │         │ request #2  │         │ request #3  │
   │ different   │         │ different   │         │ different   │
   │ framing     │         │ framing     │         │ framing     │
   │ (blind to   │         │ (blind to   │         │ (blind to   │
   │  B and C)   │         │  A and C)   │         │  A and B)   │
   └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   AGGREGATOR (pure Python)  │
                    │  vote / weight / veto       │
                    │  + Noul/Choice disagreement │
                    │    detection                │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  VERDICT: label, confidence, │
                    │  dissent report, escalation  │
                    │  flag (human review)         │
                    └─────────────────────────────┘
```

### Why separate requests (not one fan-out)?

Official docs recommend packing many questions into **one** call (fan-out) — questions in one call are **independent and blind to each other**. That is *within-request* blindness.

Our project is **cross-request** deliberation: each juror is a *separate inference call* (local Laya) with a *different framing of the question*. No juror can ever see another's answer because they don't share a request. The aggregation layer is where "deliberation" happens — in code, not in the model.

**Key distinction to internalize:**

| Mechanism | Blindness scope | Who aggregates |
|---|---|---|
| Fan-out (official pattern) | Questions within 1 request | Your code (routing) |
| **This project** | **Jurors across N requests** | **Your code (voting/weight/veto)** |
| Self-consistency cookbook | Same question × N repeats | TypeSafe mean / SD |
| LLM panel (Conclave, etc.) | Different models debate in text | LLM-generated text |

### Prior art & differentiation

The blind-jury + vote/weight pattern is established prior art. Verified references:

| Project | What it already does |
|---|---|
| CouncilLogic | Blind heterogeneous model jury, Borda aggregation, abstentions |
| CodeJury | Independent LLM jurors on PRs, parallel, never see each other |
| llm-jury (mokhld) | Majority / weighted / Bayesian judges, confidence escalation |
| TruLens Jury | mean / median / majority / weighted over parallel LLM judges |
| cross-judge (PyPI) | Multi-vendor ensemble, majority/unanimous, Krippendorff α |
| CollabEval / Comet / ChatEval | Independent jurors → vote (academic + industry) |
| quorum-cal | **Same model, 3 different prompts as jurors** — retains independence (error corr. 0.12) |
| CouncilAgent / Agora / llm-council | fan-out → aggregate → confidence |

Elements specific to this project (jury pattern applied to System One / Laya, using calibrated-decision-model outputs):

1. Juror weights from **RLCD-calibrated probabilities** (not self-reports), pre-registered emitted-label probability scale
2. Status from published uncertainty bands, simulation-gated threshold lock + Clopper-Pearson bounds
3. **Cross-primitive dissent** (Noul vs Choice inside one juror) as first-class signal
4. **Veto via Noul gates** with polarity counterbalance, composed with panel vote
5. Ablation with **competence gate + matched-coverage baseline + three-valued verdict + power/MDE** on a non-generative backend — rigor pattern has precedent (Judge Knows When It Knows); applied here, not claimed as invented
6. Laya-specific **co-batching attention probe** (appendix; batching not a headline — sequential fine at ~33ms)

External motivation only: quorum-cal; Nine Judges Two Effective Votes (effective-vote bound); Judge Knows When It Knows (ICC ~1.9-2.6 of 16, INCONCLUSIVE tier). Full stack: no exact public match; ingredients and H0-like findings do — do not oversell.

---

## 2. Theory — System One & Jev (reference baseline)

### What System One is

- TypeSafe's API for **decision models** (announced 2026-09-15). Endpoint: `POST /v1/systemone`.
- **No text generation.** You send text; you get back decisions + calibrated probabilities. Output tokens are free.
- Jev is the flagship model: `jev-1.13.0`, aliases `jev-latest` / `jev-preview`.
- Pricing: **$42 / billion input tokens** = **$0.042 / million tokens** input. State is billed once per request; questions ride along cheaply.
- Rate limits: 250k tokens/sec, 1200 req/min (dynamic — may change).
- Context: **64k** total (state + all questions), **32k** for state + single longest question.
- Input: **text only** — string, JSON object, or array of text. Pre-process images/audio to text yourself.
- Primary language: English (best accuracy). Other languages handled but test first.

### How Jev is trained — RLCD

Three post-training paths (from the official AI primer):

1. **RLHF** — trains models to say what humans prefer. Risks: sycophancy, confident hallucinations, **mode dropping** (narrows output distribution to a favored style).
2. **RLVR** — verifiable rewards; strong at math; slow/expensive.
3. **RLCD (Reinforcement Learning for Calibrated Decisions)** — TypeSafe's path. Optimizes for: *decisions + probabilities where higher probability ⇒ higher chance of being correct*.

**Calibration:** across many predictions, answers assigned 0.8 should be correct ~80% of the time. This is a **group-level** statistical property, **not** a guarantee about any single answer. Juror weighting operates on group-level properties, not per-answer guarantees.

### Two interfaces

- **OpenRouter System One:** `POST https://openrouter.ai/api/v1/systemone`, namespace `typesafe/`, header `Authorization: Bearer <OPENROUTER_API_KEY>`.
- **Direct TypeSafe:** Python SDK `typesafe-sdk`, env `TYPESAFE_API_KEY`, optional `TYPESAFE_BASE_URL`. Client: `AsyncTypeSafeClient().system_one(state=..., questions={...})`.

Primary backend is **Laya** (Section 2b). The TypeSafe interfaces above are retained as reference for optional M8 comparison only.

---

## 2b. Backend — Laya (DECIDED — API verified from model card)

**Model:** [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) — Apache 2.0, local, free. Non-autoregressive System-One decision model (ModernBERT-large backbone, 421M params, English root checkpoint).

### Installation & load

```bash
pip install laya
# If transformers hangs on import (TF probe deadlock): export USE_TF=0
```

**Single-model mode** (one fixed checkpoint per call):

```python
import laya

agent = laya.load("convaiinnovations/laya")                          # English root (~808 MB)
# agent = laya.load("convaiinnovations/laya", subfolder="multilingual")   # 100+ langs, ~647 MB
# agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions") # fine-tuned workflows
```

**Route mode** (Router auto-selects checkpoint by script/language):

```python
from laya import Router
router = Router(preload=True)   # all checkpoints in RAM; sub-35ms routing
res = router.predict(state, questions)
res["routing"]["model"]  # 'english' | 'multilingual' | 'typed-decisions'
```

### Question dict schema (verified from model card)

```python
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund..."
}

questions = {
    "department": {
        "type": "choice",                    # 'choice' | 'score' | 'noul'
        "instructions": "Which department should handle this request?",
        "criteria": {                        # dict for choice; list for score; omitted for noul
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}
```

### Predict & result shape

```python
result = agent.predict(state, questions)   # ONE forward pass answers ALL questions
answers = result["answers"]

answers["department"]["choice"]   # -> 'billing'  (plus confidence field)
answers["urgency"]["score"]       # -> 1.84 / 2.0 (ordinal, fractional allowed)
answers["churn_risk"]["noul"]     # -> 0.892      (P(yes))
```

- **Batching:** every question in one call = one forward pass. 10 questions ~ 158.6 ms (T4), 72.3 ms (multilingual).
- **Budget:** English root = **512 tokens total** (`head_max_len=192` for options -> **~320 tokens left for state**). Multilingual/typed-decisions = 1024 (`head_max_len=256` -> ~768 state).
- **Cardinality limit:** high-option Choice (>20 labels) degrades sharply at default head budget (Banking77: Jev 0.870 vs Laya 0.425). Label sets should stay <= 8-10.

### Laya vs Jev — benchmarks (from model card; Jev figures third-party published)

| Metric | TypeSafe Jev 1.13.0 | Laya (routed) | Note |
|---|---|---|---|
| typed-decisions acc (2,000 decisions) | 0.727 | **0.766** | Laya fine-tuned checkpoint clears 0.735 teacher ceiling |
| AG News (4 labels) | 0.910 | **0.950** | |
| DAIR Emotion (6 labels) | 0.480 | **0.595** | Jev assigned 0 prob to true label on 16% of examples |
| Banking77 (72-77 labels) | **0.870** | 0.425 | Jev wins high-cardinality |
| ECE (lower = better) | 0.246 | **0.081** | Laya 3x better after temp scaling |
| p50 latency, 1 question | 236-276 ms | **32.8 ms** | 7.8x faster |
| Languages usable | not published | **45 / 51** | |
| Weights / cost | closed API, $0.042/Mtok | **Apache 2.0, $0 self-hosted** | |

### Honest Limits (model card — read before trusting probabilities)

1. **Base checkpoints are near-chance on typed-decisions zero-shot** (0.362 vs 0.461 majority baseline). The 0.766 belongs to `laya-typed-decisions` **fine-tuned on that benchmark train split**. Laya is a fast base to specialize, not a zero-shot decision engine for every domain.
2. **Ships over-confident:** raw mean ECE 0.466 -> 0.081 after refitting one temperature per (question type, option count). Temperature-scale on project fixtures before using probabilities as juror weights (M7 reliability diagrams).
3. **Ordinal `score` is the weakest primitive** (SST-5: 0.372). Noul/Choice preferred for juror votes; Score reserved for ordinal severity.
4. **High-cardinality Choice fails** — keep options small (see above).
5. **English only on root** — use `laya-multilingual` for non-English input.
6. **act_probability is not a usable gate signal** (card/GitHub: gate on confidence instead; reported AUROC ~0.77 for confidence). Do not wire `act_probability` into status logic.
7. **Noul can follow option-label wording rather than state** (strongest on English root); base checkpoints below majority-class zero-shot off-domain. Mitigations: polarity counterbalance (Section 7), competence precondition (Section 7 hypotheses).

### Temperature scaling (required before trusting weights)

Raw Laya outputs are over-confident (raw mean ECE ~0.466 on their benches). Method (M7):

1. Assemble labeled fixtures: for each (question type, option count), record model probability for the true outcome (Choice: p(true label); Noul: p if true=yes else 1-p).
2. Fit a single scalar T per bucket by minimizing NLL on the development split (standard temperature scaling; no bias term):

       p_calibrated = softmax(logits / T)    # Choice
       p_calibrated = sigmoid(logit(p) / T)  # Noul (logit = log(p/(1-p)))

   (If only probabilities are exposed, optimize T directly on p via the same NLL; equivalent to Platt-style scaling without intercept.)
3. Apply T at inference inside `JurorResult` weight derivation — aggregation reads calibrated probabilities only.
4. Report ECE before/after (reliability diagram bins of 10) in README.

Fit per bucket; do not share T across question types with different option counts.

### Design implications for us (updated with verified API)

| Implication | Action |
|---|---|
| Jurors = local `agent.predict()` calls, not paid API | Tests can hit Laya live; no token budget for unit tests (use fixtures for CI) |
| ~320 tokens of state (English root) | `state_builder` precomputes dates/counts and strips irrelevant text (M1) |
| Question schema = plain dict | `typesafe-sdk` not required for Laya path; `DecisionBackend` interface retained for optional Jev M8 |
| Over-confident output (raw ECE 0.466) | M7: fit temperature on labeled fixtures; record ECE before/after |
| Zero-shot weak outside fine-tuned domains | D1 domain should match base Laya competence, or include a fine-tune step (notebook on GitHub) |
| One forward pass answers all questions in a call | Within-juror fan-out adds no forward passes; panel of N jurors = N forward passes |
| TF import deadlock | Set `USE_TF=0` when `laya.load` hangs |

**Verify on your machine before M2 (remaining):** actual CPU latency with `Router(preload=True)` vs `laya.load`, RAM footprint, and whether `USE_TF=0` is needed in your env.

---
## 3. Theory — The Three Primitives (deep dive)

Every question in a request declares its `type`. Answers come back with the primitive's payload **plus** a probability.

> **Note:** examples below use TypeSafe SDK classes (`Noul`/`Choice`/`Score`) for clarity of the type model. On the Laya path the same questions are plain dicts with `"type"`, `"instructions"`, `"criteria"` keys (Section 2b). Semantics and answer fields are equivalent.

### Noul — binary yes/no

```python
from typesafe_sdk import Noul
question = Noul(instructions="The reviewer explicitly requests a refund")
# answer.noul → float in (0, 1); P(yes)
```

- Semantics: **probability that the statement is true.**
- Cannot abstain. 0.5 is max uncertainty, not "I don't know."
- Use for: flags, gates, containment checks, "does X satisfy criterion Y."

### Choice — up to 255 options

```python
from typesafe_sdk import Choice
question = Choice(
    instructions="Pick the single best category for this ticket",
    criteria={
        "bug_report": "Something is broken or erroring",
        "billing": "Charges, invoices, refunds",
        "feature_request": "User wants new functionality",
    },
)
# answer.choice → winning label (str)
# answer.probabilities → {label: p} (sums to ~1)
```

- Semantics: distribution over labels; `choice` = argmax.
- **Known jaggedness:** labels can still flip under repetition (cookbook: 2 of 8 questions flipped at least once over 15 repeats). Top-prob threshold of **0.60** recommended → else treat as `uncertain`.

### Score — ordered levels (2–10)

```python
from typesafe_sdk import Score
question = Score(
    instructions="How severe is the issue",
    criteria=[
        "Cosmetic; no functional impact",
        "Broken; workaround exists",
        "Blocking; no workaround",
    ],
)
# answer.score → float (ordinal position, can be fractional between levels)
```

- Semantics: **ordinal**, not cardinal. Distance between level 1→2 is not guaranteed equal to 3→4.
- Use for: severity, confidence-in-evidence, frustration, priority. **Do not** do arithmetic across different questions' scores as if they share a scale — combine in code with explicit rules.

### Confidence (Choice and Score only)

- Distinct from probability. Probability = distribution over outcomes. **Confidence** = how decisive/settled the model is about that distribution (TypeSafe's extra signal).
- Noul has **no** confidence field — only the probability.
- Official guidance: use confidence for **routing** (auto-act vs escalate), not as a substitute for probability.

### Cross-primitive disagreement (detected signal)

- **Noul says no (p < 0.5) but Choice picks the positive label** (or vice versa) → structural disagreement between framings.
- **Choice top-prob < 0.60** → the model itself is on the fence → escalate.
- **Jurors vote different labels** → dissent report.

---

## 4. Theory — State design (what you feed every juror)

State is the shared evidence. All jurors receive the **same state**; only the **questions/framings** differ.

Official rules for state:

- Text / JSON / array of text. Shape it for humans *and* for the model.
- Include: raw evidence, metadata, domain rules, boundary-case examples if relevant.
- **Strip adversarial / irrelevant content** — jaggedness docs: large irrelevant state and adversarial content hurt accuracy.
- **Keep arithmetic in code.** Jev is bad at math, numbers, date comparison. Compute totals/dates yourself; put *descriptions* in state.
- State is ingested **once** per request; questions evaluate against it in parallel.

**Design:** one `build_state(domain_object) -> str | dict` function; every juror call receives identical state. Juror diversity is confined to `questions`.

---

## 5. Theory — Jaggedness & failure modes (Jev 1.13)

From `model-jaggedness/jev-1.13` — things that make Jev wrong *systematically*:

1. **Literal reading** — misses sarcasm/irony/implicature.
2. **Math & numbers** — arithmetic in the prompt degrades answers; do math in code.
3. **Date comparison** — "is this overdue?" relative to *today* is unreliable; compute the date delta yourself and state it.
4. **Indirection** — pronouns/aliases across long state; prefer explicit nouns.
5. **Large irrelevant state** — accuracy drops as state grows; prune.
6. **Adversarial content** — prompt-injection-like text in state can skew answers; sanitize untrusted text.

**Design implication:** the aggregation layer runs *sanitization + precomputed facts* (dates, counts, sums) into state before juror fan-out, countering the documented failure modes above.

**Confound warning:** precomputing facts *removes* the jaggedness channels an H0 narrative would blame. If H0 wins with builder ON, attribute carefully; M7-core includes **builder-off ablation + error-overlap analysis** so mechanism claims are measured, not assumed.

---

## 6. Theory — Official patterns applied

### 6a. Composite scoring (one call, atomic questions)

Official pattern: **don't ask one broad Score.** Decompose into atomic Noul/Choice questions in a **single** request; combine scores **in code**.

Applied here: each juror runs a composite battery (3-5 atomic questions) rather than one broad Score. Aggregation occurs at two levels: within-juror (code) and across-jurors (code).

### 6b. Confidence routing

```
if confidence high and probability extreme → act
elif probability in uncertain band → human review
else → escalate / hold
```

Noul cookbook band: **0.30-0.70 = uncertain -> human review.** Choice: top-prob **< 0.60 = uncertain.** Bands are surfaced in the verdict object.

### 6c. Speculative fan-out (within a juror)

One request per juror may include speculative questions (severity only matters if category=bug). Questions are evaluated in parallel; irrelevant answers are ignored in code.

### 6d. Self-consistency (repeat & aggregate)

- **Noul cookbook:** 14 Noul questions × 15 repeats → TypeSafe per-question probability **SD = 0.0102** (lower than all LLM conditions tested).
- **Choice cookbook:** 8 questions × 15 repeats → still 2/8 label flips; use 0.60 threshold.

**Design takeaway:** Jev is stable on Noul (SD 0.0102), less stable on Choice (label flips observed). Noul votes can be weighted more heavily; Choice votes require probability thresholds. Optional mode: K repeats per juror, averaged (used for ablation in M7).

---

## 7. Theory — Deliberation & aggregation

Classic **ensemble / jury theory** applied to blind System One jurors.

### Ensemble variance reduction (hypothesis, not assumed result)

**Pre-registered decision rule (three-valued — never binary H1-vs-H0):**

| Outcome | Condition | README placement |
|---|---|---|
| H1 supported | Panel improves primary metric by >= declared delta, held-out, CI excludes 0 (or McNemar p<.05) | Standard results lead |
| H0 supported within delta | Panel does NOT reach +delta; CI upper bound < delta (panel cannot matter by declared margin) | **Leads abstract** if this is the result |
| Inconclusive | Neither: CI spans 0 to >= delta, or power insufficient | **Must say inconclusive** — does not count as H0 support |

**Declare before any labeled run:** primary metric (e.g. balanced accuracy or Youden's J on the inference set), **minimum detectable effect delta** (e.g. +8pp), and **required n for 80% power** at assumed discordance. Report achieved power/MDE in README even if <80%. n=50 stress fixtures alone: ~5% power for +4pp, ~17% for +8pp at 15% discordance — **insufficient for hypothesis testing**; they are descriptive only.

**H₁:** framings + calibrated weighting improve the primary metric by >= delta vs single-call baseline (matched conditions).
**H₀:** shared weights + framing insufficiency => gain < delta (panel cannot help by a meaningful margin).
**Competence precondition (gates interpretation):** if single-call accuracy <= majority-class baseline on the inference set, the result is **"backend not competent on D1"** — neither H1 nor H0 is scored. Checked and reported before panel claims.

External motivation only (not our result): Judge Knows When It Knows (ICC effective votes ~1.9-2.6 of 16); Nine Judges Two Effective Votes (~2.2 effective votes bound weighting).

Framings differ by construction (separate calls, different wording), which targets *framing* variance only. Documented jaggedness (math, dates, literal reading) hits every juror with the same weights — exactly where a panel is least likely to disagree on a wrong answer. M7 ablation decides; report the outcome even if H₀ wins.

**Caveat:** all jurors share the same model weights, so errors caused by *model-level* jaggedness (math, dates, sarcasm) are **correlated**. Ensembles reduce framing variance, not shared blind spots. Mitigation: precompute facts into state (Section 5).

**External vs own numbers (policy):** quorum-cal's ~0.12 / ~0.65 figures are **motivation from a different setup** — cite in background only, never in this project's results tables, never as an expected value for our pipeline. If we report inter-juror error correlation (φ / κ / pairwise), we re-derive it on our fixtures with a **bootstrap 95% CI**; with N jurors and limited fixtures the interval may be wide — report the interval, not a bare point estimate. **README acceptance:** if the CI spans a range that changes the qualitative story (e.g. includes both near-0 and high correlation), say so in plain text ("data insufficient to pin rho") instead of quoting the point estimate.

### Ensemble math (formulas used in code)

Notation: N jurors, labels L, juror i emits label y_i with weight w_i.

**Vote (plurality).** For each label l:

    votes(l) = sum_i 1[y_i = l]
    winner = argmax_l votes(l)   # tie -> uncertain (row 3 of status table)

**Weight (heuristic).** These are documented heuristics on top of calibration — **not** a claim of optimal Bayesian combination.

**Pre-registered primary (fixed before any fixture results are examined):**

| Vote source | Primary weight w |
|---|---|
| Choice label | calibrated **top-prob of the emitted label** (same quantity as max prob mass) |
| Noul-derived binary label | **max(p, 1-p)** — probability mass on the emitted side; same scale family as Choice top-prob (both in ~[0.5, 1] for decisive binary / 2-option cases) |

Do **not** mix top-prob (Choice) with abs(p-0.5)*2 (Noul) — incompatible scales make a 0.6 Choice juror dominate a 0.6 Noul juror unfairly. Primary weight = **emitted-label probability only** for every vote source.

**Secondary / exploratory only** (reported in appendix or not at all; never selected post-hoc as the headline): confidence field (Choice/Score if present), mean-pooling of Nouns, raw p without calibration. Choosing among modes after seeing fixture outcomes is disallowed (researcher degree of freedom). Headline ablations use primary weights only.

    w_i = top_prob_i                      # Choice: probability of chosen label
    w_i = |p_i - 0.5| * 2                 # Noul: distance from max-uncertainty, scaled to [0,1]
    w_i = confidence_i                    # Choice/Score confidence field (TypeSafe/Laya), if present

    weighted(l) = sum_i w_i * 1[y_i = l]
    winner = argmax_l weighted(l)

Optional normalize: `w_i / sum_j w_j` so weights sum to 1 (useful for reporting).

**Veto.** Given veto questions V with thresholds t_v:

    fired = any( juror_i answer for q in V has p_q >= t_v )
    if fired: status = escalate   # independent of winner

**Effective panel size (corrected).** With normalized weights alpha_i (sum=1):

    # Independence (rho=0) only — Herfindahl; equals N when equal weights:
    N_eff_indep = 1 / sum_i alpha_i^2

    # Equicorrelated errors (rho = pairwise error correlation):
    N_eff(rho) = 1 / ( rho + (1-rho) * sum_i alpha_i^2 )
    # equal weights: N_eff = N / (1 + (N-1)*rho)   [matches Wavering Oracles form]

Reporting rule: always report N_eff with an explicit rho assumption. Using 1/sum(alpha^2) alone silently assumes rho=0 — the case our own H0 denies. With N=3 and unknown rho, quote the formula and a rho sensitivity range, not a single point N_eff.

**Variance intuition (why panels help).** For a scalar score s_i with Var(s_i)=sigma^2 and pairwise corr rho:

    Var(mean) = sigma^2 * ( rho + (1-rho)/N )
    -> as N grows, only the (1-rho)/N term shrinks; rho is the floor.
Same weights => rho > 0 from shared jaggedness (Section caveat). Framing diversity targets the (1-rho) component only.

**Ablation metrics (M7 definitions):**

| Metric | Definition |
|---|---|
| Agreement rate | fraction of fixtures where panel winner == single-call winner |
| Flip rate | fraction where panel winner != single-call winner (1 - agreement) |
| Veto catch rate | fraction of labeled-veto fixtures where status = escalate |
| Latency delta | t_panel - t_single under same concurrency model |
| Cost delta | N_forward_passes panel vs 1 single (local: time only; hosted: tokens) |
| Escalation rate | fraction of fixtures with status in {human_review, escalate} |
| Accuracy on auto_act | correctness restricted to fixtures the system was willing to act on (pairs with escalation rate; see risk-coverage M7-stretch) |
| Always-act baseline | single call using the **pre-registered primary framing** (Juror A), calibrated — NOT a strawman and NOT post-hoc best-of-N; if best-of-N is shown, label it explicitly as oracle upper bound |

Compare at matched conditions: same state, same label set, same temperature (post-fit).

### Aggregation strategies

1. **Vote (majority / plurality)**
   - Each juror emits a label (Choice argmax, or Noul ≥ 0.5 → yes).
   - Winner = most votes. Tie → `uncertain`.
   - Simple, robust, easy to test.

2. **Weight (probability-weighted)**
   - Weight each juror's vote by its own confidence/probability (e.g., juror's top-prob, or |p − 0.5| for Noul).
   - Sum weighted votes per label; argmax.
   - Weights scale with juror decisiveness (confidence / |p-0.5|).

3. **Veto (unanimity / supermajority / poison pill)**
   - Any juror whose Noul for a *veto criterion* (e.g., "contains PII", "unsafe") exceeds threshold → whole panel escalates regardless of majority.
   - One qualifying objection overrides majority vote.
   - Configurable: `veto_questions` list with per-question thresholds.
- **Polarity counterbalance (noul label artifact):** Laya's Noul can follow option-label wording rather than state (strongest on English root). Across jurors, **flip yes/no polarity of shared-criterion Nouns** (juror A: "contains PII?"; juror B: "is free of PII?" inverted scoring) or use label overrides so no single polarity phrasing drives all vetoes. Same reason twin paper swapped presentation order.

### Disagreement detectors (verdict fields)

- **Noul vs Choice conflict** within one juror.
- **Cross-juror label spread** (e.g., 2–1–1 split).
- **Low aggregate confidence** (mean top-prob < 0.60).
- **High mean pairwise |delta-p|** across jurors on shared-criterion Nouns (auxiliary error-detection AUROC pre-registered).

Each detector sets `verdict.status ∈ {auto_act, human_review, escalate}` and populates `dissent[]`.

### Status decision table (priority order)

Evaluate top-down; first matching row wins.

| # | Condition | Status | Notes |
|---|---|---|---|
| 1 | Any veto question fires (Noul >= its threshold, default 0.70) | `escalate` | Overrides vote/weight result |
| 2 | Any cross-primitive conflict (Noul vs Choice within a juror) | `human_review` | Structural disagreement |
| 3 | Juror label spread has no strict majority — at N=3 this is **only the 1-1-1 case** (a 2-1 split IS a strict majority); at N>3 any label with votes < ceil(N/2)+... see note | `human_review` | Fragmentation; **threshold locked only after escalation-rate simulation (M5/M7)** |
| 4 | Aggregate confidence below floor (mean top-prob < 0.60, or mean Noul in 0.30-0.70) | `human_review` | Uncertainty band |
| 5 | High **mean pairwise abs delta-p** on shared-criterion Nouns across jurors (not sample SD: N=3 => 2 df; wording confounds) | `human_review` | Deterministic-Laya dispersion; **auxiliary pre-reg: AUROC of this stat for error vs single-call confidence baseline** — report even if ~0.5 |
| 6 | Strict majority AND aggregate confidence above floor AND no conflicts | `auto_act` | |
| 7 | (fallback) | `escalate` | Unreachable if rows 1-6 exhaustive; defensive default |

`dissent[]` is populated whenever rows 2, 3, or 5 fire (strings describing which juror/condition). Row 1 also records which veto question fired.

**Determinism note (Laya):** `predict()` is a deterministic encoder forward pass — no token sampling, no documented inference noise. Calling `predict()` twice on identical state+questions returns identical outputs; **K>1 "self-consistency" repeats on Laya have SD ≡ 0** and must not be presented as a variance mechanism (pseudo-rigor). Implications:
- Row 5 uses **cross-juror** spread across different framings (jurors are the diversity), not repeat-SD.
- `--repeats K` (M8) exists only to (a) **measure the sequential-vs-sequential noise floor** for epsilon calibration (GPU nondeterminism if any), or (b) run on a **stochastic backend** (e.g. hosted Jev if nondeterministic). On Laya, K>1 is a diagnostic, not self-consistency.
- Interview line: *dispersion = mean pairwise |delta-p| across framings on shared-criterion Nouns (auxiliary error-detection AUROC pre-registered); identical reruns are ~0 on CPU (exactly 0) and ~1e-4 on GPU — not a self-consistency mechanism.*
- Follow-up ready: "does it predict error?" → answered by the pre-registered AUROC, not by asserting SD works.

**Threshold discipline:** the table encodes *candidate* rules with default numbers (veto 0.70, band 0.30-0.70, top-prob 0.60). Before locking defaults, simulate the full table on labeled fixtures and record **escalation rate** (fraction of fixtures ending `human_review`/`escalate`) alongside accuracy-on-auto-act. If escalation rate is pathologically high (automation goal defeated) or accuracy-on-auto-act is not better than always-act, retune — do not ship a priori thresholds untouched. Escalation rate is a first-class M7 metric.

### Label normalization (required for vote/weight)

Jurors must share one **panel label vocabulary** or votes cannot be tallied.

- Panel defines `LABELS: list[str]` (the Choice `criteria` keys used by every juror).
- Each juror's Choice question uses the **same keys**; only `instructions`/criteria *descriptions* differ per framing.
- Normalization step: `juror_label = answers[qid]["choice"]` mapped through `LABELS`; unknown label -> flag + treat as abstain for that juror.
- Noul-based jurors (binary framings) map `p >= 0.5` -> positive label, else negative label, into the same two-element vocabulary when the panel is binary.

Without shared keys, plurality voting is undefined.

---

## 8. Reference Code — TypeSafe Python SDK (official shapes)

```python
# pip install typesafe-sdk
# export TYPESAFE_API_KEY=...

from typesafe_sdk import AsyncTypeSafeClient, Noul, Choice, Score

async def one_juror(state: str | dict, questions: dict):
    async with AsyncTypeSafeClient() as client:
        response = await client.system_one(
            model="jev-latest",       # pin jev-1.13.0 in prod if thresholds tuned
            state=state,
            questions=questions,
        )
    return response.answers

# Minimal single-question examples
NUl_Q = {"refund_requested": Noul(instructions="User explicitly asks for a refund or credit")}
CHOICE_Q = {"category": Choice(instructions="Broad category", criteria={
    "bug_report": "Broken or erroring",
    "billing": "Charges, invoices, refunds",
    "feature_request": "New functionality wanted",
})}
SCORE_Q = {"severity": Score(instructions="How severe", criteria=[
    "Cosmetic", "Degraded with workaround", "Blocking",
])}
```

**HTTP shape (if bypassing SDK):**

```http
POST /v1/systemone
Authorization: Bearer $TYPESAFE_API_KEY
Content-Type: application/json

{
  "model": "jev-latest",
  "state": "...text or JSON...",
  "questions": {
    "q1": {"type": "noul", "instructions": "..."},
    "q2": {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}}
  }
}
```

Response: `answers.<qid>.noul | .choice/.probabilities | .score`, plus model id, usage (input tokens only billed).

**OpenRouter variant:** same body; base `https://openrouter.ai/api/v1/systemone`; model `typesafe/jev-latest`.

---

## 9. Reference Code — Deliberation skeleton (design sketch, not final)

```python
from dataclasses import dataclass, field

@dataclass
class Juror:
    name: str
    questions: dict          # unique framing — different instructions/criteria
    # optional: repeats: int = 1

@dataclass
class JurorResult:
    name: str
    answers: dict
    label: str | None        # normalized to panel label space
    weight: float            # derived from confidence / |p-0.5|
    flags: list[str] = field(default_factory=list)

@dataclass
class Verdict:
    status: str              # auto_act | human_review | escalate
    winning_label: str | None
    tally: dict[str, int]
    weighted: dict[str, float]
    dissent: list[str]
    juror_results: list[JurorResult]

def run_panel(state, jurors: list[Juror], agent, strategy: str = "weight") -> Verdict:
    # 1. Run N separate predict() calls — jurors are blind to each other
    #    Laya predict() is SYNC and CPU/GPU-bound; see concurrency note below
    # 2. Normalize each juror's answer -> label + weight
    # 3. Run disagreement detectors (Noul/Choice conflict, veto questions, thresholds)
    # 4. Aggregate via strategy: vote | weight | veto
    # 5. Map aggregate + detectors -> status (decision table in Section 7)
    ...

# Juror calls (pick one concurrency model — see note):
#   A) Sequential (simplest, N x latency)
#   B) asyncio.to_thread(predict, ...) + asyncio.gather  — offloads blocking call
#   C) ThreadPoolExecutor / ProcessPoolExecutor           — true parallelism if model supports it
#   D) Batched: one agent.predict(state, merged_questions) then split answers by juror
#      (single forward pass; jurors still blind if question dicts never cross-reference)
```

**Concurrency (Laya-specific):** `agent.predict()` is a synchronous, blocking forward pass. Naive `asyncio.gather(predict(...), ...)` does **not** parallelize it — coroutines block the event loop and run serially.

| Model | Mechanism | Latency | Notes |
|---|---|---|---|
| Sequential | for-loop over jurors | N x t_predict | simplest; fine for N=3, local ~33ms each |
| Thread offload | `asyncio.gather(*[asyncio.to_thread(agent.predict, state, j.questions) for j in jurors])` | ~t_predict if GIL released in torch forward | verify torch releases GIL in your build |
| Process pool | `ProcessPoolExecutor` (model reloaded per worker or shared via fork) | ~t_predict | highest RAM cost; pickling state only |
| Batched questions (APPENDIX / optional) | merge juror question dicts into one `predict` | ~1 t_predict if invariance holds | **Not a differentiator; not on critical path.** At ~33ms sequential, batching buys ~nothing. Option markers likely share one encoder pass => questions may attend to each other; TypeSafe blindness claim **unverified for Laya**. Run cheap probe first (Q1 alone vs Q1+Q2): if any label flip or status-relevant delta, batching stays off permanently. Known general phenomenon: co-batching/kernel invariance. |

**Recommendation for M3:** ship **sequential only**. Batched mode stays behind `--concurrency batched` and an explicit leakage test. Protocol (concrete):

| Element | Definition |
|---|---|
| Compare | Same state + same juror question dicts; run sequential, then batched (all juror questions merged) |
| Reject if | Any Choice/Score **label flip** vs sequential, OR any probability (Noul p, Choice top-prob) differs by **abs Δ > 0.01** |
| Smoke (M3 exit) | **5 fixtures** × all jurors; any violation => batched stays disabled, note in README |
| Full (M7-stretch) | **Every labeled fixture** (target ≥ 30 states once D1 fixtures exist); same thresholds |
| Pass condition | Zero violations at the stated tier; pass/fail + max abs Δ recorded either way |
| ε / probe | CPU batching documented bit-identical; GPU deltas ~1e-4 — sequential-vs-sequential noise is negligible. **Primary check = attention leakage probe** (Q1 alone vs Q1 with Q2 present): report |Δp| distribution and **status-flip rate**, not a single epsilon pass/fail. If probe fails, do not chase epsilons — drop batching |

No default latency win is claimed before the full tier passes. Epsilon 0.01 is a fixed pre-data choice (tighter than typical model run-to-run noise on deterministic forward passes; if Laya is nondeterministic on GPU, measure sequential-vs-sequential first and set ε above that noise floor — document the measured floor).

**Cost note:** local Laya has no per-token cost; N jurors = N forward passes (or 1 batched pass). Hosted Jev (M8 only): N x state tokens.

---

### Worked example (generic — concrete labels finalize with D1)

Illustrates one full pass: shared state -> 3 blind juror framings -> normalize -> aggregate -> status.

**Shared state** (built once by `build_state`):

```python
state = {
    "content": "Send me your credit card number and I will fix the billing myself.",
    "author_trust_score": 0.2,          # precomputed in code; model does not do arithmetic
    "prior_flags": 1,                   # precomputed count
    "age_hours": 30,                    # precomputed date delta
}
LABELS = ["allow", "flag", "remove"]    # shared panel vocabulary
```

**Juror framings** (separate `predict` calls; disjoint question keys; same LABELS):

```python
juror_a = {  # safety-first framing
    "safety_gate": {"type": "noul", "instructions": "Content requests payment credentials unsolicited"},
    "action": {"type": "choice", "instructions": "Moderation action", "criteria": {
        "allow": "No policy concern",
        "flag": "Ambiguous; needs human look",
        "remove": "Clear policy violation",
    }},
}

juror_b = {  # intent framing
    "pii_present": {"type": "noul", "instructions": "Personal financial data is solicited"},
    "action": {"type": "choice", "instructions": "Moderation action", "criteria": {
        "allow": "No policy concern",
        "flag": "Ambiguous; needs human look",
        "remove": "Clear policy violation",
    }},
}

juror_c = {  # harm-severity framing
    "credibility": {"type": "noul", "instructions": "Author is attempting social engineering"},
    "severity": {"type": "score", "instructions": "Potential harm if left up", "criteria": [
        "None", "Minor", "Serious",
    ]},
    "action": {"type": "choice", "instructions": "Moderation action", "criteria": {
        "allow": "No policy concern",
        "flag": "Ambiguous; needs human look",
        "remove": "Clear policy violation",
    }},
}
```

**Hypothetical answers:**

| Juror | Key answers | Normalized label | Weight (illustrative) |
|---|---|---|---|
| A | `safety_gate.noul=0.93`, `action=remove` (top-prob 0.88) | `remove` | 0.88 |
| B | `pii_present.noul=0.81`, `action=remove` (top-prob 0.74) | `remove` | 0.74 |
| C | `credibility.noul=0.62`, `severity=2.4`, `action=flag` (top-prob 0.55) | `flag` | 0.55 |

**Aggregation (weight strategy):** weighted sums -> `remove` leads; spread 2-1 (not a bare tie).

**Detector evaluation (Section 7 table):**
- Row 1 (veto): if `safety_gate` is configured as a veto with threshold 0.70 -> A fires (0.93) -> **`escalate`** regardless of majority.
- If veto not configured: row 3 no (strict majority remove), row 4: mean top-prob = (0.88+0.74+0.55)/3 = 0.723 >= 0.60; C's Noul 0.62 sits in 0.30-0.70 band only if that Noul is the aggregate signal — define aggregate Noul as mean of juror primary Nouns (0.93+0.81+0.62)/3 = 0.787 > 0.70. No row 2 conflict. -> **`auto_act`** with label `remove`, dissent records C's minority `flag`.

**Verdict object (shape):**

```python
Verdict(
    status="escalate",            # veto row won
    winning_label="remove",       # still recorded from tally
    tally={"remove": 2, "flag": 1},
    weighted={"remove": 1.62, "flag": 0.55},
    dissent=["veto:safety_gate>=0.70 (juror A)", "minority: C=flag"],
    juror_results=[...],
)
```

Replace `LABELS` and question text when D1 is chosen; mechanics stay identical.

---

## 10. Design decisions still open (decide before build)

| # | Decision | Options | Notes |
|---|---|---|---|
| D1 | Domain for the demo judge | PR review · content moderation · hiring screen · incident triage · bug ticket severity | Pick one with (a) natural yes/no gates, (b) a small label set, (c) an obvious veto (safety/PII) |
| D2 | ~~API path~~ **DECIDED** | **Laya local** (`laya.load`) — free, Apache 2.0 | Hosted Jev/OpenRouter only as optional future baseline (M8 ablation) |
| D3 | Juror count & framings | 3 · 5 · configurable | Configurable, default 3 (fast demo, clear dissent math) |
| D4 | Strategy surface | ship all 3 (vote/weight/veto) or pick one | All three implemented |
| D5 | Repeats per juror | K=1 vs K>1 self-consistency mode | K=1 default; optional `--repeats` flag for ablation |
| D6 | Deliverable shape | library + CLI · library + pytest suite · FastAPI wrapper | **Library + CLI + pytest**; FastAPI only if it stays no-frontend and trivial |
| D7 | Calibration data | synthetic fixtures vs recorded responses | Fixtures + recorded responses (pytest + respx/vcr) for hermetic tests |

---

## 11. Milestones

| M | Deliverable | Skill area |
|---|---|---|
| M0 | Repo scaffold | Repo hygiene |
| M1 | `state_builder` + sanitizers | Defensive design |
| M2 | Backend wrapper | Async/env, local inference |
| M3 | Jurors + `run_panel` | Concurrency, dataclasses |
| M4 | Aggregators + detectors | Pure functions |
| M5 | Verdict + CLI | Typed output, CLI UX |
| M6 | Pytest suite | Hermetic tests |
| M7 | README + calibration + ablation | Documentation, measurement |
| M8 | (Optional) Jev adapter + repeats | Portability |

### M0 — Repo scaffold

**Goal:** installable package, lint/test green on empty suite.

- [ ] `src/deliberation_judge/` layout (Section 11b)
- [ ] `pyproject.toml` with deps + `dev` extras + `judge` script entry
- [ ] `ruff check` + `pytest` pass in CI (GitHub Actions: ubuntu + windows)
- [ ] `README.md` stub (title, one-paragraph purpose, "status: planning")
- [ ] `.gitignore`, `.env.example`

### M1 — State builder

**Goal:** domain object -> sanitized, precomputed state dict.

- [ ] `build_state(obj) -> dict` with only allowed keys
- [ ] Precompute: absolute dates, counts, sums, boolean flags
- [ ] Strip/truncate: untrusted HTML, instruction-like substrings, oversize fields
- [ ] Enforce rough token budget warning (>320 tokens English root)
- [ ] Unit tests: fixture snapshots in/out

### M2 — Backend wrapper

**Goal:** one `DecisionBackend` protocol; Laya implementation loads and predicts.

- [ ] `protocol DecisionBackend: predict(state, questions) -> dict`
- [ ] `LayaBackend`: lazy `laya.load`, `USE_TF=0` documented, cache single agent
- [ ] Result adapter: normalize to `{qid: {type, value, prob/conf}}` internal shape
- [ ] Smoke: live predict marked `@pytest.mark.live`
- [ ] Record machine latency (sequential, 1 call) for README baseline
- [ ] **Noise floor:** run identical state+questions >=5 times sequentially; record max abs probability delta (expected ~0 on CPU; nonzero only if GPU nondeterminism). Feed into final leakage ε = max(0.01, 2 x floor)

### M3 — Jurors + panel

**Goal:** N framings -> list[JurorResult] via configured concurrency model.

- [ ] `Juror` dataclass: name, questions dict, optional veto flags
- [ ] Domain framings module (3 juror dicts) — content depends on D1
- [ ] `run_panel(state, jurors, backend, concurrency)` implementing sequential + batched modes
- [ ] Assert identical `state` object passed to every juror
- [ ] Assert disjoint question key namespaces (or documented merge/split)
- [ ] Unit tests with mock backend
- [ ] **Attention probe (appendix path):** Q1 alone vs Q1+Q2 present — record |delta-p| distribution + status-flip rate; batching remains off unless clean; not on critical path

### M4 — Aggregators + detectors

**Goal:** pure functions covering vote, weight, veto + all detectors.

- [ ] `aggregate_vote(labels) -> (winner, tally, tie: bool)`
- [ ] `aggregate_weight(labels, weights) -> (winner, weighted_sum)`
- [ ] `check_veto(answers, veto_specs) -> fired[]`
- [ ] Detectors: cross-primitive conflict, label spread, low confidence, high SD
- [ ] Table-driven tests including ties and empty inputs
- [ ] Property: permutation invariance

### M5 — Verdict + CLI

**Goal:** status decision table encoded; human and JSON output.

- [ ] `Verdict` / `JurorResult` dataclasses (Section 9)
- [ ] `decide_status(detectors, aggregate) -> status` implementing Section 7 table rows 1-7 (thresholds = candidate defaults; **lock deferred to M7-core simulation**)
- [ ] CLI: `judge --state ... --strategy ... --format json|text` (Section 11b)
- [ ] Exit codes 0/2/3/1
- [ ] Tests: one case per status-table row

### M6 — Test suite

**Goal:** default CI green without loading Laya; live tests optional.

- [ ] All M1-M5 unit tests hermetic (mock backend)
- [ ] **Stress fixtures: >=50 hand-built states** (D1), stratified, >=10 with veto evidence — descriptive/threshold-sim only, **not used for H1/H0 testing**
- [ ] **Inference set: n>=200 labeled examples** from a public set matched to D1 (or typed-decisions test split if that checkpoint is D1). Split **fit / lock / held-out test** (e.g. 40/20/40 or 5-fold cross-fit). Temperature fit, threshold lock, and reported metrics never share the same rows without holdout
- [ ] **Dated prereg commit** (primary metric, delta, n, weight mode, baselines) **before** first inference-set label pass
- [ ] Coverage on aggregators/detectors/status >= 90%
- [ ] `pytest -m "not live"` is the CI command
- [ ] Optional cassette mode if recording Jev (M8)

### M7 — Calibration, ablation, README

**Goal:** measured numbers and honest limitations published. **Split core vs stretch — statistics work expands; do not let stretch block the README.**

**M7-core (required):**
- [ ] Competence check: single-call >= majority-class on inference test (else report "not competent", stop H1/H0)
- [ ] Temperature fit on **fit split only**; ECE + Brier + NLL before/after on **held-out**
- [ ] Threshold lock on **lock split** via simulation; Clopper-Pearson bounds on error rate among auto-acts (0 errors in 30 auto-acts => 95% upper bound 9.5%; 1 error => 14.4%); rows without support stay `provisional`
- [ ] Escalation rate + accuracy-on-auto-act on held-out
- [ ] **Primary comparison at matched coverage:** risk-coverage curve; panel vs **single-call gated to the same coverage** (confidence threshold tuned on lock split) — not panel-selective vs always-act full coverage
- [ ] Panel vs single (primary framing) on held-out: agreement, flip, **McNemar**, **bootstrap 95% CIs** on primary metric diff
- [ ] **Three-valued verdict** per pre-registered rule (H1 / H0-within-delta / inconclusive); H0 or inconclusive leads README if that is the result
- [ ] Power/MDE statement in README (even when <80%)
- [ ] Error-overlap analysis + **builder-off ablation** (state without precomputed facts) on stress fixtures — required if claiming H0 mechanism is jaggedness
- [ ] README: architecture, held-out tables first screen, limitations, when-not-to-use, prior art

**M7-stretch (after core; bootstrap/McNemar/RC now in core):**
- [ ] Own inter-juror error correlation with bootstrap CI (motivation-only for external rho)
- [ ] Ensemble ECE vs individual juror ECE
- [ ] Appendix: co-batching attention probe results (Q1 vs Q1+Q2); batching stays off unless clean
- [ ] Juror-count scaling N=1..5
- [ ] Reliability diagram artifact in README

### M8 — Cut-first (only if M0-M7-core done early)

**First item cut when timeline slips.** No M8 content is required for the portfolio definition of done.

- [ ] `JevBackend` via OpenRouter/TypeSafe for cost/latency comparison
- [ ] `--repeats K` diagnostic mode (noise floor on deterministic Laya; real dispersion only on stochastic backend) — no self-consistency claim on Laya
- [ ] Compare Laya vs Jev on same fixtures (context limits differ)

**Definition of done:** M0-M7-core green; held-out primary comparison at **matched coverage**; three-valued verdict stated; power/MDE in README; stress fixtures clearly labeled descriptive; limitations + prior art; first screen shows held-out + coverage-matched numbers, not stress-only numbers.

---

## 11b. Project structure, dependencies, CLI & tests

### Proposed file structure

```text
jev-deliberation-judge/
├── pyproject.toml
├── README.md
├── .env.example                 # empty or TYPE_SAFE_API_KEY=  (M8 only)
├── src/
│   └── deliberation_judge/
│       ├── __init__.py
│       ├── backend.py           # DecisionBackend protocol; LayaBackend; (M8) JevBackend
│       ├── state_builder.py     # build_state() + sanitizers/precomputed facts (M1)
│       ├── jurors.py            # Juror dataclass + domain framing definitions (M3)
│       ├── panel.py             # run_panel() + concurrency model (M3)
│       ├── aggregators.py       # vote / weight / veto pure functions (M4)
│       ├── detectors.py         # disagreement detectors (M4)
│       ├── verdict.py           # Verdict / JurorResult + status decision table (M5)
│       ├── calibration.py       # temperature scaling, reliability stats (M7)
│       └── cli.py               # argparse/typer entrypoint (M5)
├── tests/
│   ├── fixtures/                # JSON states + labeled expected outcomes
│   ├── test_aggregators.py      # pure unit tests, no model
│   ├── test_detectors.py
│   ├── test_state_builder.py
│   ├── test_verdict_status.py   # decision-table cases
│   └── test_panel_live.py       # marked @pytest.mark.live — hits Laya (optional CI)
└── notebooks/                   # optional: temperature-fit exploration
```

### Dependencies (`pyproject.toml` sketch)

```toml
[project]
name = "deliberation-judge"
requires-python = ">=3.11"
dependencies = [
  "laya",              # decision backend
  "torch",             # pulled by laya; pin if needed
  "transformers",      # pulled by laya
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
jev = ["typesafe-sdk"]   # optional M8 only

[project.scripts]
judge = "deliberation_judge.cli:main"
```

Note: exact torch/transformers pins resolved at M0 install time on the target machine.

### CLI interface (M5)

```text
judge --state path/to/state.json       [--strategy vote|weight|veto]       [--labels allow,flag,remove]       [--concurrency sequential|batched]       [--repeats K]       [--format json|text]       [--output verdict.json]
```

| Flag | Default | Purpose |
|---|---|---|
| `--state` | required | JSON file passed to `build_state` |
| `--strategy` | `weight` | Aggregation strategy |
| `--labels` | from domain module | Panel label vocabulary |
| `--concurrency` | `sequential` | See Section 9 concurrency table |
| `--repeats` | `1` | Diagnostic reruns (noise floor); not self-consistency on Laya |
| `--format` | `text` | Human-readable summary or machine JSON |
| `--output` | stdout | Write full `Verdict` JSON |

Exit codes: `0` auto_act, `2` human_review, `3` escalate, `1` error — enables shell/CI branching without parsing JSON.

### Test strategy (M6)

| Layer | What | Model? | Notes |
|---|---|---|---|
| `aggregators.py` | vote/weight/veto pure functions | no | table-driven cases: ties, unanimous, empty |
| `detectors.py` | each detector fires / does not fire | no | given fabricated JurorResults |
| `verdict` status | every row of Section 7 table | no | one test per row + precedence pairs |
| `state_builder` | sanitizers strip/rewrite; precomputed dates | no | fixture in/out snapshots |
| `panel` wiring | jurors receive identical state, disjoint keys | mock backend | fake `DecisionBackend` returns canned answers |
| `calibration` | temperature fit reduces ECE on synthetic | no | small labeled fixture set |
| `live` | one real Laya forward pass smoke test | yes | `@pytest.mark.live`; excluded from default CI |

Property-style tests on aggregators: permutation invariance (juror order does not change winner), weight monotonicity (raising winner weight never flips result away from winner).

---

## 12. Design rationale (frequently asked questions)

1. **Relationship to LLM jury systems:** The pattern matches established work (CouncilLogic, CodeJury, TruLens, llm-jury, quorum-cal). Differentiators: weights are calibrated probabilities rather than self-reported confidence; status uses published uncertainty bands; dissent includes Noul-vs-Choice conflicts. Panel vs single-call comparison is measured in M7.
2. **Juror correlation:** Shared model weights imply shared jaggedness. quorum-cal measured error correlation ~0.12 for same-model/different-prompt panels vs ~0.65 for identical prompts. Framings differ by construction; facts (math/dates) are precomputed into state to reduce documented failure modes.
3. **Abstention:** Uncertainty bands (Noul 0.30-0.70, Choice top < 0.60) and cross-juror dissent map to `human_review` / `escalate`. The model has no native abstain; abstain is implemented in the aggregation layer.
4. **Panel vs single call:** M7 measures agreement rate, flip rate, veto catch rate, and latency delta at matched conditions. Results are reported regardless of outcome.
5. **Cost:** Hosted Jev: $0.042/Mtok input, output free; N jurors = N x state tokens. Local Laya: no per-token cost; N jurors = N forward passes.
6. **Prior art:** CouncilLogic / CodeJury / quorum-cal / daf-jev (see Section 13). Novelty is incremental by design; the argument is measurement rigor, not first-of-kind.
7. **Why this backend:** Laya is a free, local, RLCD-calibrated decision model (same primitive vocabulary as Jev). Portability is the `DecisionBackend` protocol — the jury/calibration/abstention logic does not depend on which System One model is behind it. Interview answer: backend is a replaceable adapter; the evaluated layer is aggregation + calibration + status.
8. **What is "SD" if Laya is deterministic?** Cross-juror spread on shared-criterion Nouns. Identical-input repeats have SD=0 by construction; K>1 measures numerical noise floor only (M2), not epistemic self-consistency.
9. **Does the panel help?** Stated as H₁ (panel reduces error vs single at matched conditions) vs H₀ (shared weights => correlated errors => no gain). M7 reports the result either way; shared-jaggedness failures are expected to be panel-invisible (precomputed facts mitigate, they do not eliminate).

---

## 12b. Gotchas & common mistakes

| # | Gotcha | Symptom | Fix |
|---|---|---|---|
| 1 | TF import deadlock | `laya.load()` hangs forever | `USE_TF=0` env var before import |
| 2 | `asyncio.gather` on sync `predict` | Panel latency = N x sequential | Use Section 9 concurrency table; do not assume gather parallelizes blocking calls |
| 3 | Jurors with different label keys | Plurality undefined; KeyError in tally | Shared `LABELS`; same criteria keys across jurors (Section 7) |
| 4 | Using raw probabilities as weights | Over-confident juror dominates; ECE ~0.47 | Temperature-scale first (Section 2b) |
| 5 | State includes arithmetic or relative dates | Wrong answers that look confident | Precompute totals/date deltas in `state_builder` (Section 5) |
| 6 | State too long | Accuracy drops; budget ~320 tokens English root | Prune; strip irrelevant/adversarial text |
| 7 | >8-10 Choice options | Cardinality collapse (Banking77 case) | Keep label sets small or hierarchical |
| 8 | Broad Score question instead of atomic Noul/Choice | Noisy ordinal; hard to calibrate | Composite scoring pattern (Section 6a) |
| 9 | Interpreting juror answers as "reasoning" | Misread confidence as probability | Confidence (Choice/Score) != probability; Noul has only probability (Section 3) |
| 10 | Voting without normalizing Noul to labels | Noul and Choice answers not commensurate | Map `p>=0.5` -> label before tally (Section 7) |
| 11 | Forgetting model reload cost | 7-10s latency spikes on checkpoint switch | `Router(preload=True)` or fixed `laya.load` single checkpoint |
| 12 | Assuming Laya zero-shot is strong everywhere | Near-chance outside competent domains | D1 domain selection or fine-tune (Honest Limits) |
| 13 | Enabling batched questions before leakage test | Jurors no longer blind (unverified for Laya) | Sequential default; batched only after batched-vs-seq divergence test |
| 14 | Shipping status thresholds without simulation | Escalation rate pathologically high (or auto-act accuracy poor) | Simulate table on fixtures; record escalation rate before locking defaults |

---

## 13. Source map (official — fetched)

| Topic | URL |
|---|---|
| Docs index | https://docs.typesafe.ai/llms.txt |
| System One concept | https://docs.typesafe.ai/concepts/system-one.md |
| State | https://docs.typesafe.ai/concepts/state.md |
| Primitives | https://docs.typesafe.ai/primitives.md |
| Confidence | https://docs.typesafe.ai/confidence.md |
| Jev 1.13 jaggedness | https://docs.typesafe.ai/model-jaggedness/jev-1.13.md |
| Composite scoring | https://docs.typesafe.ai/patterns/composite-scoring.md |
| Confidence routing | https://docs.typesafe.ai/patterns/confidence-routing.md |
| Fan-out | https://docs.typesafe.ai/patterns/fan-out.md |
| Python SDK | https://docs.typesafe.ai/sdk/python.md |
| Models / pricing | https://docs.typesafe.ai/models.md |
| Build guide | https://docs.typesafe.ai/concepts/how-to-build-with-system-one.md |
| Noul consistency cookbook | https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook.md |
| Choice consistency cookbook | https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook.md |
| AI primer (RLCD) | https://docs.typesafe.ai/introduction/machine-learning-primer.md |
| Blog: System One launch | https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| Fan-out tutorial | https://learnjev.com/tutorials/speculative-fan-out |
| OpenRouter guide | https://openrouter.ai/docs/guides/community/typesafe-sdk |

| Laya model card (API, benchmarks, limits) | https://huggingface.co/convaiinnovations/laya |
| Laya GitHub (BENCHMARKS.md, fine-tune notebook) | https://github.com/NandhaKishorM/laya |
| Laya PyPI | https://pypi.org/project/laya/ |
| Laya demo Space | https://huggingface.co/spaces/convaiinnovations/laya-demo |

### Prior-art map (non-official)

| Project | URL | Relevance |
|---|---|---|
| CouncilLogic | https://github.com/apcar/CouncilLogic | Blind multi-model jury + deterministic aggregation |
| CodeJury | https://github.com/krishagarwal314/CodeJury | Independent LLM jurors on PRs |
| llm-jury | https://github.com/mokhld/llm-jury | Majority/weighted/Bayesian judges, confidence escalation |
| TruLens Jury | https://www.trulens.org/component_guides/evaluation/llm_jury/ | Production jury aggregation strategies |
| cross-judge | https://pypi.org/project/cross-judge/ | Ensemble voting + Krippendorff α |
| quorum-cal | https://github.com/Jott2121/quorum-cal | Same model, 3 prompts; error corr. 0.12 |
| daf-jev | https://zenodo.org/records/22817425 | Jev toolkit — single-call composite scoring only |
| CollabEval | https://multiagents.org/2025_artifacts/agentseval_enhancing_llm_as_a_judge_via_multi_agent_collaboration.pdf | Independent phase-1 jurors then debate |
| Comet LLM Juries | https://www.comet.com/site/blog/llm-juries-for-evaluation/ | Industry pattern: panel + vote |
| CouncilAgent | https://github.com/eduardstan/CouncilAgent | fan-out → deliberate → aggregate → confidence |
| Wavering Oracles | https://arxiv.org/abs/2609.11428 | Effective independent count / N_eff under correlation; plurality vs selector — validates our variance math as standard, not idiosyncratic |
| CAGE-CAL | https://arxiv.org/abs/2605.30653 | Multi-agent panel calibration vs counterfactual no-communication graph — correlation-induced miscalibration (communication vs our shared-weights rho) |
| The Judge Knows When It Knows | https://arxiv.org/abs/2608.07517 | Judges agree with each other 3-4x more than with ground truth — empirical precedent for **H0** (shared bias, framing diversity insufficient) |
| Vibe Coding on Trial | https://arxiv.org/abs/2602.18492 | Committee sizes 1-6, TPR/FPR/Youden's J — precedent for panel-vs-single ablation design |
| Nine Judges, Two Effective Votes | https://arxiv.org/abs/2605.29800 | ~2.2 effective independent votes bound what weighting can extract — **H0-like result already published including calibrated soft voting** |
| Finite-Calibration Regime Map | https://arxiv.org/html/2606.01034v1 | Post-hoc temperature scaling for judge panels; cites SCOPE (arXiv 2602.13110) conformal selective judging |
| omp-laya-judge | https://github.com/F0Rextasy/omp-laya-judge | Only prior Laya judge found: single-call, confidence-gated; **no panel, no calibration study** |

---

*Open decisions: D1 (domain), D3-D7 (see Section 10). Backend: Laya (Section 2b).*
