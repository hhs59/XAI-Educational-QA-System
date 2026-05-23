# EXACT 2026 Project Roadmap

## Current State

- FastAPI endpoint exists at `/predict`.
- The API accepts `question`, optional `type`, and optional `premises-NL`.
- Logic evaluation exists in `scripts/evaluate.py` and runs over the 808 logic questions.
- The solver currently uses an always-`No` baseline for logic questions.
- Physics evaluation and real physics solving are not implemented yet.

## Milestone Checklist

### 1. Fix Current Baseline Issues

- Correct the evaluator accuracy formula to:

```python
correct / total * 100
```

- Make `solve(...)` handle `query_type=None` safely before calling `.lower()`.
- Keep every API response returning at least:

```json
{
  "answer": "...",
  "explanation": "..."
}
```

### 2. Complete Local Evaluation

- Keep logic evaluation with:
  - total questions
  - correct answers
  - unknown answers
  - accuracy percentage
- Add physics evaluation from the physics dataset.
- Add an optional `--limit` argument for fast testing while developing.
- Call `solve(...)` directly from the evaluator, not through HTTP.

### 3. Improve The Logic Baseline

- Detect multiple-choice questions and default to `A`.
- Detect Yes/No/Unknown questions and default to `No`.
- Normalize predictions and gold answers before comparison.
- Later, add retrieval over similar training examples.
- Return useful explanation text, even for baseline predictions.

### 4. Add Type 1 Symbolic Reasoning

- Treat Z3 or a similar solver as a reasoning upgrade after the evaluator and simple baselines work.
- Do not expect Z3 to solve raw natural language directly.
- Use an intermediate representation before symbolic solving:
  - extract candidate facts and rules from `premises-NL`
  - keep only simple, high-confidence implications and facts
  - convert those rules into a restricted symbolic form
- Use symbolic solving for:
  - implication chains
  - simple consistency checks
  - Yes/No/Unknown verification
  - answer validation for multiple-choice candidates
  - explanation evidence through cited premises or generated `fol`
- Use the LLM for natural-language interpretation and symbolic candidate generation.
- Use Z3 or a similar solver to verify candidate reasoning where the formalization is reliable.
- Fall back to retrieval/LLM reasoning when the premises cannot be safely formalized.

### 5. Improve The Physics Baseline

- Parse numbers and units from the question.
- Add formula handlers for common topics:
  - Ohm's law: `V = I * R`
  - Resistance: `R = U / I`
  - Power: `P = U * I`
  - Capacitor energy: `E = 0.5 * C * U^2`
  - Capacitance: `C = Q / U`
  - Coulomb force: `F = k * |q1 * q2| / r^2`
  - Electric field: `E = F / q`
- Use Python for arithmetic instead of relying on the LLM.
- Return `answer`, `unit`, `explanation`, and `confidence` when possible.

### 6. Add Open-Source LLM Support

- Use only eligible self-hosted open-source models in the nominal `<= 8B` class.
- Do not use GPT, Claude, Gemini, or third-party inference APIs at inference time.
- Use the LLM for:
  - question classification
  - value extraction
  - formula selection
  - explanation generation
  - logic premise matching
- Keep arithmetic and symbolic checks in Python or solver tools where possible.

### 7. Prepare Deployment

- Serve the LLM through `vLLM` or a compatible OpenAI-style server.
- Keep `/predict` publicly reachable during evaluation windows.
- Keep request latency under 60 seconds.
- Make sure `/v1/models` can show the hosted model identity.
- Test both logic and physics requests against the deployed endpoint.

### 8. Prepare Final Submission

- Public API URL.
- One-page solution description PDF.
- Data disclosure PDF.
- List all models, tools, datasets, retrieval sources, and generated data.

## API Contract

### Logic Input

```json
{
  "type": "logic",
  "premises-NL": [
    "If a Python code is well-tested, then the project is optimized.",
    "All Python code is well-tested."
  ],
  "question": "Does it follow that the project is optimized?"
}
```

### Physics Input

```json
{
  "type": "physics",
  "question": "Current is 2 A and resistance is 5 ohm. Find voltage."
}
```

### Required Output

```json
{
  "answer": "...",
  "explanation": "..."
}
```

### Optional Output Fields

```json
{
  "cot": ["Step 1 ...", "Step 2 ..."],
  "premises": ["Premise 1 ..."],
  "fol": "ForAll(x, A(x) -> B(x))",
  "unit": "V",
  "confidence": 0.8
}
```

## Test Plan

Run the logic evaluator:

```bash
python -m scripts.evaluate
```

Expected current behavior:

- Total should be 808 logic questions.
- Correct should be nonzero while the always-`No` baseline is active.
- Accuracy should be printed as a percentage after the formula is fixed.

After physics evaluation is added, the evaluator should print separate summaries:

```text
Logic:
  total: ...
  correct: ...
  unknown: ...
  accuracy: ...%

Physics:
  total: ...
  correct: ...
  unknown: ...
  accuracy: ...%
  average_runtime: ...s
```

Manual API checks should include:

- One logic request with `premises-NL`.
- One physics request with a simple Ohm's law question.
- One unknown or ambiguous request to confirm the API still returns valid JSON.

Before applying symbolic reasoning to the full logic dataset, add small hand-written checks:

- `A -> B`, `A`, question `B` should return `Yes`.
- `A -> B`, `not B`, question `A` should return a contradiction-aware result based on the exact question wording.
- `A -> B`, question `C` should return `Unknown`.

Track logic scores separately for:

- always-`No` baseline
- question-type baseline
- retrieval/LLM-assisted baseline
- Z3-assisted baseline

## Assumptions

- This file is an internal project roadmap, not the final competition PDF.
- The roadmap should stay practical and checklist-oriented.
- The next engineering priority is fixing evaluator accuracy and adding physics evaluation.
- Full model training or fine-tuning should wait until the evaluator can measure both datasets.
- Z3 or a similar solver is optional under the challenge rules, but useful for reasoning depth.
- The first symbolic implementation should support only simple formal patterns, not full natural-language theorem proving.
- Physics should use Python or SymPy-style computation, not Z3.
