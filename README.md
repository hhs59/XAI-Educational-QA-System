# XAI — Explainable Educational QA

**LLM reasoning meets formal verification.** A language model picks the answer. A theorem prover (Z3) checks if it's actually correct.

---

## What It Does

Answers two types of educational questions:

| | Logic | Physics |
|---|---|---|
| **Input** | Premises + multiple-choice question | Word problem |
| **Approach** | LLM with 3-vote majority | Regex extraction + SymPy, fallback to LLM |
| **Verification** | Z3 theorem prover — "does the answer LOGICALLY follow?" | 8 formula families, answer normalization |
| **Output** | Answer + cited premises + explanation | Answer + unit + explanation |

```
POST /predict {"question": "Which must be true?\nA. It rains\nB. It snows",
               "premises-NL": ["If it rains, the ground is wet", "It is raining"]}
→ {"answer": "A", "idx": [1, 2], "z3_verified": true, "explanation": "..."}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│              (src/core/orchestrator.py)                  │
├──────────────────────┬──────────────────────────────────┤
│     Logic Solver     │        Physics Solver            │
│  (src/reasoner/)     │      (src/physics/)              │
│                      │                                  │
│  ┌────────────────┐  │  ┌────────────────────┐          │
│  │ Premise Filter │  │  │ Formula Registry   │          │
│  │ (LLM, 1st pass)│  │  │ (8 families)       │          │
│  └───────┬────────┘  │  └────────┬───────────┘          │
│          ▼           │           ▼                      │
│  ┌────────────────┐  │  ┌────────────────────┐          │
│  │ Answer (×3     │  │  │ SymPy Compute      │          │
│  │ majority vote) │  │  │ (deterministic)    │          │
│  └───────┬────────┘  │  └────────┬───────────┘          │
│          ▼           │           ▼                      │
│  ┌────────────────┐  │  ┌────────────────────┐          │
│  │ Z3 Entailment  │  │  │ CoT-guided LLM     │          │
│  │ Verification   │  │  │ (fallback level 2) │          │
│  └───────┬────────┘  │  └────────┬───────────┘          │
│          ▼           │           ▼                      │
│  ┌────────────────┐  │  ┌────────────────────┐          │
│  │ Retry with     │  │  │ Bare LLM           │          │
│  │ Z3 feedback    │  │  │ (fallback level 3) │          │
│  └───────┬────────┘  │  └────────┬───────────┘          │
│          ▼           │           ▼                      │
│  ┌────────────────┐  │                                 │
│  │ Explanation    │  │                                 │
│  │ Generator      │  │                                 │
│  └────────────────┘  │                                  │
├──────────────────────┴──────────────────────────────────┤
│                    FOL Pipeline                          │
│  Parser (Lark) → AST (16 nodes) → Z3 Encoder → Checker  │
│              (src/logic/)                                │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── api/                 FastAPI server + Pydantic schemas
├── core/
│   └── orchestrator.py  Routes logic vs physics, manages pipeline
├── logic/
│   ├── grammar.lark     LALR grammar for first-order logic
│   ├── nodes.py         16 AST node types
│   ├── parser.py        FOL text → AST (Lark-based)
│   ├── z3_encoder.py    AST → Z3 (auto-detects numeric predicates)
│   └── z3_checker.py    Entailment + consistency verification
├── reasoner/
│   ├── direct_answer.py  LLM reasoner: filter → vote → Z3 → retry
│   ├── answer_question.py Thin wrapper
│   └── llm_client.py     OpenAI-compatible API client
├── physics/
│   └── solver.py         SymPy engine + 3-level fallback
├── explanation/
│   └── generator.py      Structured explanation, cited premises
└── evaluation/
scripts/
└── evaluate.py           Full-dataset evaluation
```

## Results

### Logic (411 records — 20-record sample)

| Metric | Score | What happened |
|--------|-------|---------------|
| Answer accuracy | **72.2%** | Up from 66.7% after adding 3-vote majority |
| Premise selection | **77.8%** | Up from 55.6% after adding premise filtering |
| Z3 encode rate | **100%** | 4,466 premises — zero failures after fixing comparison operators |

### Physics (1,354 problems)

| Level | % Solved | How |
|-------|---------|-----|
| Deterministic | **8%** | Regex extraction → SymPy (no LLM call) |
| CoT-guided LLM | **71%** | Shows a solved example, asks LLM to follow it |
| Bare LLM | **21%** | LLM with formula hints only |

## Quick Start

```bash
pip install -e ".[dev]"
echo "API_KEY=your_key" > .env
uvicorn src.api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Which must be true?\nA. Yes\nB. No",
       "premises-NL": ["If it rains, the ground is wet", "It is raining"],
       "premises-FOL": ["∀x (Rain(x) → Wet(x))", "Rain(today)"]}'
```

## Tech Stack

Python 3.11+ · FastAPI · Z3 Theorem Prover · SymPy · Lark · OpenAI-compatible LLM · Pydantic
