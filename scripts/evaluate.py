import csv
import json
from pathlib import Path

from src.core.orchestrator import solve

LOGIC_PATH = Path(
    "data/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json"
)
PHYSICS_PATH = Path("data/Physics_Problems_Text_Only/Physics_Problems_Text_Only")


with LOGIC_PATH.open("r", encoding="utf-8") as logic:
    records = json.load(logic)


logic_total = 0
logic_correct = 0
logic_unknown = 0
logic_explanation_lengths = []
logic_idx_overlap_scores = []

physics_total = 0
physics_correct = 0
physics_unknown = 0

for record in records:
    premises = record["premises-NL"]
    premises_fol = record.get("premises-FOL")
    gold_explanations = record.get("explanation", [])
    gold_idx_list = record.get("idx", [])

    for q_idx, (question, gold_answer) in enumerate(zip(record["questions"], record["answers"])):
        prediction = solve(
            question=question,
            premises_nl=premises,
            premises_fol=premises_fol,
        )

        predicted_answer = prediction.get("answer", "")
        is_correct = predicted_answer.strip() == gold_answer.strip()

        logic_total += 1
        if is_correct:
            logic_correct += 1
        if predicted_answer.lower().strip() in {"unknown", "do not know yet"}:
            logic_unknown += 1

        explanation = prediction.get("explanation", "")
        if explanation:
            logic_explanation_lengths.append(len(explanation.split()))

        pred_idx = set(prediction.get("idx", []))
        if q_idx < len(gold_idx_list):
            gold_idx = set(gold_idx_list[q_idx])
            if gold_idx and pred_idx:
                overlap = len(pred_idx & gold_idx) / len(gold_idx) if gold_idx else 0
                logic_idx_overlap_scores.append(overlap)

if logic_total > 0:
    avg_len = sum(logic_explanation_lengths) / len(logic_explanation_lengths) if logic_explanation_lengths else 0
    avg_overlap = sum(logic_idx_overlap_scores) / len(logic_idx_overlap_scores) if logic_idx_overlap_scores else 0
    print(f"""
    Logic:
        Total: {logic_total},
        Correct: {logic_correct},
        Unknown: {logic_unknown},
        Accuracy: {logic_correct / logic_total * 100:.2f}%
        Avg explanation length: {avg_len:.0f} words
        Avg premise overlap: {avg_overlap:.2f}
    """)
else:
    print("    Logic: No records found.")

with PHYSICS_PATH.open("r", encoding="utf-8") as physics:
    rows = csv.DictReader(physics)

    for row in rows:
        question = row["question"]
        gold_answer = row["answer"]
        prediction = solve(question=question)

        predicted_answer = prediction.get("answer", "")
        is_correct = predicted_answer.strip() == gold_answer.strip()

        physics_total += 1
        if is_correct:
            physics_correct += 1
        if predicted_answer.lower().strip() in {"unknown", "do not know yet"}:
            physics_unknown += 1

if physics_total > 0:
    print(f"""
    Physics:
        Total: {physics_total},
        Correct: {physics_correct},
        Unknown: {physics_unknown},
        Accuracy: {physics_correct / physics_total * 100:.2f}%
    """)
else:
    print("    Physics: No records found.")
