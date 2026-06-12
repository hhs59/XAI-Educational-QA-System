import csv
import json
from pathlib import Path

from src.core.orchestrator import solve

LOGIC_PATH = Path(
    "data/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json"
)  # noqa: E501
PHYSICS_PATH = Path("data/Physics_Problems_Text_Only/Physics_Problems_Text_Only")


# Logics
with LOGIC_PATH.open("r", encoding="utf-8") as logic:
    records = json.load(logic)


logic_total = 0
logic_correct = 0
logic_unknown = 0

physics_total = 0
physics_correct = 0
physics_unknown = 0

for record in records:
    premises = record["premises-NL"]

    for question, gold_answer in zip(record["questions"], record["answers"]):
        prediction = solve(question=question, query_type="logic", premises_nl=premises)

        predicted_answer = prediction.get("answer", "")
        is_correct = predicted_answer.strip() == gold_answer.strip()

        logic_total += 1
        if is_correct:
            logic_correct += 1
        if predicted_answer.lower().strip() in {"unknown", "do not know yet"}:
            logic_unknown += 1

print(f"""
    Logic:
        Total: {logic_total},
        Correct: {logic_correct},
        Unknown: {logic_unknown},
        Accuracy: {logic_correct / logic_total * 100:.2f}%
    """)

# Physics
with PHYSICS_PATH.open("r", encoding="utf-8") as physics:
    rows = csv.DictReader(physics)

    for row in rows:
        question = row["question"]
        gold_answer = row["answer"]
        prediction = solve(question=question, query_type="physics", premises_nl=None)

        predicted_answer = prediction.get("answer", "")
        is_correct = predicted_answer.strip() == gold_answer.strip()

        physics_total += 1
        if is_correct:
            physics_correct += 1
        if predicted_answer.lower().strip() in {"unknown", "do not know yet"}:
            physics_unknown += 1


print(f"""
    Physics:
        Total: {physics_total},
        Correct: {physics_correct},
        Unknown: {physics_unknown},
        Accuracy: {physics_correct / physics_total * 100:.2f}%
    """)
