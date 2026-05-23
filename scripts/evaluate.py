import json
import csv
from pathlib import Path
from app.solver import solve

LOGIC_PATH = Path('data/Logic_Based_Educational_Queries_Text_Only/Logic_Based_Educational_Queries.json')
PHYSICS_PATH = Path('data/Physics_Problems_Text_Only/Physics_Problems_Text_Only')


# Logics
with LOGIC_PATH.open('r', encoding='utf-8') as logic:
    records = json.load(logic)



total = 0
correct = 0
unknown = 0

for record in records:
    premises = record['premises-NL']

    for question, gold_answer in zip(record['questions'], record['answers']):
        prediction = solve(
            question=question,
            query_type='logic',
            premises_nl=premises
        )

        predicted_answer = prediction.get('answer', '')
        is_correct = predicted_answer.strip() == gold_answer.strip()

        total += 1
        if is_correct:
            correct += 1
        if predicted_answer.lower().strip() in {'unknown', 'do not know yet'}:
            unknown += 1

print(f"""
    Logic:
        Total: {total},
        Correct: {correct},
        Unknown: {unknown},
        Accuracy:, {correct/total*100}
    """)

# Physics
with PHYSICS_PATH.open('r', encoding='utf-8') as physics:
    rows = csv.DictReader(physics)

    for row in rows:
        question = row['question']
        gold_answer = row['answer']
        prediction = solve(
            question = question,
            query_type = 'physics',
            premises_nl = None
        )

        predicted_answer = prediction.get('answer', '')
        is_correct = predicted_answer.strip() == gold_answer.strip()

        total += 1
        if is_correct:
            correct += 1
        if predicted_answer.lower().strip() in {'unknown', 'do not know yet'}:
            unknown += 1


print(f"""
    Locgic:
        Total: {total},
        Correct: {correct},
        Unknown: {unknown},
        Accuracy:, {correct/total*100}
    """)
