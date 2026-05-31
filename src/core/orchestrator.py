from src.logic import solve_logic
from src.physics import solve_physics

def format_number(value):
    if value.is_integer():
        return str(int(value))
    return str(value)

def solve(question, query_type=None, premises_nl=None):
    query_type = (query_type or '').lower()
    if query_type == 'logic' or premises_nl:
        return solve_logic(question, premises_nl)
    elif query_type == 'physics':
        return solve_physics(question)
    else:
        return {
            'answer': 'No',
            'explanation': 'Baseline: always predicts No for logic questions',
            'confidence': '0.0'
        }
