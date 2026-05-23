# def solve(question: str):
#     if 'current' in question.lower() and 'resistance' in question.lower():
#         return {
#             'answer': 'Need to calculate voltage',
#             'explanation': "This looks like Ohm's law.",
#             'confidence': 0.5
#         }
#     return {
#         "answer": "unknown",
#         "explanation": "I do not know how to solve this yet.",
#         "confidence": 0.0
#     }

import re

def format_number(value):
    if value.is_integer():
        return str(int(value))
    return str(value)

def solve_logic(question: str, premises_nl: list[str] | None):
    if '\nA.' in question or 'A.' in question and 'B.' in question:
        return {
            'answer': 'A',
            'explanation': 'this is models explanation for logic question'
        }
    else:
        return {
            'answer': 'No',
            'explanation': 'this is models explanation for logic question'
        }

def solve_physics(question: str):
    question_lower = question.lower()
    if 'current' in question_lower and 'resistance' in question_lower and 'voltage' in question_lower:
        current_match = re.search(r'(\d+(?:\.\d+)?)\s*A\b', question)
        resistance_match = re.search(r'(\d+(?:\.\d+)?)\s*(ohm|Ω)\b', question, re.IGNORECASE)
        if not current_match or not resistance_match:
            return {
                "answer": "unknown",
                "explanation": "The question mentions Ohm's law quantities, but current or resistance could not be extracted.",
                "confidence": 0.0
            }
        I = float(current_match.group(1))
        R = float(resistance_match.group(1))
        V = I * R
        return {
            'answer': format_number(V),
            'unit': 'V',
            'explanation': f"Using Ohm's law V = I * R. With I = {I} A and R = {R} ohm, V = {V} V.",
            'confidence': 0.8
        }
    else:
        return {
            'answer': 'unknown',
            'explanation': 'No supported physics formula matched this question.',
            'confidence': 0.0
        }

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


# def extract_number_before_unit(question, units):
#     current_match = re.search(r'([0-9.]+)\s*A\b', question)
#     resistance_match = re.search(r'([0-9.]+)\s*(ohm|Ω)\b', question, re.IGNORECASE
#     I = float(current_match.group(1))
#     R = float(resistance_match.group(1))
#     V = I * R

if __name__ == '__main__':
    question = 'Current is 2 A and resistance is 5 ohm. Find voltage.'
    result = solve(question)
    print(result)
