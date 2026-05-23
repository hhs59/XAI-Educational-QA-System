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


def solve_logic(question: str, premises_nl: list[str] | None):
    return {
        'answer': 'No',
        'explanation': 'this is models explanation for logic question'
    }

def solve_physics(question: str):
    return {
        'answer': 'this is models answer for physics question',
        'explanation': 'this is models answer for physics question'
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


if __name__ == '__main__':
    question = 'Current is 2 A and resistance is 5 ohm. Find voltage.'
    result = solve(question)
    print(result)
