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
