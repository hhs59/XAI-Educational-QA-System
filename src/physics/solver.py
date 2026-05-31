import re
from src.core import format_number

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
