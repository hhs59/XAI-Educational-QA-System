def solve(question: str):
    if 'current' in question.lower() and 'resistance' in question.lower():
        return {
            'answer': 'Need to calculate voltage',
            'explanation': "This looks like Ohm's law.",
            'confidence': 0.5
        }
    return {
        "answer": "unknown",
        "explanation": "I do not know how to solve this yet.",
        "confidence": 0.0
    }


if __name__ == '__main__':
    question = 'Current is 2 A and resistance is 5 ohm. Find voltage.'
    result = solve(question)
    print(result)
