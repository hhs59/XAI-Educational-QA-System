"""Answer a full natural-language multiple-choice logic question.

This module now uses the direct LLM solver. It sends all premises, the question,
and all choices together so the LLM can compare options directly.
"""

from src.reasoner.direct_answer import answer_logic_direct


def answer_question(
    premises_raw: list[str],
    question: str,
    choices: list[str],
    *,
    premises_fol: list[str] | None = None,
) -> dict:
    """Answer a multiple-choice question given logic premises.

    Args:
        premises_raw: List of premise strings (natural language, for LLM reasoning).
        question: Natural language question like "Which must be true?"
        choices: List of answer choices in natural language.
        premises_fol: Optional FOL premises for Z3 entailment verification.

    Returns:
        dict with:
          - "answer":    The chosen letter ("A", "B", ...) or "Unknown"
          - "choice":    The text of the chosen answer
          - "reasoning": Explanation from the LLM
          - "idx":       List of 1-based premise indices that support the answer
    """
    return answer_logic_direct(
        premises=premises_raw,
        question=question,
        choices=choices,
        premises_fol=premises_fol,
    )
