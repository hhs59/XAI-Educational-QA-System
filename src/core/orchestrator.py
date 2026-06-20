import logging
import re

logger = logging.getLogger(__name__)


def _extract_choices(question: str) -> list[str] | None:
    pattern = r"([A-Z])\.\s*(.*?)(?=\s+[A-Z]\.\s*|$)"
    matches = re.findall(pattern, question, re.DOTALL)
    if len(matches) >= 2:
        return [text.strip() for _, text in matches]
    return None


def _split_question_and_choices(question: str) -> tuple[str, list[str] | None]:
    choices = _extract_choices(question)
    if choices:
        match = re.search(r"\s+[A-Z]\.\s*", question)
        if match:
            stem = question[: match.start()].strip()
            return stem, choices
    return question, choices


def solve(
    question: str,
    premises_nl: list[str] | None = None,
    premises_fol: list[str] | None = None,
) -> dict:
    if premises_nl:
        return _solve_logic(question, premises_nl, premises_fol)
    else:
        return _solve_physics(question)


def _solve_logic(
    question: str,
    premises_nl: list[str] | None,
    premises_fol: list[str] | None = None,
) -> dict:
    from src.reasoner.answer_question import answer_question
    from src.explanation.generator import generate_explanation

    stem, choices = _split_question_and_choices(question)

    if not premises_nl:
        return {
            "answer": "Unknown",
            "choice": "",
            "reasoning": "No premises provided for logic question.",
            "idx": [],
            "explanation": "No premises provided for logic question.",
        }

    if not choices:
        choices = ["Yes", "No"]

    result = answer_question(
        premises_raw=premises_nl,
        question=stem,
        choices=choices,
        premises_fol=premises_fol,
    )

    explanation = generate_explanation(
        premises=premises_nl,
        question=stem,
        answer=result.get("answer", "Unknown"),
        choice=result.get("choice", ""),
        reasoning=result.get("reasoning", ""),
        idx=result.get("idx", []),
    )

    return {
        "answer": result.get("answer", "Unknown"),
        "choice": result.get("choice", ""),
        "reasoning": result.get("reasoning", ""),
        "idx": result.get("idx", []),
        "explanation": explanation,
    }


def _solve_physics(question: str) -> dict:
    from src.physics.solver import solve_physics

    result = solve_physics(question)

    if "explanation" not in result:
        result["explanation"] = result.get("reasoning", "")

    return result
