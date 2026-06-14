import logging
import re

logger = logging.getLogger(__name__)


def generate_explanation(
    premises: list[str],
    question: str,
    answer: str,
    choice: str,
    reasoning: str,
    idx: list[int],
) -> str:
    if not reasoning:
        return _build_basic_explanation(premises, answer, choice, idx)

    cleaned = _clean_reasoning(reasoning)

    cited_premises = _extract_cited_premises(cleaned, premises, idx)

    parts = []
    if cited_premises:
        parts.append("Given the premises:")
        for i, p in cited_premises:
            parts.append(f"  {i}. {p}")
        parts.append("")

    parts.append(cleaned)

    if answer != "Unknown" and choice:
        parts.append(f"\nTherefore, the answer is {answer}: {choice}.")

    explanation = "\n".join(parts)

    words = explanation.split()
    if len(words) > 500:
        explanation = " ".join(words[:500]) + "..."

    return explanation


def _build_basic_explanation(
    premises: list[str],
    answer: str,
    choice: str,
    idx: list[int],
) -> str:
    parts = []

    if idx:
        parts.append("Based on the premises:")
        for i in idx:
            if 1 <= i <= len(premises):
                parts.append(f"  {i}. {premises[i - 1]}")
        parts.append("")

    if answer != "Unknown" and choice:
        parts.append(f"The answer is {answer}: {choice}.")
    else:
        parts.append("The answer could not be determined from the given premises.")

    return "\n".join(parts)


def _clean_reasoning(reasoning: str) -> str:
    cleaned = reasoning.strip()

    cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'^[\s*]*reasoning[\s*:]*', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip('"\' \n')

    return cleaned


def _extract_cited_premises(
    reasoning: str,
    premises: list[str],
    idx: list[int],
) -> list[tuple[int, str]]:
    cited = []

    mentioned = set()
    for match in re.finditer(r'[Pp]remise\s*(\d+)', reasoning):
        try:
            num = int(match.group(1))
            mentioned.add(num)
        except ValueError:
            pass

    for match in re.finditer(r'(?<!\d)(\d+)\.\s', reasoning):
        try:
            num = int(match.group(1))
            if 1 <= num <= len(premises):
                mentioned.add(num)
        except ValueError:
            pass

    all_indices = sorted(mentioned | set(idx))

    for i in all_indices:
        if 1 <= i <= len(premises):
            cited.append((i, premises[i - 1]))

    return cited


def format_explanation_for_api(
    premises: list[str],
    answer: str,
    choice: str,
    reasoning: str,
    idx: list[int],
) -> dict:
    explanation = generate_explanation(
        premises=premises,
        question="",
        answer=answer,
        choice=choice,
        reasoning=reasoning,
        idx=idx,
    )

    return {
        "answer": answer,
        "choice": choice,
        "reasoning": reasoning,
        "idx": idx,
        "explanation": explanation,
    }
