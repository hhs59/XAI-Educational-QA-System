from src.reasoner.direct_answer import answer_logic_direct


def answer_question(
    premises_raw: list[str],
    question: str,
    choices: list[str],
    *,
    premises_fol: list[str] | None = None,
) -> dict:
    return answer_logic_direct(
        premises=premises_raw,
        question=question,
        choices=choices,
        premises_fol=premises_fol,
    )
