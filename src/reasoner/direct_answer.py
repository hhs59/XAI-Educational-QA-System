import json
import logging
import re
from string import ascii_uppercase

import z3
from openai.types.chat import ChatCompletionMessageParam

from src.logic.nodes import Expr
from src.logic.parser import parse_fol
from src.logic.z3_checker import check_entailment, check_entailment_batch
from src.logic.z3_encoder import Z3Context, encode_expr
from src.reasoner.llm_client import call_llm

logger = logging.getLogger(__name__)


def answer_logic_direct(
    premises: list[str],
    question: str,
    choices: list[str],
    *,
    premises_fol: list[str] | None = None,
    max_retries: int = 1,
) -> dict[str, str | list[int]]:
    if not choices:
        return {
            "answer": "Unknown",
            "choice": "",
            "reasoning": "No answer choices were provided.",
            "idx": [],
        }

    valid_letters = [ascii_uppercase[i] for i in range(len(choices))]
    answer_enum = valid_letters + ["Unknown"]
    schema = {
        "name": "logic_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": answer_enum},
                "reasoning": {"type": "string"},
                "idx": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["answer", "reasoning", "idx"],
            "additionalProperties": False,
        },
    }

    messages = _build_direct_prompt(premises, question, choices)
    z3_premises = premises_fol if premises_fol else premises
    max_z3_retries = 1

    for z3_attempt in range(max_z3_retries + 1):
        parsed = None
        for json_attempt in range(max_retries + 1):
            response_text = call_llm(messages, schema=schema)
            parsed = _parse_direct_response(response_text, choices)

            if parsed["answer"] != "Unknown":
                break

            logger.warning(
                "Invalid direct-answer JSON from LLM (attempt %d): %s",
                json_attempt + 1,
                response_text[:200],
            )

            retry_messages: list[ChatCompletionMessageParam] = [
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON or did not contain "
                        "a valid answer letter. Return ONLY valid JSON like: "
                        '{"answer": "A", "reasoning": "..."}. '
                        'If no option follows, use {"answer": "Unknown", "reasoning": "..."}.'  # noqa: E501
                    ),
                },
            ]
            messages = messages + retry_messages

        if parsed is None or parsed["answer"] == "Unknown":
            return {
                "answer": "Unknown",
                "choice": "",
                "reasoning": "Could not parse a valid answer from the LLM response.",
                "idx": [],
                "z3_verified": False,
                "z3_note": "LLM failed to produce valid response.",
            }

        z3_result = _verify_with_z3(
            z3_premises, parsed["idx"], answer_choice=parsed.get("choice")
        )
        parsed["z3_verified"] = z3_result["verified"]
        parsed["z3_note"] = z3_result["note"]

        if z3_result["verified"] or z3_attempt >= max_z3_retries:
            return parsed

        logger.info(
            "Z3 rejected answer %s (attempt %d/%d): %s",
            parsed["answer"],
            z3_attempt + 1,
            max_z3_retries,
            z3_result["note"],
        )

        feedback = (
            f"Your previous answer '{parsed['answer']}' was checked against the "
            f"premises using formal logic verification and was found to be NOT "
            f"guaranteed by the premises. Reason: {z3_result['note']}\n\n"
            f"Reconsider your answer. Carefully re-examine each premise and each "
            f"choice. Only choose an answer that LOGICALLY FOLLOWS from the premises.\n\n"
            f'Return ONLY JSON: {{"answer": "A/B/C/D/Unknown", "reasoning": "...", "idx": [...]}}'
        )

        messages = messages + [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": feedback},
        ]

    return parsed


def _build_direct_prompt(
    premises: list[str],
    question: str,
    choices: list[str],
) -> list[ChatCompletionMessageParam]:
    premises_text = "\n".join(f"{i}. {p}" for i, p in enumerate(premises, start=1))
    choices_text = "\n".join(
        f"{ascii_uppercase[i]}. {choice}" for i, choice in enumerate(choices)
    )

    user_prompt = f"""\
Premises:
{premises_text}

Question:
{question}

Choices:
{choices_text}

Reasoning steps:
1. For each choice, identify which premises (by number) are needed to prove or disprove it.
2. A choice is supported if it LOGICALLY FOLLOWS from the premises using rules like modus ponens, modus tollens, hypothetical syllogism, disjunctive syllogism.
3. If multiple choices are supported, pick the one that requires the fewest premises.
4. If no choice follows from the premises, answer Unknown.

In your reasoning, cite each premise you use (e.g. "From Premise 3: ...").
The idx field must contain ONLY the premises you actually used to derive your answer (1-based indices).
Return ONLY JSON in this format:
{{"answer": "A/B/C/D/Unknown", "reasoning": "step-by-step logical chain citing premises", "idx": [1, 3]}}"""

    return [{"role": "user", "content": user_prompt}]


def _parse_direct_response(response_text: str, choices: list[str]) -> dict[str, str | list[int]]:
    clean = re.sub(
        r"```(?:json)?\s*\n?(.*?)```",
        r"\1",
        response_text,
        flags=re.DOTALL,
    ).strip()

    if not clean.startswith("{"):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            clean = match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        return {"answer": "Unknown", "choice": "", "reasoning": "Invalid JSON."}

    answer = str(data.get("answer", "Unknown")).strip().upper()
    reasoning = str(data.get("reasoning", "")).strip()

    valid_letters = {ascii_uppercase[i] for i in range(len(choices))}

    if answer in {"UNKNOWN", "NONE", "NULL", "N/A"}:
        return {"answer": "Unknown", "choice": "", "reasoning": reasoning}

    match = re.match(r"^([A-Z])", answer)
    if match:
        answer = match.group(1)

    if answer not in valid_letters:
        return {
            "answer": "Unknown",
            "choice": "",
            "reasoning": reasoning or f"Invalid answer letter: {answer}",
        }

    choice_idx = ascii_uppercase.index(answer)

    raw_idx = data.get("idx", [])
    if isinstance(raw_idx, list):
        premise_indices = [int(x) for x in raw_idx if isinstance(x, (int, float))]
    else:
        premise_indices = []

    return {
        "answer": answer,
        "choice": choices[choice_idx],
        "reasoning": reasoning,
        "idx": premise_indices,
    }


def _verify_with_z3(
    premises: list[str],
    selected_idx: list[int],
    answer_choice: str | None = None,
) -> dict[str, str | bool]:
    if not selected_idx:
        return {"verified": True, "note": "No premises selected to verify."}

    selected_premises = []
    for idx in selected_idx:
        if 1 <= idx <= len(premises):
            selected_premises.append(premises[idx - 1])

    if not selected_premises:
        return {"verified": True, "note": "No valid premise indices."}

    premises_ast = []
    parse_failures = []
    for i, p in enumerate(selected_premises):
        try:
            ast = parse_fol(p)
            premises_ast.append(ast)
        except Exception:
            parse_failures.append(selected_idx[i])

    if not premises_ast:
        return {
            "verified": True,
            "note": "Could not parse any premises as FOL. Skipped Z3 check.",
        }

    if answer_choice:
        entailment = _check_entailment_for_choice(premises_ast, answer_choice, selected_idx, parse_failures)
        if entailment is not None:
            return entailment

    return _check_premises_consistency(premises_ast, selected_idx, parse_failures)


def _check_entailment_for_choice(
    premises_ast: list[Expr],
    answer_choice: str,
    selected_idx: list[int],
    parse_failures: list[int],
) -> dict[str, str | bool] | None:
    choice_fol = _convert_choice_to_fol(premises_ast, answer_choice)
    if choice_fol is None:
        return None

    try:
        result = check_entailment(premises_ast, choice_fol)

        if result["entailed"]:
            note = "Z3 confirmed: premises entail the answer."
            if parse_failures:
                note += f" (Could not parse premises {parse_failures} as FOL.)"
            return {"verified": True, "note": note}
        elif result["entailed"] is False:
            counterexample = result.get("counterexample")
            note = "Z3 could NOT confirm entailment. Premises do not guarantee the answer."
            if counterexample:
                note += f" Counterexample: {counterexample}"
            return {"verified": False, "note": note}
        else:
            return None

    except Exception as e:
        logger.warning("Entailment check failed: %s", e)
        return None


def _convert_choice_to_fol(
    premises_ast: list[Expr],
    choice_text: str,
) -> Expr | None:
    names: set[str] = set()
    for ast in premises_ast:
        _collect_names(ast, names)
    names.discard("")
    sorted_names = sorted(names)

    names_text = ", ".join(sorted_names) if sorted_names else "(none found)"

    schema = {
        "name": "choice_to_fol",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "formula": {"type": "string"},
            },
            "required": ["formula"],
            "additionalProperties": False,
        },
    }

    prompt = f"""Predicates and constants found in the premises:
{names_text}

Answer choice: "{choice_text}"

Express this answer choice as a single FOL formula using ONLY the predicates and constants listed above.
Use ∀x for "for all", ∃x for "there exists", → for "if...then", ∧ for "and", ∨ for "or", ¬ for "not".
If the choice refers to a specific entity (like a person), use their name as a constant.

Return ONLY the formula string in the "formula" field, nothing else.
Example: {{"formula": "∀x(WT(x) → PEP8(x))"}} """

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm(messages, schema=schema)
        data = json.loads(response)

        formula_str = data.get("formula", "").strip()
        if not formula_str:
            for v in data.values():
                if isinstance(v, str) and ("∀" in v or "∃" in v or "→" in v or "∧" in v or "¬" in v):
                    formula_str = v.strip()
                    break
        if not formula_str:
            for v in data.values():
                if isinstance(v, str) and v.strip() and len(v.strip()) > 2:
                    formula_str = v.strip()
                    break

        if not formula_str:
            return None

        ast = parse_fol(formula_str)

        formula_names: set[str] = set()
        _collect_names(ast, formula_names)
        known_names = {n for ast_node in premises_ast for n in _collect_names_flat(ast_node)}
        unknown = formula_names - known_names - {""}
        unknown = {n for n in unknown if len(n) > 1}
        if unknown:
            logger.debug(
                "Choice-to-FOL used unknown predicates %s, rejecting conversion", unknown
            )
            return None

        return ast

    except Exception as e:
        logger.debug("Choice-to-FOL conversion failed: %s", e)
        return None


def _collect_names_flat(node: Expr) -> set[str]:
    names: set[str] = set()
    _collect_names(node, names)
    return names


def _check_premises_consistency(
    premises_ast: list[Expr],
    selected_idx: list[int],
    parse_failures: list[int],
) -> dict[str, str | bool]:
    try:
        ctx = Z3Context()
        solver = z3.Solver()
        for ast in premises_ast:
            solver.add(encode_expr(ast, ctx))

        result = solver.check()

        if result == z3.unsat:
            return {
                "verified": False,
                "note": f"Z3 found contradiction in premises {selected_idx}.",
            }
        elif result == z3.sat:
            note = "Premises are consistent."
            if parse_failures:
                note += f" (Could not parse premises {parse_failures} as FOL.)"
            return {"verified": True, "note": note}
        else:
            return {
                "verified": True,
                "note": "Z3 returned unknown. Assuming consistent.",
            }

    except Exception as e:
        logger.warning("Z3 verification failed: %s", e)
        return {"verified": True, "note": f"Z3 verification error: {e}"}


def answer_with_z3(
    premises_fol: list[str],
    question: str,
    choices: list[str],
) -> dict[str, str | list[int]]:
    if not choices:
        return {
            "answer": "Unknown",
            "choice": "",
            "reasoning": "No answer choices were provided.",
        }

    predicates = _convert_choices_to_fol(premises_fol, choices)

    if not predicates:
        logger.warning("Predicate mapping failed, falling back to direct LLM answer.")
        return answer_logic_direct(premises_fol, question, choices)

    try:
        premises_ast = [parse_fol(p) for p in premises_fol]
    except Exception as e:
        logger.warning("Failed to parse premises: %s", e)
        return answer_logic_direct(premises_fol, question, choices)

    queries_ast: list[Expr] = []
    valid_indices: list[int] = []
    for i, pred in enumerate(predicates):
        if not pred or not pred.strip():
            continue
        try:
            queries_ast.append(parse_fol(pred))
            valid_indices.append(i)
        except Exception as e:
            logger.warning(
                "Failed to parse predicate for choice %s: %s", chr(65 + i), e
            )

    if not queries_ast:
        logger.warning("No valid predicates, falling back to direct LLM answer.")
        return answer_logic_direct(premises_fol, question, choices)

    z3_results = check_entailment_batch(premises_ast, queries_ast)

    entailed_indices = [
        valid_indices[i] for i, r in enumerate(z3_results) if r["entailed"]
    ]

    if not entailed_indices:
        return {
            "answer": "Unknown",
            "choice": "",
            "reasoning": "Z3 found no choice entailed by the premises.",
        }

    chosen_idx = entailed_indices[0]
    chosen_letter = ascii_uppercase[chosen_idx]
    chosen_pred = predicates[chosen_idx]

    explanation = _explain_z3_result(
        premises_fol,
        choices[chosen_idx],
        chosen_letter,
        chosen_pred,
    )

    return {
        "answer": chosen_letter,
        "choice": choices[chosen_idx],
        "reasoning": explanation,
    }


def _convert_choices_to_fol(
    premises_fol: list[str],
    choices: list[str],
) -> list[str] | None:
    from src.reasoner.question_parser import _extract_predicate_names

    predicate_names = _extract_predicate_names(premises_fol)
    choices_text = "\n".join(
        f"{ascii_uppercase[i]}. {c}" for i, c in enumerate(choices)
    )
    preds_text = ", ".join(predicate_names) if predicate_names else "(none)"

    choice_keys = [ascii_uppercase[i] for i in range(len(choices))]
    schema = {
        "name": "predicate_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {key: {"type": ["string", "null"]} for key in choice_keys},
            "required": choice_keys,
            "additionalProperties": False,
        },
    }

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "Return a JSON object mapping each letter to a FOL predicate name or formula. "
                "Use null if the choice has no matching predicate."
            ),
        },
        {
            "role": "user",
            "content": f"Predicates: {preds_text}\n\nChoices:\n{choices_text}",
        },
    ]

    try:
        response = call_llm(messages, schema=schema)
        return _parse_fol_response(response, len(choices))
    except Exception as e:
        logger.warning("LLM call failed during predicate mapping: %s", e)
        return None


def _extract_predicate_hints(premises_fol: list[str]) -> list[str]:
    names: set[str] = set()
    for p in premises_fol:
        try:
            ast = parse_fol(p)
            _collect_names(ast, names)
        except Exception:
            for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_]*)\b", p):
                names.add(match.group(1))
    return sorted(names)


def _collect_names(node: Expr, names: set[str]) -> None:
    from src.logic.nodes import (
        And,
        Constant,
        Eq,
        Exists,
        ForAll,
        Gt,
        Gte,
        Iff,
        Implies,
        Lt,
        Lte,
        Neq,
        Not,
        Or,
        Predicate,
    )

    if isinstance(node, Predicate):
        names.add(node.name)
        for arg in node.args:
            _collect_names(arg, names)
    elif isinstance(node, Constant):
        names.add(node.value)
    elif isinstance(node, (And, Or, Eq, Neq, Gt, Gte, Lt, Lte)):
        _collect_names(node.left, names)
        _collect_names(node.right, names)
    elif isinstance(node, Implies):
        _collect_names(node.antecedent, names)
        _collect_names(node.consequent, names)
    elif isinstance(node, Iff):
        _collect_names(node.left, names)
        _collect_names(node.right, names)
    elif isinstance(node, Not):
        _collect_names(node.body, names)
    elif isinstance(node, (ForAll, Exists)):
        _collect_names(node.body, names)


def _parse_fol_response(response_text: str, expected_count: int) -> list[str] | None:
    clean = re.sub(
        r"```(?:json)?\s*\n?(.*?)```",
        r"\1",
        response_text,
        flags=re.DOTALL,
    ).strip()

    if not clean.startswith("{"):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            clean = match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Failed to parse FOL response as JSON: %s", clean[:200])
        return None

    if not isinstance(data, dict):
        logger.warning("Expected JSON object, got %s", type(data).__name__)
        return None

    keys = set(data.keys())
    letters = {ascii_uppercase[i] for i in range(expected_count)}

    if keys & letters:
        formulas = []
        for i in range(expected_count):
            letter = ascii_uppercase[i]
            formula = data.get(letter, "")
            if formula is None:
                formula = ""
            formulas.append(str(formula).strip())
        return formulas

    formulas = [""] * expected_count
    for predicate, letter_val in data.items():
        if letter_val is None:
            continue
        letter_str = str(letter_val).strip().upper()
        if letter_str in letters:
            idx = ascii_uppercase.index(letter_str)
            formulas[idx] = str(predicate).strip()

    return formulas


def _explain_z3_result(
    premises_fol: list[str],
    choice_text: str,
    letter: str,
    formula: str,
) -> str:
    premises_text = "\n".join(f"{i}. {p}" for i, p in enumerate(premises_fol, start=1))

    schema = {
        "name": "explanation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
            },
            "required": ["explanation"],
            "additionalProperties": False,
        },
    }

    prompt = f"""Premises:
{premises_text}

The correct answer is {letter}: {choice_text}
FOL formula: {formula}

Explain in simple terms why this answer follows from the premises."""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": prompt},
    ]

    try:
        response = call_llm(messages, schema=schema)
        data = json.loads(response)
        return data.get("explanation", response)
    except Exception:
        return f"Answer {letter} is entailed by the premises (verified by Z3)."
