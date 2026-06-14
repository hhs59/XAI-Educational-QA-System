import json
import logging
import re

from openai.types.chat import ChatCompletionMessageParam

from src.reasoner.llm_client import call_llm

logger = logging.getLogger(__name__)

FORMULA_FAMILIES = {
    "capacitor_energy": {
        "keywords": ["capacitor", "energy", "stored"],
        "formulas": ["E = 0.5 * C * U^2", "Q = C * U"],
    },
    "inductor_energy": {
        "keywords": ["inductor", "inductance", "magnetic field energy"],
        "formulas": ["E = 0.5 * L * I^2"],
    },
    "coulomb": {
        "keywords": ["charge", "coulomb", "force between"],
        "formulas": ["F = k * q1 * q2 / r^2", "k = 8.99e9 N*m^2/C^2"],
    },
    "solenoid": {
        "keywords": ["solenoid"],
        "formulas": ["B = mu0 * N * I / L", "L_ind = mu0 * N^2 * A / l"],
    },
    "rlc": {
        "keywords": ["resonan", "RLC", "impedance"],
        "formulas": ["Z = sqrt(R^2 + (XL - XC)^2)", "at resonance Z = R"],
    },
    "electric_field": {
        "keywords": ["electric field", "field strength"],
        "formulas": ["E = F / q", "E = k * q / r^2"],
    },
    "ohm_law": {
        "keywords": ["current", "resistance", "voltage"],
        "formulas": ["V = I * R", "P = V * I"],
    },
}


def _identify_family(question: str) -> list[str]:
    q = question.lower()
    matches = []
    for name, info in FORMULA_FAMILIES.items():
        score = sum(1 for kw in info["keywords"] if kw in q)
        if score >= 1:
            matches.append((score, name))
    matches.sort(reverse=True)
    return [name for _, name in matches]


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 1:
        if value == int(value):
            return str(int(value))
        return str(round(value, 4))
    return str(round(value, 6))


def _llm_solve(question: str, families: list[str]) -> dict:
    family_info = ""
    if families:
        family_info = "Relevant formulas:\n"
        for fname in families[:2]:
            info = FORMULA_FAMILIES[fname]
            family_info += "  " + ", ".join(info["formulas"]) + "\n"

    schema = {
        "name": "physics_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["answer", "unit"],
            "additionalProperties": False,
        },
    }

    prompt = f"""Solve this physics problem. Give ONLY the numeric answer and unit.

{family_info}
Question: {question}

Return the answer as a number (no scientific notation, no sqrt, no expressions). If the answer is 0.045 J, return "0.045" and "J".
Example: {{"answer": "45", "unit": "J"}}"""

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    try:
        response = call_llm(messages, schema=schema)
        clean = re.sub(
            r"```(?:json)?\s*\n?(.*?)```", r"\1", response, flags=re.DOTALL
        ).strip()
        if not clean.startswith("{"):
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if match:
                clean = match.group(0)
        data = json.loads(clean)
        answer = str(data.get("answer", "unknown")).strip()
        unit = str(data.get("unit", "")).strip()
        return {
            "answer": answer,
            "unit": unit,
            "explanation": "",
            "confidence": 0.7,
        }
    except Exception as e:
        logger.warning("LLM physics solve failed: %s", e)
        return {
            "answer": "unknown",
            "unit": "",
            "explanation": f"Could not solve: {e}",
            "confidence": 0.0,
        }


def solve_physics(question: str) -> dict:
    families = _identify_family(question)
    return _llm_solve(question, families)
