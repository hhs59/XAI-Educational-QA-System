import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable

import sympy as sp
from openai.types.chat import ChatCompletionMessageParam

from src.reasoner.llm_client import call_llm

logger = logging.getLogger(__name__)


def _si_micro(x: float) -> float:
    return x * 1e-6


def _si_milli(x: float) -> float:
    return x * 1e-3


def _si_nano(x: float) -> float:
    return x * 1e-9


def _si_pico(x: float) -> float:
    return x * 1e-12


def _si_centi(x: float) -> float:
    return x * 0.01


def _si_kilo(x: float) -> float:
    return x * 1e3


def _si_pass(x: float) -> float:
    return x


UNIT_TO_SI: dict[str, Callable[[float], float]] = {
    "μF": _si_micro, "pF": _si_pico, "nF": _si_nano, "mF": _si_milli, "F": _si_pass,
    "μC": _si_micro, "nC": _si_nano, "mC": _si_milli, "C": _si_pass,
    "μH": _si_micro, "mH": _si_milli, "H": _si_pass,
    "cm": _si_centi, "mm": _si_milli, "m": _si_pass, "km": _si_kilo,
    "mV": _si_milli, "V": _si_pass, "kV": _si_kilo,
    "mA": _si_milli, "A": _si_pass,
    "mJ": _si_milli, "J": _si_pass, "nJ": _si_nano,
    "mW": _si_milli, "W": _si_pass, "kW": _si_kilo,
    "kHz": _si_kilo, "ms": _si_milli, "s": _si_pass,
}


SI_TO_OUTPUT: dict[str, Callable[[float], tuple[float, str]]] = {
    "J": lambda x: (x * 1000, "mJ") if x < 1 else (x * 1e6, "μJ") if x < 1e-3 else (x, "J"),
    "F": lambda x: (x * 1e6, "μF") if x < 1 else (x * 1e12, "pF") if x < 1e-6 else (x, "F"),
    "H": lambda x: (x * 1000, "mH") if x < 1 else (x, "H"),
    "Ω": lambda x: (x, "Ω"),
    "N": lambda x: (x, "N"),
    "T": lambda x: (x, "T"),
    "V/m": lambda x: (x, "V/m"),
    "%": lambda x: (x, "%"),
    "Hz": lambda x: (x * 1e-3, "kHz") if x >= 1000 else (x, "Hz"),
    "W": lambda x: (x, "W"),
    "V": lambda x: (x, "V"),
    "A": lambda x: (x, "A"),
}

FORMULA_OUTPUT: dict[str, dict[str, str]] = {
    "TD": {"energy": "J", "capacitance": "F", "charge": "C", "voltage": "V"},
    "CH": {"resonance_impedance": "Ω", "power": "W", "impedance": "Ω",
           "inductive_reactance": "Ω", "capacitive_reactance": "Ω", "resonant_frequency": "Hz"},
    "LD": {"force_vector_same_dir": "N", "force_vector_angle": "N",
           "coulomb_force": "N", "coulomb_field": "V/m", "force_field": "V/m"},
    "DDT": {"magnetic_field": "T", "solenoid_field": "T", "inductance": "H"},
    "NL": {"inductor_energy": "J", "capacitor_energy": "J"},
    "DT": {"point_charge_field": "V/m", "field_from_force": "V/m"},
    "THCB": {"error_relative": "%"},
    "CHLT": {},
}


@dataclass
class FormulaFamily:
    name: str
    formulas: dict[str, str]
    extractors: dict[str, str]
    formula_selector: Callable[[dict[str, float]], str] | None = None
    output_unit: str = ""

    def select_formula(self, values: dict[str, float]) -> str | None:
        if self.formula_selector:
            return self.formula_selector(values)
        return next(iter(self.formulas.values()), None)


def _capacitor_selector(values: dict[str, float]) -> str | None:
    has = set(values.keys())
    if "C" in has and "U" in has:
        return "energy"
    if "Q" in has and "U" in has:
        return "capacitance"
    if "Q" in has and "C" in has:
        return "voltage"
    return next(iter(values.keys())) if has else None


def _force_selector(values: dict[str, float]) -> str | None:
    has = set(values.keys())
    if "theta" in has and "F1" in has and "F2" in has:
        return "force_vector_angle"
    if "F1" in has and "F2" in has:
        return "force_vector_same_dir"
    if "q1" in has and "q2" in has and "r" in has:
        return "coulomb_force"
    if "q" in has and "r" in has:
        return "coulomb_field"
    return "force_vector_same_dir"


FORMULA_REGISTRY: dict[str, FormulaFamily] = {
    "TD": FormulaFamily(
        name="capacitor",
        formulas={
            "energy": "E = 0.5 * C * U**2",
            "capacitance": "C = Q / U",
            "charge": "Q = C * U",
            "voltage": "U = Q / C",
        },
        extractors={
            "C": r"(?:\bC\b)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(μF|pF|nF|F)",
            "U": r"(?:\bU\b|voltage|potential)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(V|kV)",
            "Q": r"(?:\bQ\b|charge)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(C|mC|μC|nC)",
            "E": r"(?:energy|E)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(J|mJ|nJ)",
        },
        formula_selector=_capacitor_selector,
        output_unit="J",
    ),
    "CH": FormulaFamily(
        name="rlc_circuit",
        formulas={
            "resonance_impedance": "Z = R",
            "power": "P = U**2 / R",
            "impedance": "Z = sqrt(R**2 + (XL - XC)**2)",
            "inductive_reactance": "XL = 2 * pi * f * L",
            "capacitive_reactance": "XC = 1 / (2 * pi * f * C)",
            "resonant_frequency": "f0 = 1 / (2 * pi * sqrt(L * C))",
        },
        extractors={
            "R": r"(?:\bR\b|resistance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(Ω|kΩ)",
            "L": r"(?:\bL\b|inductance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(H|mH|μH)",
            "C": r"(?:\bC\b|capacitance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(μF|pF|F|nF)",
            "U": r"(?:\bU\b|V\b|voltage)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(V|kV)",
            "Z": r"(?:\bZ\b|impedance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(Ω|kΩ)",
            "f": r"(?:\bf\b|frequency)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(Hz|kHz)",
            "P": r"(?:\bP\b|power)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(W|kW)",
            "I": r"(?:\bI\b|current)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(A|mA)",
        },
        output_unit="Ω",
    ),
    "LD": FormulaFamily(
        name="coulomb_force",
        formulas={
            "force_vector_same_dir": "F = F1 + F2",
            "force_vector_angle": "F = sqrt(F1**2 + F2**2 + 2*F1*F2*cos(pi*theta/180))",
            "coulomb_force": "F = k * q1 * q2 / r**2",
            "coulomb_field": "E = k * q / r**2",
            "force_field": "E = F / q",
        },
        extractors={
            "q1": r"q1\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(?:×\s*10\^?[-–]?\d+\s*)?(C|μC|nC)",
            "q2": r"q2\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(?:×\s*10\^?[-–]?\d+\s*)?(C|μC|nC)",
            "q3": r"q3\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(?:×\s*10\^?[-–]?\d+\s*)?(C|μC|nC)",
            "r": r"(\d+(?:\.\d+)?)\s*(cm|m)\s*apart",
            "F": r"(?:\bF\b|force)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(N)",
        },
        formula_selector=_force_selector,
        output_unit="N",
    ),
    "DDT": FormulaFamily(
        name="solenoid",
        formulas={
            "magnetic_field": "B = mu0 * n * I",
            "solenoid_field": "B = mu0 * N * I / L",
            "inductance": "L_ind = mu0 * N**2 * A / l",
        },
        extractors={
            "N": r"(\d+)\s*(?:turns|vòng)",
            "L": r"(?:length|dài|dài\s+của)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(m|cm|mm)",
            "I": r"(?:\bI\b|current|dòng)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(A|mA)",
            "A": r"(?:area|diện tích)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(cm2|m2)",
            "n": r"(\d+)\s*(?:turns/m|vòng/m)",
        },
        output_unit="T",
    ),
    "NL": FormulaFamily(
        name="inductor_energy",
        formulas={
            "inductor_energy": "E = 0.5 * L * I**2",
            "capacitor_energy": "E = 0.5 * C * U**2",
        },
        extractors={
            "L": r"(?:\bL\b|inductance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(H|mH|μH)",
            "I": r"(?:\bI\b|current)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(A|mA)",
            "C": r"(?:\bC\b|capacitance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(μF|pF|F)",
            "U": r"(?:\bU\b|V\b|voltage)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(V)",
            "E": r"(?:energy|E)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(J|mJ|nJ)",
        },
        output_unit="mJ",
    ),
    "DT": FormulaFamily(
        name="electric_field",
        formulas={
            "point_charge_field": "E = k * q / r**2",
            "field_from_force": "E = F / q",
        },
        extractors={
            "q": r"q\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(?:×\s*10\^?[-–]?\d+\s*)?(C|μC|nC)",
            "r": r"(\d+(?:\.\d+)?)\s*(cm|m)(?:\s*apart|\s*cách)",
            "E": r"(?:field strength|E)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(V/m|N/C)",
        },
        output_unit="V/m",
    ),
    "THCB": FormulaFamily(
        name="measurement_error",
        formulas={
            "error_relative": "delta = abs(value - reference) / reference * 100",
        },
        extractors={
            "value": r"(?:value|measured|giá trị)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(A|V|Ω|W|Hz)",
            "reference": r"(?:reference|true|actual|thực)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(A|V|Ω|W|Hz)",
        },
        output_unit="%",
    ),
    "CHLT": FormulaFamily(
        name="resonance_check",
        formulas={
            "resonant_frequency": "f0 = 1 / (2 * pi * sqrt(L * C))",
            "reactance_check": "XL = 2 * pi * f * L; XC = 1 / (2 * pi * f * C)",
        },
        extractors={
            "R": r"\bR\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(Ω)",
            "L": r"(?:\bL\b|inductance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(H|mH)",
            "C": r"(?:\bC\b|capacitance)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(μF|F)",
            "f": r"(?:\bf\b|frequency)\s*[=: ]+\s*(\d+(?:\.\d+)?)\s*(Hz|kHz)",
        },
        output_unit="-",
    ),
}


_GENERIC_EXTRACTORS = {
    "C": r"\bC\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(μF|pF|nF|F)",
    "U": r"\b[UV]\b\s*(?:[=:]|(?:of|is|có|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(V|kV|mV)",
    "R": r"\bR\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(Ω|kΩ)",
    "I": r"\bI\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(A|mA)",
    "L": r"\bL\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(H|mH|μH)",
    "Q": r"\bQ\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(C|mC|μC|nC)",
    "f": r"\bf\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(Hz|kHz)",
    "Z": r"\bZ\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(Ω|kΩ)",
    "P": r"\bP\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(W|kW)",
    "E": r"\bE\b\s*(?:[=:]|(?:of|is|bằng|là)\s+)\s*(\d+(?:\.\d+)?)\s*(J|mJ|nJ)",
    "N": r"(\d+)\s*(?:turns|vòng)",
    "n": r"(\d+(?:\.\d+)?)\s*(?:turns/m|vòng/m)",
}


def _extract_values(question: str, family: str) -> dict[str, float]:
    family_obj = FORMULA_REGISTRY.get(family)
    if not family_obj:
        return {}
    values: dict[str, float] = {}

    for var_name in family_obj.extractors:
        if var_name in _GENERIC_EXTRACTORS:
            pattern = _GENERIC_EXTRACTORS[var_name]
        else:
            pattern = family_obj.extractors[var_name]
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1))
                unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                converter = UNIT_TO_SI.get(unit, _si_pass)
                values[var_name] = converter(raw)
            except (ValueError, IndexError):
                pass

    if family == "LD" and not values:
        values = _extract_force_magnitudes(question)

    return values


def _extract_force_magnitudes(question: str) -> dict[str, float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*N", question)
    angle_match = re.search(r"(\d+(?:\.\d+)?)\s*°", question)
    if len(matches) >= 2:
        vals: dict[str, float] = {
            "F1": float(matches[0]),
            "F2": float(matches[1]),
        }
        if angle_match:
            vals["theta"] = float(angle_match.group(1))
        return vals
    return {}


def _deterministic_solve(question: str) -> dict | None:
    family = _classify(question)
    if not family:
        return None

    values = _extract_values(question, family)
    if not values:
        return None

    family_obj = FORMULA_REGISTRY.get(family)
    if not family_obj:
        return None

    formula_key = family_obj.select_formula(values)
    if not formula_key:
        return None

    formula_str = family_obj.formulas.get(formula_key)
    if not formula_str:
        return None

    result = _eval_formula(formula_str, values, family)
    if result is None and formula_key != "energy":
        alt_key = None
        for k in ("energy", "power", "resonance_impedance", "force_vector_same_dir"):
            if k in family_obj.formulas and k != formula_key:
                alt_key = k
                break
        if alt_key:
            result = _eval_formula(family_obj.formulas[alt_key], values, family)
            if result:
                formula_key = alt_key
                formula_str = family_obj.formulas[alt_key]

    if not result:
        return None

    si_unit = family_obj.output_unit
    per_formula = FORMULA_OUTPUT.get(family, {})
    if formula_key in per_formula:
        si_unit = per_formula[formula_key]

    converter = SI_TO_OUTPUT.get(si_unit)
    if converter:
        converted_val, output_unit = converter(result)
    else:
        converted_val, output_unit = result, si_unit

    answer = _format_result(converted_val)

    return {
        "answer": answer,
        "unit": output_unit,
        "explanation": f"Computed {formula_key}: {formula_str} with values {values}",
        "confidence": 0.9,
    }


def _eval_formula(formula_str: str, values: dict[str, float], family: str) -> float | None:
    lhs, rhs = formula_str.split("=", 1)
    expr_str = rhs.strip()

    for old, new in [("pi", str(math.pi)), ("mu0", str(4 * math.pi * 1e-7)), ("k", "8.99e9")]:
        expr_str = expr_str.replace(old, new)

    local_dict: dict = {}
    sym_vars: dict[str, sp.Symbol] = {}
    for var_name in values:
        s = sp.Symbol(var_name, real=True)
        sym_vars[var_name] = s
        local_dict[var_name] = s
    for reserved in ("Q", "C", "I", "N", "E", "S", "O"):
        if reserved not in local_dict:
            local_dict[reserved] = sp.Symbol(reserved, real=True)
    local_dict.update({"sqrt": sp.sqrt, "cos": sp.cos, "sin": sp.sin,
                         "tan": sp.tan, "abs": sp.Abs, "pi": math.pi})

    try:
        sym_expr = sp.sympify(expr_str, locals=local_dict)
    except sp.SympifyError:
        logger.debug("SymPy parse failed: %s", expr_str)
        return None

    for var_name, val in values.items():
        if var_name in sym_vars:
            sym_expr = sym_expr.subs(sym_vars[var_name], val)

    try:
        return float(sym_expr.evalf())
    except (TypeError, AttributeError):
        logger.debug("SymPy evaluation failed for %s (unresolved symbols)", family)
        return None


def _format_result(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 100000:
        exp = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10 ** exp)
        return f"{mantissa:.4f} × 10^{exp}"
    if value == int(value):
        return str(int(value))
    if abs(value) >= 1:
        return str(round(value, 4))
    return str(round(value, 6))


def normalize_answer(answer: str) -> float | str | None:
    """Normalize a physics answer string to a float or pass it through as text.

    Handles: plain numbers, scientific notation (x/×/10^ variants),
    LaTeX sqrt expressions, Yes/No, Vietnamese text.
    """
    s = answer.strip()
    if not s:
        return None

    if s.lower() in ("yes", "no", "không", "có"):
        return s.lower()

    s = s.replace("×", "x").replace(",", ".")

    sqrt_match = re.search(r"(\d+(?:\.\d+)?)\s*\\sqrt\s*\{(\d+)\}", s)
    if sqrt_match:
        as_float = float(sqrt_match.group(1)) * math.sqrt(float(sqrt_match.group(2)))
        rest = s[sqrt_match.end():]
        match = re.search(r"x?\s*10\^?[-–]?(\d+)", rest)
        if match:
            as_float *= 10 ** (-int(match.group(1)))
        return as_float

    frac_match = re.match(r"^\s*\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}\s*$", s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            return num / den
        except ValueError:
            pass

    sci_match = re.match(
        r"(\d+(?:\.\d+)?)\s*(?:x|\.)\s*10\^?[-–]?(\d+)", s
    )
    if sci_match:
        return float(sci_match.group(1)) * 10 ** (-int(sci_match.group(2)))

    sci_match2 = re.match(
        r"(\d+(?:\.\d+)?)\s*(?:x|\.)\s*10\^(\d+)", s
    )
    if sci_match2:
        return float(sci_match2.group(1)) * 10 ** int(sci_match2.group(2))

    dot_sci_match = re.match(
        r"(\d+(?:\.\d+)?)\s*\.\s*10\^\{[-–]?(\d+)\}", s
    )
    if dot_sci_match:
        return float(dot_sci_match.group(1)) * 10 ** (-int(dot_sci_match.group(2)))

    sci_match3 = re.match(
        r"(\d+(?:\.\d+)?)\.10\^\{[-–]?(\d+)\}", s
    )
    if sci_match3:
        return float(sci_match3.group(1)) * 10 ** (-int(sci_match3.group(2)))

    sci_match4 = re.match(
        r"(\d+(?:\.\d+)?)\.10\^\{(\d+)\}", s
    )
    if sci_match4:
        return float(sci_match4.group(1)) * 10 ** int(sci_match4.group(2))

    try:
        return float(s)
    except ValueError:
        pass

    return s


def answers_match(predicted: str, gold: str, tolerance: float = 0.01) -> bool:
    """Check if predicted and gold physics answers match.

    Strings match exactly. Floats match within tolerance (default 1%).
    """
    norm_pred = normalize_answer(predicted)
    norm_gold = normalize_answer(gold)

    if norm_pred is None or norm_gold is None:
        return predicted.strip() == gold.strip()

    if isinstance(norm_pred, str) or isinstance(norm_gold, str):
        return str(norm_pred).lower() == str(norm_gold).lower()

    if isinstance(norm_pred, (int, float)) and isinstance(norm_gold, (int, float)):
        if norm_gold == 0:
            return abs(norm_pred) < 1e-9
        return abs(norm_pred - norm_gold) <= tolerance * abs(norm_gold)

    return predicted.strip() == gold.strip()


def _classify(question: str) -> str | None:
    q = question.lower()
    patterns = [
        ("CHLT", r"(?:resonance\s*occur|does\s+resonance|experience\s+resonance)"),
        ("TD", r"(?:capacitor|c=.*\d+\s*(?:μ|p|n|m)?f|voltage.*\d+\s*v)"),
        ("CH", r"(?:rlc|resonan|impedance|\d+\s*Ω.*\d+\s*Ω)"),
        ("DDT", r"(?:solenoid|magnetic\s+field.*inside|inductance.*solenoid)"),
        ("NL", r"(?:energy.*stored.*inductor|energy.*stored.*capacitor|inductor.*current)"),
        ("DT", r"(?:electric\s+field.*strength|field.*point\s+charge|field strength)"),
        ("LD", r"(?:q1\s*=|q2\s*=|charges?.*coulomb|force.*magnitude|forces?.*resultant|resultant.*force|forces.*act)"),
        ("THCB", r"(?:ammeter|voltmeter|least\s+count|measurement.*error|measuring\s+range)"),
    ]
    for prefix, pattern in patterns:
        if re.search(pattern, q):
            return prefix
    return None


def _get_example(family: str | None) -> dict | None:
    if not family:
        return None
    try:
        from pathlib import Path
        path = Path("data/physics_examples.json")
        if not path.exists():
            return None
        with path.open("r") as f:
            examples = json.load(f)
        return examples.get(family)
    except Exception:
        return None


def _cot_guided_solve(question: str, family: str) -> dict | None:
    example = _get_example(family)
    if not example:
        return None

    schema = {
        "name": "physics_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["answer", "unit"],
            "additionalProperties": False,
        },
    }

    prompt = f"""Here is a solved example of a similar physics problem:

Example question: {example['question']}
Example solution: {example['cot']}
Example answer: {example['answer']} {example.get('unit', '')}

Now solve this problem following the exact same step-by-step method:

Question: {question}

Follow the same reasoning pattern, formula, unit conversions, and calculation steps.
Return the numeric answer: {{"answer": "value", "unit": "unit"}}"""

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    try:
        response = call_llm(messages, schema=schema)
        clean = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", response, flags=re.DOTALL).strip()
        if not clean.startswith("{"):
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if match:
                clean = match.group(0)
        data = json.loads(clean)
        return {
            "answer": str(data.get("answer", "unknown")).strip(),
            "unit": str(data.get("unit", "")).strip(),
            "explanation": "CoT-guided LLM solve",
            "confidence": 0.8,
        }
    except Exception as e:
        logger.debug("CoT-guided solve failed: %s", e)
        return None


def _llm_solve(question: str) -> dict:
    families = _classify(question)
    family_obj = FORMULA_REGISTRY.get(families or "", None)
    formula_hint = ""
    if family_obj:
        formula_hint = "Relevant formulas:\n  " + "\n  ".join(
            f"{k}: {v}" for k, v in family_obj.formulas.items()
        ) + "\n\n"

    schema = {
        "name": "physics_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["answer", "unit"],
            "additionalProperties": False,
        },
    }

    prompt = f"""Solve this physics problem. Give ONLY the numeric answer and unit.

{formula_hint}Question: {question}

Return the answer as a number. Example: {{"answer": "45", "unit": "J"}}"""

    messages: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]

    try:
        response = call_llm(messages, schema=schema)
        clean = re.sub(r"```(?:json)?\s*\n?(.*?)```", r"\1", response, flags=re.DOTALL).strip()
        if not clean.startswith("{"):
            match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
            if match:
                clean = match.group(0)
        data = json.loads(clean)
        return {
            "answer": str(data.get("answer", "unknown")).strip(),
            "unit": str(data.get("unit", "")).strip(),
            "explanation": "",
            "confidence": 0.7,
        }
    except Exception as e:
        logger.warning("LLM physics solve failed: %s", e)
        return {"answer": "unknown", "unit": "", "explanation": f"Failed: {e}", "confidence": 0.0}


def solve_physics(question: str) -> dict:
    result = _deterministic_solve(question)
    if result:
        return result

    family = _classify(question)
    result = _cot_guided_solve(question, family)
    if result:
        return result

    return _llm_solve(question)
