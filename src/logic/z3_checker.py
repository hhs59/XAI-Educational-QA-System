from __future__ import annotations

import z3

from src.logic.nodes import Expr
from src.logic.z3_encoder import Z3Context, encode_expr, encode_premises


def check_entailment(premises: list[Expr], query: Expr) -> dict:
    ctx = Z3Context()
    for p in premises:
        from src.logic.z3_encoder import _collect_numeric_predicates
        _collect_numeric_predicates(p, ctx)
    _collect_numeric_predicates(query, ctx)

    solver = z3.Solver()
    for premise in premises:
        solver.add(encode_expr(premise, ctx))
    solver.add(z3.Not(encode_expr(query, ctx)))

    result = solver.check()

    if result == z3.unsat:
        return {"entailed": True, "status": "unsat", "counterexample": None}

    if result == z3.sat:
        return {
            "entailed": False,
            "status": "sat",
            "counterexample": _extract_counterexample(solver),
        }

    return {"entailed": None, "status": "unknown", "counterexample": None}


def _extract_counterexample(solver: z3.Solver) -> dict:
    model = solver.model()
    assignments = {}
    for decl in model:
        name = str(decl)
        value = model[decl]
        assignments[name] = str(value)
    return assignments


def check_entailment_batch(
    premises: list[Expr],
    queries: list[Expr],
) -> list[dict]:
    ctx = Z3Context()
    for p in premises:
        from src.logic.z3_encoder import _collect_numeric_predicates
        _collect_numeric_predicates(p, ctx)
    for q in queries:
        _collect_numeric_predicates(q, ctx)

    results = []
    for query in queries:
        solver = z3.Solver()
        for premise in premises:
            solver.add(encode_expr(premise, ctx))
        solver.add(z3.Not(encode_expr(query, ctx)))

        result = solver.check()
        if result == z3.unsat:
            results.append({"entailed": True, "status": "unsat"})
        elif result == z3.sat:
            results.append({
                "entailed": False,
                "status": "sat",
                "counterexample": _extract_counterexample(solver),
            })
        else:
            results.append({"entailed": None, "status": "unknown"})

    return results
