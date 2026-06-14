from __future__ import annotations

import re

import z3

from src.logic.nodes import (
    And,
    Constant,
    Eq,
    Exists,
    Expr,
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
    Variable,
)


class Z3Context:
    def __init__(self) -> None:
        self.entity_sort = z3.DeclareSort("Entity")
        self.int_sort = z3.IntSort()
        self._functions: dict[str, z3.FuncDeclRef] = {}
        self._constants: dict[str, z3.ExprRef] = {}
        self._numeric_predicates: set[str] = set()

    def mark_numeric(self, name: str) -> None:
        self._numeric_predicates.add(name)

    def is_numeric(self, name: str) -> bool:
        return name in self._numeric_predicates

    def get_function(self, name: str, arity: int) -> z3.FuncDeclRef:
        if name not in self._functions:
            if self.is_numeric(name):
                sorts = [self.entity_sort] * arity + [self.int_sort]
            else:
                sorts = [self.entity_sort] * arity + [z3.BoolSort()]
            self._functions[name] = z3.Function(name, *sorts)
        return self._functions[name]

    def get_constant(self, name: str) -> z3.ExprRef:
        if name not in self._constants:
            self._constants[name] = z3.Const(name, self.entity_sort)
        return self._constants[name]

    def get_int_constant(self, name: str) -> z3.ArithRef:
        key = f"__int__{name}"
        if key not in self._constants:
            self._constants[key] = z3.Const(name, self.int_sort)
        return self._constants[key]


def _is_numeric_string(value: str) -> bool:
    return bool(re.match(r"^-?\d+(\.\d+)?$", value.strip()))


def _is_float_string(value: str) -> bool:
    return bool(re.match(r"^-?\d+\.\d+$", value.strip()))


def _collect_numeric_predicates(expr: Expr, ctx: Z3Context) -> None:
    if isinstance(expr, (Gt, Gte, Lt, Lte, Eq, Neq)):
        if isinstance(expr.left, Predicate):
            ctx.mark_numeric(expr.left.name)
        if isinstance(expr.right, Predicate):
            ctx.mark_numeric(expr.right.name)
    if isinstance(expr, And):
        _collect_numeric_predicates(expr.left, ctx)
        _collect_numeric_predicates(expr.right, ctx)
    if isinstance(expr, Or):
        _collect_numeric_predicates(expr.left, ctx)
        _collect_numeric_predicates(expr.right, ctx)
    if isinstance(expr, Implies):
        _collect_numeric_predicates(expr.antecedent, ctx)
        _collect_numeric_predicates(expr.consequent, ctx)
    if isinstance(expr, Iff):
        _collect_numeric_predicates(expr.left, ctx)
        _collect_numeric_predicates(expr.right, ctx)
    if isinstance(expr, Not):
        _collect_numeric_predicates(expr.body, ctx)
    if isinstance(expr, (ForAll, Exists)):
        _collect_numeric_predicates(expr.body, ctx)


def encode_premises(premises: list[Expr]) -> tuple[z3.Solver, Z3Context]:
    ctx = Z3Context()
    for p in premises:
        _collect_numeric_predicates(p, ctx)
    solver = z3.Solver()
    for p in premises:
        solver.add(encode_expr(p, ctx))
    return solver, ctx


def encode_expr(expr: Expr, ctx: Z3Context | None = None) -> z3.ExprRef:
    """Encode an AST node into a Z3 expression.

    Handles all 16 node types: Constant, Variable, Predicate, Not, And, Or,
    Implies, Iff, ForAll, Exists, Eq, Neq, Gt, Gte, Lt, Lte.

    Numeric constants are encoded as IntVal/RealVal. Predicates marked as
    numeric (via _collect_numeric_predicates) return IntSort and are wrapped
    with != 0 when used as boolean conditions.

    Args:
        expr: AST node to encode.
        ctx: Z3 context managing sorts, functions, and constants.
            Created if not provided.

    Returns:
        Z3 expression (BoolRef for formulas, ArithRef for terms).
    """
    if ctx is None:
        ctx = Z3Context()
    if isinstance(expr, Constant):
        if _is_float_string(expr.value):
            return z3.RealVal(expr.value)
        if _is_numeric_string(expr.value):
            return z3.IntVal(expr.value)
        return z3.Bool(expr.value)
    if isinstance(expr, And):
        return z3.And(encode_expr(expr.left, ctx), encode_expr(expr.right, ctx))
    if isinstance(expr, Or):
        return z3.Or(encode_expr(expr.left, ctx), encode_expr(expr.right, ctx))
    if isinstance(expr, Not):
        return z3.Not(encode_expr(expr.body, ctx))
    if isinstance(expr, Implies):
        return z3.Implies(
            encode_expr(expr.antecedent, ctx),
            encode_expr(expr.consequent, ctx),
        )
    if isinstance(expr, Iff):
        left = encode_expr(expr.left, ctx)
        right = encode_expr(expr.right, ctx)
        return left == right
    if isinstance(expr, ForAll):
        var = ctx.get_constant(expr.variable)
        body = encode_expr(expr.body, ctx)
        return z3.ForAll([var], body)
    if isinstance(expr, Exists):
        var = ctx.get_constant(expr.variable)
        body = encode_expr(expr.body, ctx)
        return z3.Exists([var], body)
    if isinstance(expr, Predicate):
        func = ctx.get_function(expr.name, len(expr.args))
        args = [_encode_term(arg, ctx) for arg in expr.args]
        result = func(*args)
        if ctx.is_numeric(expr.name):
            return result != z3.IntVal(0)
        return result
    if isinstance(expr, Variable):
        return ctx.get_constant(expr.name)
    if isinstance(expr, Eq):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left == right
    if isinstance(expr, Neq):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left != right
    if isinstance(expr, Gt):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left > right
    if isinstance(expr, Gte):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left >= right
    if isinstance(expr, Lt):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left < right
    if isinstance(expr, Lte):
        left = _encode_term_or_predicate(expr.left, ctx)
        right = _encode_term_or_predicate(expr.right, ctx)
        return left <= right
    raise ValueError(f"Cannot encode unknown node type: {type(expr).__name__}")


def _encode_term(expr: Expr, ctx: Z3Context) -> z3.ExprRef:
    if isinstance(expr, Constant):
        return ctx.get_constant(expr.value)
    if isinstance(expr, Variable):
        return ctx.get_constant(expr.name)
    if isinstance(expr, Predicate):
        func = ctx.get_function(expr.name, len(expr.args))
        args = [_encode_term(arg, ctx) for arg in expr.args]
        return func(*args)
    raise ValueError(
        f"Expected a term (Constant, Variable, or Predicate), got {type(expr).__name__}"
    )


def _encode_term_or_predicate(expr: Expr, ctx: Z3Context) -> z3.ArithRef:
    if isinstance(expr, Predicate):
        func = ctx.get_function(expr.name, len(expr.args))
        args = [_encode_term(arg, ctx) for arg in expr.args]
        return func(*args)
    if isinstance(expr, Constant):
        if _is_float_string(expr.value):
            return z3.RealVal(expr.value)
        if _is_numeric_string(expr.value):
            return z3.IntVal(expr.value)
        return ctx.get_int_constant(expr.value)
    if isinstance(expr, Variable):
        return ctx.get_int_constant(expr.name)
    raise ValueError(
        f"Expected a term, Variable, or Predicate, got {type(expr).__name__}"
    )
