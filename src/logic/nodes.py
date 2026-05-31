from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Expr:
    pass


@dataclass
class Variable(Expr):
    name: str


@dataclass
class Constant(Expr):
    value: str


@dataclass
class Predicate(Expr):
    name: str
    args: list[Expr]


@dataclass
class Not(Expr):
    body: Expr


@dataclass
class And(Expr):
    left: Expr
    right: Expr


@dataclass
class Or(Expr):
    left: Expr
    right: Expr


@dataclass
class Implies(Expr):
    antecedent: Expr
    consequent: Expr


@dataclass
class ForAll(Expr):
    variable: str
    body: Expr


@dataclass
class Exists(Expr):
    variable: str
    body: Expr


@dataclass
class Eq(Expr):
    left: Expr
    right: Expr


@dataclass
class Neq(Expr):
    left: Expr
    right: Expr


@dataclass
class Gt(Expr):
    left: Expr
    right: Expr


@dataclass
class Gte(Expr):
    left: Expr
    right: Expr


@dataclass
class Lt(Expr):
    left: Expr
    right: Expr


@dataclass
class Lte(Expr):
    left: Expr
    right: Expr
