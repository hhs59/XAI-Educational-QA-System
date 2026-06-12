from __future__ import annotations

import re
from pathlib import Path

from lark import Lark, Transformer

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

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
_GRAMMAR = _GRAMMAR_PATH.read_text()
_parser = Lark(_GRAMMAR, parser="lalr")


def _preprocess(text: str) -> str:
    text = _replace_quantifier(text, "ForAll", "∀")
    text = _replace_quantifier(text, "Exists", "∃")
    text = re.sub(r"(∃\w+)\s*,", r"\1", text)
    text = re.sub(r"\band\b", "∧", text)
    text = re.sub(r"\bor\b", "∨", text)
    text = re.sub(r"\bnot\b", "¬", text)
    text = re.sub(r"\biff\b", "↔", text)
    return text


def _replace_quantifier(text: str, keyword: str, symbol: str) -> str:
    result = text
    pattern = re.compile(rf"{keyword}\s*\(")
    while True:
        match = pattern.search(result)
        if not match:
            break
        start = match.end()
        comma_pos = result.index(",", start)
        var_name = result[start:comma_pos].strip()
        paren_start = comma_pos + 1
        body_end = _find_matching_paren(result, match.start() + len(keyword))
        body = result[paren_start:body_end].strip()
        replacement = f"{symbol}{var_name} ({body})"
        result = result[: match.start()] + replacement + result[body_end + 1 :]
    return result


def _find_matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching closing paren at position {open_pos}")


class FOLTransformer(Transformer):
    def implies(self, children: list) -> Implies:
        left, _op, right = children
        return Implies(left, right)

    def iff_(self, children: list) -> Iff:
        left, _op, right = children
        return Iff(left, right)

    def or_(self, children: list) -> Or:
        left, _op, right = children
        return Or(left, right)

    def and_(self, children: list) -> And:
        left, _op, right = children
        return And(left, right)

    def not_(self, children: list) -> Not:
        _op, body = children
        return Not(body)

    def predicate(self, children: list) -> Predicate:
        name = str(children[0])
        args = children[1]
        return Predicate(name, args)

    def args(self, children: list) -> list[Expr]:
        return list(children)

    def term_ref(self, children: list) -> Expr:
        name = str(children[0])
        if len(name) == 1 and name.islower():
            return Variable(name)
        return Constant(name)

    def term_number(self, children: list) -> Constant:
        return Constant(str(children[0]))

    def term_string(self, children: list) -> Constant:
        val = str(children[0])
        return Constant(val[1:-1])

    def forall_node(self, children: list) -> ForAll:
        _qtok, cname, body = children
        return ForAll(str(cname), body)

    def exists_node(self, children: list) -> Exists:
        _qtok, cname, body = children
        return Exists(str(cname), body)

    def eq(self, children: list) -> Eq:
        left, _op, right = children
        return Eq(left, right)

    def neq(self, children: list) -> Neq:
        left, _op, right = children
        return Neq(left, right)

    def gt(self, children: list) -> Gt:
        left, _op, right = children
        return Gt(left, right)

    def gte(self, children: list) -> Gte:
        left, _op, right = children
        return Gte(left, right)

    def lt(self, children: list) -> Lt:
        left, _op, right = children
        return Lt(left, right)

    def lte(self, children: list) -> Lte:
        left, _op, right = children
        return Lte(left, right)


def parse_fol(text: str) -> Expr:
    preprocessed = _preprocess(text)
    tree = _parser.parse(preprocessed)
    return FOLTransformer().transform(tree)
