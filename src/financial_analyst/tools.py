from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import tool

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    """Evaluate basic arithmetic without eval(), names, calls, or attributes."""
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is outside the allowed range.")
            return float(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](visit(node.operand)))
        raise ValueError("Only basic numeric arithmetic is allowed.")

    return visit(tree)


@tool
def calculator(expression: str) -> str:
    """Calculate an arithmetic expression, for example '(14.1-12.4)/12.4*100'."""
    try:
        return f"{safe_calculate(expression):,.6g}"
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation error: {exc}"
