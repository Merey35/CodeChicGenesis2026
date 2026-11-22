from __future__ import annotations

import ast
import operator
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext

ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPR_LEN = 256


def evaluate_expression(expression: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("INVALID_EXPRESSION: Expression must be a non-empty string")
    if len(expression) > MAX_EXPR_LEN:
        raise ValueError("INVALID_EXPRESSION: Expression exceeds maximum length")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError("INVALID_EXPRESSION: Syntax error") from exc

    def _eval(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float, Decimal)):
                raise ValueError("INVALID_EXPRESSION: Only numeric literals allowed")
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp):
            op_fn = ALLOWED_UNARYOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("INVALID_EXPRESSION: Unsupported unary operator")
            return op_fn(_eval(node.operand))
        if isinstance(node, ast.BinOp):
            op_fn = ALLOWED_BINOPS.get(type(node.op))
            if op_fn is None:
                raise ValueError("INVALID_EXPRESSION: Unsupported binary operator")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("DIVISION_BY_ZERO: Division by zero")
            if isinstance(node.op, ast.Pow):
                if not right.is_finite():
                    raise ValueError("INVALID_EXPRESSION: Exponent must be finite")
                if right != right.to_integral_value() or abs(right) > 20:
                    raise ValueError("INVALID_EXPRESSION: Exponent out of bounds")
            res = op_fn(left, right)
            if not res.is_finite():
                raise ValueError("NUMERIC_OVERFLOW: Non-finite result")
            return res

        raise ValueError("INVALID_EXPRESSION: Disallowed construct")

    try:
        with localcontext() as ctx:
            ctx.prec = max(28, len(expression) * 20)
            ctx.rounding = ROUND_HALF_UP
            val = _eval(tree)
            norm = val.normalize()
            if norm == norm.to_integral():
                return format(norm.quantize(Decimal("1")), "f")
            return format(norm, "f")
    except (InvalidOperation, ValueError, ArithmeticError) as exc:
        raise ValueError("INVALID_EXPRESSION: Evaluation failed") from exc