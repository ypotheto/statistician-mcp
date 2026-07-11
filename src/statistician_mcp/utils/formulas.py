from __future__ import annotations

import ast

import numpy as np
import pandas as pd

_ALLOWED_FUNCS = {"log": np.log, "log10": np.log10, "sqrt": np.sqrt, "exp": np.exp, "abs": np.abs}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.BitAnd, ast.BitOr)
_ALLOWED_UNARYOPS = (ast.USub, ast.UAdd, ast.Invert)
_ALLOWED_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


class FormulaError(ValueError):
    pass


def evaluate_expression(df: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate a restricted arithmetic/boolean expression against `df`'s columns.

    Only column names, numeric literals, `+ - * /`, comparisons, `& | ~`, and the
    functions log/log10/sqrt/exp/abs are permitted — this is the whole safety
    argument for `transform_dataset`, so no other AST node type is allowed through.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(f"could not parse expression: {exc}") from exc

    columns = set(df.columns.astype(str))
    _validate(tree, columns)

    namespace: dict[str, object] = {**_ALLOWED_FUNCS, **{col: df[col] for col in df.columns}}
    code = compile(tree, "<expression>", mode="eval")
    return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - AST-validated above


def _validate(node: ast.AST, columns: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate(node.body, columns)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise FormulaError(f"operator '{type(node.op).__name__}' is not allowed")
        _validate(node.left, columns)
        _validate(node.right, columns)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise FormulaError(f"operator '{type(node.op).__name__}' is not allowed")
        _validate(node.operand, columns)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], _ALLOWED_CMPOPS):
            raise FormulaError("only a single simple comparison is allowed per term")
        _validate(node.left, columns)
        _validate(node.comparators[0], columns)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise FormulaError("only log, log10, sqrt, exp, and abs may be called")
        if node.keywords:
            raise FormulaError("keyword arguments are not allowed")
        for arg in node.args:
            _validate(arg, columns)
    elif isinstance(node, ast.Name):
        if node.id not in columns and node.id not in _ALLOWED_FUNCS:
            raise FormulaError(f"unknown column '{node.id}'")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, int | float) or isinstance(node.value, bool):
            raise FormulaError("only numeric literals are allowed")
    else:
        raise FormulaError(f"expression contains a disallowed construct: {type(node).__name__}")
