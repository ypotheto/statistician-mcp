from __future__ import annotations

import pandas as pd
import pytest

from statistician_mcp.utils.formulas import FormulaError, evaluate_expression

DF = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


def test_arithmetic_expression() -> None:
    result = evaluate_expression(DF, "a * 2 + b")
    assert result.tolist() == [6.0, 9.0, 12.0]


def test_boolean_expression() -> None:
    result = evaluate_expression(DF, "(a > 1) & (b < 6)")
    assert result.tolist() == [False, True, False]


def test_allowed_functions() -> None:
    result = evaluate_expression(DF, "sqrt(a)")
    assert result.tolist() == pytest.approx([1.0, 2**0.5, 3**0.5])


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "a.__class__",
        "[x for x in range(3)]",
        "lambda x: x",
        "unknown_column + 1",
        "open('/etc/passwd')",
        "a if b else 0",
    ],
)
def test_malicious_or_disallowed_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(FormulaError):
        evaluate_expression(DF, expression)
