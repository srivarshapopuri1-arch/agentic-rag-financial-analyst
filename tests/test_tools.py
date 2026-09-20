import pytest

from financial_analyst.tools import safe_calculate


def test_basic_financial_arithmetic():
    result = safe_calculate("(14.1 - 12.4) / 12.4 * 100")
    assert result == pytest.approx(13.7096774194)


def test_rejects_code_execution():
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('echo unsafe')")


def test_rejects_large_exponent():
    with pytest.raises(ValueError):
        safe_calculate("2 ** 100")
