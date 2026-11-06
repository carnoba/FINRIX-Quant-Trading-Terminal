import math

import pytest

from macro_forecast_engine.trader_risk import (
    GOLD_CONTRACT_SIZE_OUNCES,
    GOLD_PIP_VALUE_PER_STANDARD_LOT,
    GOLD_PIP_SIZE,
    calculate_lot_size,
    size_position,
)


@pytest.mark.parametrize("equity", [0, 1, 100, 1000, 10000, 100000])
@pytest.mark.parametrize("risk", [0, 0.5, 1, 2, 5])
def test_lot_size_is_finite_and_nonnegative(equity, risk):
    value = calculate_lot_size(equity, risk, 20)
    assert math.isfinite(value)
    expected = 0.0 if equity == 0 else max(0.01, equity * risk / 100.0 / 2000.0)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("risk", [0.5, 1, 2, 4])
def test_lot_size_scales_with_risk(risk):
    low = calculate_lot_size(10000, risk, 20)
    high = calculate_lot_size(10000, risk * 2, 20)
    assert high >= low
    assert high / low == pytest.approx(2, rel=0.01)


@pytest.mark.parametrize("stop", [20, 50, 100, 200, 500])
def test_lot_size_reduces_with_stop_distance(stop):
    near = calculate_lot_size(10000, 1, stop)
    far = calculate_lot_size(10000, 1, stop * 2)
    assert near >= far
    assert near > 0


@pytest.mark.parametrize("entry,q10,q50,q90", [
    (4000, 3900, 4020, 4200),
    (4000, 3999, 4000, 4001),
    (1, 0.5, 1, 2),
    (0, 0, 0, 0),
])
def test_position_sizing_guards(entry, q10, q50, q90):
    result = size_position(entry, q10, q50, q90, 10000, risk_fraction=0.01)
    assert math.isfinite(result["reward_risk"])
    assert math.isfinite(result["allocation"])
    assert result["units"] == pytest.approx(result["ounces"])
    assert result["lot_size"] == pytest.approx(result["ounces"] / 100.0)


def test_gold_pip_definition():
    assert GOLD_PIP_SIZE == pytest.approx(0.10)
    assert GOLD_CONTRACT_SIZE_OUNCES == pytest.approx(100.0)
    assert GOLD_PIP_VALUE_PER_STANDARD_LOT == pytest.approx(10.0)
    assert calculate_lot_size(10000, 1, 20) == pytest.approx(0.05)


def test_lot_size_stop_loss_dollar_risk_is_exact():
    lots = calculate_lot_size(10000, 1.0, 20.0)
    loss = lots * 20.0 * GOLD_CONTRACT_SIZE_OUNCES
    assert loss == pytest.approx(100.0)
