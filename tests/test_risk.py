import math

import pytest

from macro_forecast_engine.trader_risk import (
    GOLD_PIP_SIZE,
    calculate_lot_size,
    size_position,
)


@pytest.mark.parametrize("equity", [0, 1, 100, 1000, 10000, 100000])
@pytest.mark.parametrize("risk", [0, 0.5, 1, 2, 5])
def test_lot_size_is_finite_and_nonnegative(equity, risk):
    value = calculate_lot_size(equity, risk, 100)
    assert math.isfinite(value)
    assert value >= 0


@pytest.mark.parametrize("risk", [0.5, 1, 2, 4])
def test_lot_size_scales_with_risk(risk):
    low = calculate_lot_size(10000, risk, 100)
    high = calculate_lot_size(10000, risk * 2, 100)
    assert high >= low
    assert high / low == pytest.approx(2, rel=0.01)


@pytest.mark.parametrize("stop", [1, 10, 50, 100, 500])
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
    assert result["units"] >= 0


def test_gold_pip_definition():
    assert GOLD_PIP_SIZE == pytest.approx(0.10)
    assert calculate_lot_size(10000, 1, 100) == pytest.approx(1.0)
