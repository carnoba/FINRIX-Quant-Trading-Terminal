import numpy as np
import pandas as pd

from macro_forecast_engine.modal_engine import _volatility_envelope
from macro_forecast_engine.modal_engine import _tft_quantile_paths


def test_tft_sqrt_time_diffusion_has_realistic_day_252_corridor():
    median = np.full(252, 4000.0)
    historical_returns = pd.Series(
        np.resize(np.array([-0.011, 0.011], dtype=float), 500),
        dtype=float,
    )
    lower, upper = _volatility_envelope(median, historical_returns, 0.0)
    assert 2800.0 <= lower[-1] <= 3200.0
    assert 4800.0 <= upper[-1] <= 5600.0
    assert np.all(lower < median)
    assert np.all(upper > median)


def test_tft_quantile_paths_direct():
    returns = np.tile([[-0.01, 0.0, 0.01]], (252, 1))
    historical = pd.Series(np.full(252, 0.01), dtype=float)
    lower, median, upper = _tft_quantile_paths(4000.0, returns, historical)
    assert np.all(lower < median)
    assert np.all(median < upper)
    assert np.isclose(median[-1], 4000.0)
    expected = 4000.0 * np.exp(-1.2816 * np.sqrt(252.0) * (0.01 / 1.2816))
    np.testing.assert_allclose(lower[-1], expected, rtol=1e-10)
