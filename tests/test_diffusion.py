import numpy as np
import pandas as pd

from macro_forecast_engine.modal_engine import _volatility_envelope


def test_tft_sqrt_time_diffusion_has_realistic_day_252_corridor():
    median = np.full(252, 4000.0)
    historical_returns = pd.Series(
        np.full(500, 0.175 / np.sqrt(252.0)),
        dtype=float,
    )
    lower, upper = _volatility_envelope(median, historical_returns, 0.0)
    assert 2800.0 <= lower[-1] <= 3200.0
    assert 4800.0 <= upper[-1] <= 5600.0
    assert np.all(lower < median)
    assert np.all(upper > median)
