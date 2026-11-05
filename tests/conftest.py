import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlc():
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=500, freq="D")
    close = 4000 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, len(index))))
    return pd.DataFrame({
        "gold_open": close,
        "gold_high": close + 2.0,
        "gold_low": close - 2.0,
        "gold_close": close,
    }, index=index)
