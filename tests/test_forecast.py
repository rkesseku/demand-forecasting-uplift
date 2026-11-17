import numpy as np
import pytest

from demand_uplift.data import generate
from demand_uplift.forecast import BacktestConfig, backtest, coverage, mase, smape


def test_mase_known_values():
    train = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # seasonal-naive (m=7) MAE = 7
    assert np.isclose(mase(np.array([5.0, 5.0]), np.array([5.0, 5.0]), train), 0.0)
    assert np.isclose(mase(np.array([0.0, 0.0]), np.array([7.0, 7.0]), train), 1.0)


def test_smape_known_values():
    assert np.isclose(smape(np.array([100.0]), np.array([100.0])), 0.0)
    assert np.isclose(smape(np.array([100.0]), np.array([50.0])), 100 * 2 * 50 / 150)
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0


def test_coverage_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(y, y - 0.5, y + 0.5) == 1.0
    assert coverage(y, np.array([2.0, 2, 3, 4]), np.array([3.0, 2, 3, 4])) == 0.75


def test_seasonal_naive_is_exact_on_pure_weekly_series():
    df = generate(2, 1, 200, seed=3)
    df["units"] = 10 + df["date"].dt.dayofweek.to_numpy() * 3.0  # perfectly weekly
    cfg = BacktestConfig(n_origins=2, n_series=2, models=("SeasonalNaive",), ensemble=False)
    res = backtest(df, cfg)
    # In-sample seasonal-naive error is 0, so skip the MASE division; check raw errors.
    fc = res.forecasts
    assert np.allclose(fc["yhat"], fc["y"])


@pytest.fixture(scope="module")
def bt(panel):
    cfg = BacktestConfig(
        horizon=14, n_origins=2, step=28, n_series=2,
        models=("SeasonalNaive", "AutoETS", "AutoARIMA_x", "XGBoost"),
    )
    return backtest(panel, cfg)


def test_backtest_outputs(bt):
    expected = {"SeasonalNaive", "AutoETS", "AutoARIMA_x", "XGBoost", "Ensemble"}
    assert set(bt.summary["model"]) == expected
    assert (bt.summary[["mase", "smape", "width"]] > 0).all().all()
    assert bt.summary["coverage"].between(0, 1).all()
    assert (bt.forecasts.groupby("model").size() == 2 * 14 * 2).all()
    assert (bt.forecasts["date"] > bt.forecasts["origin"]).all()


def test_models_that_use_promo_beat_seasonal_naive(bt):
    s = bt.summary.set_index("model")["mase"]
    assert s["XGBoost"] < s["SeasonalNaive"]
    assert s["AutoARIMA_x"] < s["SeasonalNaive"]


def test_prophet_smoke(panel):
    pytest.importorskip("prophet")
    cfg = BacktestConfig(n_origins=1, n_series=1, models=("Prophet",), ensemble=False)
    res = backtest(panel, cfg)
    assert res.summary.loc[0, "model"] == "Prophet"
    assert np.isfinite(res.summary.loc[0, "mase"])
