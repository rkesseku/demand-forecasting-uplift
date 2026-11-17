"""Leakage-safe feature engineering for the global gradient-boosting forecaster.

For a forecast horizon ``h``, every target-derived feature (lags, rolling stats) is shifted by at
least ``h`` days. A row dated ``t`` therefore only uses ``y`` up to ``t - h``, so a model trained
on data up to forecast origin ``T`` can score any date in ``(T, T + h]`` with a single direct
prediction and no recursion. Future-known covariates (calendar, holiday, price, planned promo)
are used as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "units"
KNOWN_FUTURE = ["promo", "log_price", "holiday"]
CALENDAR = ["dow", "month", "sin1", "cos1", "sin2", "cos2", "t"]
CATEGORICAL_CODES = ["store_code", "product_code"]


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    d = df["date"]
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    ang = 2 * np.pi * d.dt.dayofyear / 365.25
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(k * ang)
        out[f"cos{k}"] = np.cos(k * ang)
    out["t"] = (d - d.min()).dt.days
    return out


def lag_columns(horizon: int) -> list[str]:
    lags = [f"lag_{lag}" for lag in (horizon, horizon + 7, horizon + 14, horizon + 28)]
    return lags + ["lag_364", f"rmean7_h{horizon}", f"rmean28_h{horizon}", f"rstd28_h{horizon}"]


def feature_columns(horizon: int) -> list[str]:
    return CATEGORICAL_CODES + CALENDAR + KNOWN_FUTURE + lag_columns(horizon)


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return ``df`` (sorted by series, date) with model features appended."""
    if horizon > 364:
        raise ValueError("horizon must be <= 364 so lag_364 is leak-free")
    df = df.sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
    out = pd.concat([df, calendar_features(df)], axis=1)
    out["log_price"] = np.log(out["price"])
    out["store_code"] = pd.factorize(out["store"], sort=True)[0]
    out["product_code"] = pd.factorize(out["product"], sort=True)[0]
    g = out.groupby("series_id", sort=False)[TARGET]
    for lag in (horizon, horizon + 7, horizon + 14, horizon + 28, 364):
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    by_series = shifted.groupby(out["series_id"], sort=False)
    out[f"rmean7_h{horizon}"] = by_series.transform(lambda s: s.rolling(7, min_periods=7).mean())
    out[f"rmean28_h{horizon}"] = by_series.transform(lambda s: s.rolling(28, min_periods=28).mean())
    out[f"rstd28_h{horizon}"] = by_series.transform(lambda s: s.rolling(28, min_periods=28).std())
    return out
