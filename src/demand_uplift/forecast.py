"""Forecasting models and rolling-origin backtesting.

Models (all produce a point forecast plus a nominal 80% prediction interval):

* ``SeasonalNaive``  - statsforecast seasonal naive (m=7), the baseline every model must beat.
* ``AutoETS``        - statsforecast exponential smoothing with automatic model selection (m=7).
* ``AutoARIMA_x``    - statsforecast AutoARIMA (SARIMAX-style) with promo / log-price / holiday
                       regressors plus yearly Fourier terms.
* ``Prophet``        - Prophet with promo / log-price / holiday regressors.
* ``XGBoost``        - one global XGBoost model over all series with leakage-safe
                       lag/rolling/calendar features; intervals from quantile XGBoost models
                       conformalised on a trailing calibration window (CQR).
* ``Ensemble``       - mean of the non-baseline models above (point and interval bounds).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .features import build_features, feature_columns

SEASON = 7
ALPHA = 0.2  # 80% prediction interval
ALL_MODELS = ("SeasonalNaive", "AutoETS", "AutoARIMA_x", "Prophet", "XGBoost")


# ----------------------------------------------------------------------------- metrics
def mase(y: np.ndarray, yhat: np.ndarray, y_train: np.ndarray, m: int = SEASON) -> float:
    """Mean absolute scaled error; scale is the in-sample seasonal-naive MAE."""
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))) / scale)


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Symmetric MAPE in percent (0 when both actual and forecast are 0)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(y - yhat) / np.where(denom == 0, 1, denom))
    return float(100.0 * np.mean(ratio))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Empirical fraction of actuals inside the prediction interval."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


# ------------------------------------------------------------------------- model wrappers
def _exog(df: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Regressors for ARIMA: promo, log-price, holiday, yearly Fourier terms."""
    out = pd.DataFrame(
        {
            "promo": df["promo"].to_numpy(dtype=float),
            "log_price": np.log(df["price"].to_numpy()),
            "holiday": df["holiday"].to_numpy(dtype=float),
        },
        index=df.index,
    )
    ang = 2 * np.pi * df["date"].dt.dayofyear.to_numpy() / 365.25
    for k in (1, 2):
        out[f"sin{k}"] = np.sin(k * ang)
        out[f"cos{k}"] = np.cos(k * ang)
    return out


def _sf_to_frame(fc: pd.DataFrame, col: str, series_col: str = "unique_id") -> pd.DataFrame:
    lo, hi = f"{col}-lo-80", f"{col}-hi-80"
    out = pd.DataFrame(
        {
            "series_id": fc[series_col].to_numpy(),
            "date": pd.to_datetime(fc["ds"]).to_numpy(),
            "yhat": fc[col].to_numpy(),
            "lo": fc[lo].to_numpy(),
            "hi": fc[hi].to_numpy(),
        }
    )
    return out


def forecast_statsforecast(train: pd.DataFrame, test: pd.DataFrame, models: list[str]) -> dict:
    """Seasonal naive, AutoETS (no regressors) and AutoARIMA with regressors via statsforecast."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    out: dict[str, pd.DataFrame] = {}
    base = train.rename(columns={"series_id": "unique_id", "date": "ds", "units": "y"})
    base = base[["unique_id", "ds", "y"]]
    h = int(test.groupby("series_id").size().max())

    plain = []
    if "SeasonalNaive" in models:
        plain.append(SeasonalNaive(season_length=SEASON))
    if "AutoETS" in models:
        plain.append(AutoETS(season_length=SEASON))
    if plain:
        sf = StatsForecast(models=plain, freq="D", n_jobs=1)
        fc = sf.forecast(df=base, h=h, level=[80]).reset_index()
        if "unique_id" not in fc.columns:
            fc = fc.reset_index()
        if "SeasonalNaive" in models:
            out["SeasonalNaive"] = _sf_to_frame(fc, "SeasonalNaive")
        if "AutoETS" in models:
            out["AutoETS"] = _sf_to_frame(fc, "AutoETS")

    if "AutoARIMA_x" in models:
        xtr = _exog(train, train["date"].min())
        xte = _exog(test, train["date"].min())
        df_x = pd.concat(
            [base.reset_index(drop=True), xtr.reset_index(drop=True)], axis=1
        )
        x_df = pd.concat(
            [
                test[["series_id", "date"]]
                .rename(columns={"series_id": "unique_id", "date": "ds"})
                .reset_index(drop=True),
                xte.reset_index(drop=True),
            ],
            axis=1,
        )
        arima = AutoARIMA(
            season_length=SEASON, max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1, max_D=0,
            approximation=True,
        )
        sf = StatsForecast(models=[arima], freq="D", n_jobs=1)
        fc = sf.forecast(df=df_x, h=h, X_df=x_df, level=[80]).reset_index()
        out["AutoARIMA_x"] = _sf_to_frame(fc, "AutoARIMA")
    return out


def forecast_prophet(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Prophet per series with promo / log-price / holiday regressors."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    frames = []
    for sid, tr in train.groupby("series_id", sort=False):
        te = test[test["series_id"] == sid]
        m = Prophet(
            interval_width=1 - ALPHA,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=200,
        )
        for reg in ("promo", "log_price", "holiday"):
            m.add_regressor(reg)
        d_tr = pd.DataFrame(
            {
                "ds": tr["date"].to_numpy(),
                "y": tr["units"].to_numpy(),
                "promo": tr["promo"].to_numpy(dtype=float),
                "log_price": np.log(tr["price"].to_numpy()),
                "holiday": tr["holiday"].to_numpy(dtype=float),
            }
        )
        m.fit(d_tr)
        d_te = pd.DataFrame(
            {
                "ds": te["date"].to_numpy(),
                "promo": te["promo"].to_numpy(dtype=float),
                "log_price": np.log(te["price"].to_numpy()),
                "holiday": te["holiday"].to_numpy(dtype=float),
            }
        )
        pr = m.predict(d_te)
        frames.append(
            pd.DataFrame(
                {
                    "series_id": sid,
                    "date": pr["ds"].to_numpy(),
                    "yhat": pr["yhat"].to_numpy(),
                    "lo": pr["yhat_lower"].to_numpy(),
                    "hi": pr["yhat_upper"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


XGB_TREES = 300  # fixed budget (original setting, used when early stopping is off)
XGB_MAX_TREES = 2000  # cap when early stopping is on
XGB_ES_ROUNDS = 50


def _xgb(n_estimators: int = XGB_TREES, **kw) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, tree_method="hist", n_jobs=2, random_state=0,
        **kw,
    )


def _fit_xgb(
    X: pd.DataFrame, y: np.ndarray, mask: np.ndarray, dates: pd.Series, cut: pd.Timestamp,
    val_days: int, early_stopping: bool, **kw,
) -> tuple[XGBRegressor, int]:
    """Fit on rows ``mask`` (all dated <= ``cut``). Returns (model, n_trees).

    With ``early_stopping`` the number of trees is chosen on a time-ordered validation window,
    the last ``val_days`` days up to ``cut``: a model is trained on the rows before that window
    with early stopping on it, then refit on *all* ``mask`` rows with the selected tree count.
    Only data dated <= ``cut`` is ever used, so nothing from the test window leaks in.
    """
    if not early_stopping:
        return _xgb(**kw).fit(X[mask], y[mask]), XGB_TREES
    v_cut = cut - pd.Timedelta(days=val_days)
    fit_m = mask & (dates <= v_cut).to_numpy()
    val_m = mask & (dates > v_cut).to_numpy()
    es = _xgb(XGB_MAX_TREES, early_stopping_rounds=XGB_ES_ROUNDS, **kw).fit(
        X[fit_m], y[fit_m], eval_set=[(X[val_m], y[val_m])], verbose=False)
    n_trees = int(es.best_iteration) + 1
    return _xgb(n_trees, **kw).fit(X[mask], y[mask]), n_trees


def forecast_xgb(
    feats: pd.DataFrame, origin: pd.Timestamp, horizon: int, cal_days: int = 56,
    early_stopping: bool = True, val_days: int = 28, info: dict | None = None,
) -> pd.DataFrame:
    """Global XGBoost forecast for dates in (origin, origin+horizon] for *every* series.

    ``feats`` is the full feature frame from :func:`build_features` (features are leak-free by
    construction, so building once is safe). Point model trains on rows <= origin. Quantile
    models (10%/90%) train on rows <= origin - cal_days and are conformalised (CQR) on the
    trailing ``cal_days`` window, giving finite-sample-calibrated 80% intervals in log space.
    With ``early_stopping`` each model's tree count is picked on the ``val_days`` days just
    before its own training cut-off (see :func:`_fit_xgb`); tree counts go into ``info``.
    """
    cols = feature_columns(horizon)
    need = [c for c in cols if c.startswith("rmean28") or c == f"lag_{horizon + 28}"]
    ok = feats[need].notna().all(axis=1)
    y_log = np.log1p(feats["units"].to_numpy())
    X = feats[cols]
    dates = feats["date"]
    end = origin + pd.Timedelta(days=horizon)
    test_mask = ((dates > origin) & (dates <= end)).to_numpy()
    train_mask = ((dates <= origin) & ok).to_numpy()
    cal_cut = origin - pd.Timedelta(days=cal_days)
    q_train = train_mask & (dates <= cal_cut).to_numpy()
    cal_mask = train_mask & (dates > cal_cut).to_numpy()

    fit = {"X": X, "y": y_log, "dates": dates, "val_days": val_days,
           "early_stopping": early_stopping}
    point, n_pt = _fit_xgb(mask=train_mask, cut=origin, **fit)
    qlo, n_lo = _fit_xgb(mask=q_train, cut=cal_cut, objective="reg:quantileerror",
                         quantile_alpha=ALPHA / 2, **fit)
    qhi, n_hi = _fit_xgb(mask=q_train, cut=cal_cut, objective="reg:quantileerror",
                         quantile_alpha=1 - ALPHA / 2, **fit)
    if info is not None:
        info.update(point_trees=n_pt, qlo_trees=n_lo, qhi_trees=n_hi)

    lo_c, hi_c = qlo.predict(X[cal_mask]), qhi.predict(X[cal_mask])
    yc = y_log[cal_mask]
    scores = np.maximum(lo_c - yc, yc - hi_c)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)
    qhat = float(np.quantile(scores, level))

    te = feats.loc[test_mask, ["series_id", "date"]].reset_index(drop=True)
    Xt = X[test_mask]
    te["yhat"] = np.expm1(point.predict(Xt))
    lo = qlo.predict(Xt) - qhat
    hi = qhi.predict(Xt) + qhat
    te["lo"] = np.expm1(np.minimum(lo, point.predict(Xt)))
    te["hi"] = np.expm1(np.maximum(hi, point.predict(Xt)))
    return te


# ------------------------------------------------------------------------------ backtest
@dataclass
class BacktestConfig:
    horizon: int = 14
    n_origins: int = 4
    step: int = 28
    n_series: int = 6
    models: tuple = ("SeasonalNaive", "AutoETS", "AutoARIMA_x", "Prophet", "XGBoost")
    train_window: int = 730  # days of history given to the per-series statistical models
    ensemble: bool = True
    xgb_early_stopping: bool = True  # False reproduces the original fixed 300-tree XGBoost


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame
    metrics: pd.DataFrame  # one row per model x series x origin
    summary: pd.DataFrame  # one row per model
    config: BacktestConfig = field(default_factory=BacktestConfig)
    xgb_trees: list = field(default_factory=list)  # tree counts per origin (XGBoost only)


def select_series(df: pd.DataFrame, n_series: int) -> list[str]:
    ids = sorted(df["series_id"].unique())
    if n_series >= len(ids):
        return ids
    idx = np.linspace(0, len(ids) - 1, n_series).round().astype(int)
    return [ids[i] for i in sorted(set(idx))]


def _origins(df: pd.DataFrame, cfg: BacktestConfig) -> list[pd.Timestamp]:
    last = df["date"].max()
    final = last - pd.Timedelta(days=cfg.horizon)
    return [final - pd.Timedelta(days=cfg.step * k) for k in range(cfg.n_origins - 1, -1, -1)]


def backtest(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    """Rolling-origin evaluation. XGBoost is global (trained on all series); the rest are per series.

    All models are scored on the same ``cfg.n_series`` series and the same test windows.
    """
    cfg = cfg or BacktestConfig()
    series = select_series(df, cfg.n_series)
    sub = df[df["series_id"].isin(series)]
    feats = build_features(df, cfg.horizon) if "XGBoost" in cfg.models else None
    stat_models = [m for m in cfg.models if m in ("SeasonalNaive", "AutoETS", "AutoARIMA_x")]

    rows = []
    xgb_trees: list[dict] = []
    for origin in _origins(df, cfg):
        end = origin + pd.Timedelta(days=cfg.horizon)
        start = origin - pd.Timedelta(days=cfg.train_window)
        train = sub[(sub["date"] <= origin) & (sub["date"] > start)]
        test = sub[(sub["date"] > origin) & (sub["date"] <= end)]
        preds: dict[str, pd.DataFrame] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if stat_models:
                preds.update(forecast_statsforecast(train, test, stat_models))
            if "Prophet" in cfg.models:
                preds["Prophet"] = forecast_prophet(train, test)
            if "XGBoost" in cfg.models:
                info: dict = {"origin": str(origin.date())}
                g = forecast_xgb(feats, origin, cfg.horizon,
                                 early_stopping=cfg.xgb_early_stopping, info=info)
                xgb_trees.append(info)
                preds["XGBoost"] = g[g["series_id"].isin(series)]
        if cfg.ensemble:
            members = [m for m in ("AutoETS", "AutoARIMA_x", "Prophet", "XGBoost") if m in preds]
            if len(members) >= 2:
                stack = [
                    p.set_index(["series_id", "date"]).sort_index() for p in
                    (preds[m] for m in members)
                ]
                avg = sum(stack) / len(stack)
                preds["Ensemble"] = avg.reset_index()
        for name, p in preds.items():
            m = p.merge(test[["series_id", "date", "units"]], on=["series_id", "date"])
            m["model"] = name
            m["origin"] = origin
            rows.append(m)
    fc = pd.concat(rows, ignore_index=True)
    fc = fc.rename(columns={"units": "y"})

    recs = []
    hist = sub.sort_values(["series_id", "date"])
    for (model, sid, origin), g in fc.groupby(["model", "series_id", "origin"], sort=False):
        h_train = hist[(hist["series_id"] == sid) & (hist["date"] <= origin)]["units"].to_numpy()
        recs.append(
            {
                "model": model,
                "series_id": sid,
                "origin": origin,
                "mase": mase(g["y"], g["yhat"], h_train[-365:]),
                "smape": smape(g["y"], g["yhat"]),
                "coverage": coverage(g["y"], g["lo"], g["hi"]),
                "width": float(np.mean(g["hi"] - g["lo"])),
            }
        )
    metrics = pd.DataFrame(recs)
    summary = (
        metrics.groupby("model")[["mase", "smape", "coverage", "width"]]
        .mean()
        .sort_values("mase")
        .reset_index()
    )
    return BacktestResult(forecasts=fc, metrics=metrics, summary=summary, config=cfg,
                          xgb_trees=xgb_trees)
