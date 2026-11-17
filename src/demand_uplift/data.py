"""Synthetic multi-store x multi-product daily demand with known causal ground truth.

Data-generating process (per series s = store x product, day t)::

    log mu0[s,t] = log base[s] + trend[s]*t/365 + weekly[dow] + yearly[s](doy)
                   + holiday effect + elasticity[s] * log(price[s,t] / base_price[s])
    promo[s,t]  ~ Bernoulli(sigmoid(a + b * z[s,t] + store_bias[store]))      (CONFOUNDED)
    tau[s,t]     = mu0[s,t] * lift[store, product] * weekend_multiplier       (HETEROGENEOUS)
    units[s,t]   = mu0[s,t] + promo[s,t] * tau[s,t] + noise

``z`` is the standardised expected baseline demand, so promotions are scheduled when demand is
already expected to be high and more often in "promo-happy" (and larger) stores. A naive
promo-vs-no-promo comparison is therefore biased upward. Columns prefixed ``true_`` are the
ground truth and must never be used as model features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (month, day) -> log-demand effect
HOLIDAYS = {
    (1, 1): 0.25,
    (2, 14): 0.10,
    (7, 4): 0.15,
    (11, 25): 0.30,
    (12, 24): 0.35,
    (12, 25): -0.40,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_stores: int = 8,
    n_products: int = 5,
    n_days: int = 1095,
    seed: int = 42,
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """Generate a long-format panel (one row per store x product x day), reproducible by seed."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days)
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    hol_effect = np.array([HOLIDAYS.get((d.month, d.day), 0.0) for d in dates])
    is_hol = (hol_effect != 0).astype(int)

    # Store-level attributes. Bigger stores are also more promo-happy (store-level confounding).
    store_scale = rng.normal(0.0, 0.30, n_stores)
    store_bias = 0.45 * store_scale / 0.30 + rng.normal(0, 0.25, n_stores)
    store_lift = rng.normal(0.0, 0.06, n_stores)
    # Product-level attributes.
    prod_base = rng.normal(np.log(60), 0.5, n_products)
    prod_price = rng.uniform(3.0, 20.0, n_products)
    prod_elast = rng.uniform(-2.0, -1.0, n_products)
    prod_lift = rng.normal(0.0, 0.07, n_products)
    prod_amp = rng.uniform(0.10, 0.35, n_products)
    prod_phase = rng.uniform(0, 2 * np.pi, n_products)
    weekly_pattern = np.array([-0.05, -0.02, 0.0, 0.03, 0.10, 0.22, 0.12])  # Mon..Sun

    frames = []
    for i in range(n_stores):
        for j in range(n_products):
            log_base = prod_base[j] + store_scale[i] + rng.normal(0, 0.1)
            trend = rng.normal(0.08, 0.05)
            weekly = weekly_pattern[dow] * (0.6 + 0.8 * rng.random())
            yearly = prod_amp[j] * np.sin(2 * np.pi * doy / 365.25 + prod_phase[j])
            # Weekly price levels (with a little daily jitter) around a base price.
            n_weeks = n_days // 7 + 2
            wk_level = rng.normal(0.0, 0.08, n_weeks)
            price = prod_price[j] * np.exp(wk_level[t // 7] + rng.normal(0, 0.01, n_days))
            log_mu0 = (
                log_base
                + trend * t / 365.0
                + weekly
                + yearly
                + hol_effect
                + prod_elast[j] * np.log(price / prod_price[j])
            )
            mu0 = np.exp(log_mu0)
            z = (log_mu0 - log_mu0.mean()) / log_mu0.std()  # standardised expected demand
            p = _sigmoid(-1.5 + 0.9 * z + store_bias[i])
            promo = (rng.random(n_days) < p).astype(int)
            lift = np.clip(0.25 + store_lift[i] + prod_lift[j], 0.05, 0.6)
            wk_mult = np.where(dow >= 5, 1.3, 1.0)
            tau = mu0 * lift * wk_mult
            noise = rng.normal(0.0, 1.0, n_days) * 0.10 * mu0
            units = np.maximum(mu0 + promo * tau + noise, 0.0)
            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "store": f"S{i + 1:02d}",
                        "product": f"P{j + 1:02d}",
                        "units": units,
                        "price": price,
                        "promo": promo,
                        "holiday": is_hol,
                        "true_baseline": mu0,
                        "true_tau": tau,
                        "true_propensity": p,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df["series_id"] = df["store"] + "_" + df["product"]
    cols = [
        "date", "series_id", "store", "product", "units", "price", "promo", "holiday",
        "true_baseline", "true_tau", "true_propensity",
    ]
    df = df[cols].sort_values(["series_id", "date"], kind="stable").reset_index(drop=True)
    return df


def true_effects(df: pd.DataFrame) -> dict:
    """Ground-truth estimands computed from the known per-row effects."""
    w = df["true_propensity"] * (1 - df["true_propensity"])
    return {
        "ate": float(df["true_tau"].mean()),
        "att": float(df.loc[df["promo"] == 1, "true_tau"].mean()),
        # The partially-linear (DML) estimand is a Var(D|X)-weighted average of the CATE.
        "plr_weighted": float((w * df["true_tau"]).sum() / w.sum()),
    }
