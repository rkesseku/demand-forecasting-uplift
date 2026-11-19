"""Causal uplift estimation for promotions.

Estimators
----------
* ``naive``         - difference in mean units, promo vs. no promo (biased under confounding).
* ``regression``    - OLS adjustment for covariates with HC3 standard errors.
* ``LinearDML``     - EconML double/debiased ML with a constant effect (partially-linear model;
                      estimand is the Var(D|X)-weighted average effect).
* ``LinearDML-het`` - EconML LinearDML with a linear CATE in X; ATE = sample average of the CATE.
* ``CausalForestDML`` - EconML causal forest (nonparametric CATE, honest inference for the ATE).
* ``DRLearner``     - EconML doubly-robust (AIPW) learner with a linear final stage.
* ``AIPW``          - cross-fitted augmented IPW with influence-function CI (own numpy/sklearn
                      implementation, used as an independent cross-check and for DR scores).

CATE learners (EconML ``TLearner``, ``XLearner``, ``CausalForestDML``) are evaluated out of
sample against the known truth, and against doubly-robust scores for a calibration table,
a best-linear-predictor calibration slope and an uplift (Qini-style) curve.

DoWhy is used for the explicit causal graph, backdoor identification, an estimate through the
identified estimand, and refutation tests.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .data import true_effects

Z = stats.norm.ppf(0.975)


def covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Observed pre-treatment covariates (confounders and effect modifiers). No ``true_*``."""
    d = df["date"]
    ang = 2 * np.pi * d.dt.dayofyear / 365.25
    X = pd.concat(
        [
            pd.get_dummies(df["store"], prefix="store", drop_first=True, dtype=float),
            pd.get_dummies(df["product"], prefix="product", drop_first=True, dtype=float),
            pd.get_dummies(d.dt.dayofweek, prefix="dow", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    X["sin1"], X["cos1"] = np.sin(ang), np.cos(ang)
    X["sin2"], X["cos2"] = np.sin(2 * ang), np.cos(2 * ang)
    X["holiday"] = df["holiday"].to_numpy(dtype=float)
    X["log_price"] = np.log(df["price"].to_numpy())
    X["t"] = (d - d.min()).dt.days.to_numpy() / 365.0
    return X.reset_index(drop=True)


@dataclass
class Estimate:
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    note: str = ""


def _est(method: str, est: float, se: float, note: str = "") -> Estimate:
    return Estimate(method, float(est), float(se), float(est - Z * se), float(est + Z * se), note)


def _trees(n: int) -> int:
    """EconML forests need n_estimators divisible by the sub-forest size (4)."""
    return max(4, (int(n) // 4) * 4)


# Nuisance budgets. "fixed" is the original setting: at most ``iters`` boosting rounds (sklearn's
# default early_stopping="auto" still stops earlier on fits with > 10,000 rows). "es" raises the
# cap to ES_MAX_ITER and always early-stops on a 10% validation split of the *training* fold, i.e.
# the number of rounds is chosen by held-out nuisance loss, never by the (unknown) treatment effect.
NUISANCE_MODES = ("fixed", "es")
ES_MAX_ITER = 1000


def _hgb_kw(iters: int, nuisance: str) -> dict:
    kw = {"learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 30}
    if nuisance == "fixed":
        return {**kw, "max_iter": iters}
    if nuisance == "es":
        return {
            **kw, "max_iter": ES_MAX_ITER, "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        }
    raise ValueError(f"nuisance must be one of {NUISANCE_MODES}, got {nuisance!r}")


def _reg(seed: int, iters: int, nuisance: str = "fixed") -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **_hgb_kw(iters, nuisance))


def _clf(seed: int, iters: int, nuisance: str = "fixed") -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=seed, **_hgb_kw(iters, nuisance))


# ------------------------------------------------------------------ simple estimators
def naive_difference(y: np.ndarray, d: np.ndarray) -> Estimate:
    y1, y0 = y[d == 1], y[d == 0]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return _est("naive", y1.mean() - y0.mean(), se, "promo vs non-promo mean units")


def regression_adjustment(y: np.ndarray, d: np.ndarray, X: pd.DataFrame) -> Estimate:
    design = sm.add_constant(np.column_stack([d, X.to_numpy()]))
    fit = sm.OLS(y, design).fit(cov_type="HC3")
    return _est("regression", fit.params[1], fit.bse[1], "OLS + covariates, HC3 SE")


# -------------------------------------------------------------------- AIPW (own code)
def cross_fit_aipw(
    y: np.ndarray, d: np.ndarray, X: np.ndarray, n_folds: int = 5, seed: int = 0,
    iters: int = 100, clip: float = 0.02, nuisance: str = "fixed",
) -> tuple[Estimate, np.ndarray]:
    """Cross-fitted AIPW. Returns the ATE estimate and per-row influence/DR scores psi_i."""
    n = len(y)
    mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        t1, t0 = tr[d[tr] == 1], tr[d[tr] == 0]
        mu1[te] = _reg(seed, iters, nuisance).fit(X[t1], y[t1]).predict(X[te])
        mu0[te] = _reg(seed, iters, nuisance).fit(X[t0], y[t0]).predict(X[te])
        e[te] = _clf(seed, iters, nuisance).fit(X[tr], d[tr]).predict_proba(X[te])[:, 1]
    e = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + d * (y - mu1) / e - (1 - d) * (y - mu0) / (1 - e)
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    return _est("AIPW", ate, se, "cross-fitted, influence-function CI"), psi


# ------------------------------------------------------------------ EconML estimators
def _econml_ate(est, X, y, d, name: str, note: str, **fit_kw) -> Estimate:
    est.fit(y, d, X=X, **fit_kw)
    inf = est.effect_inference(X).population_summary()
    return _est(name, float(np.squeeze(inf.mean_point)), float(np.squeeze(inf.stderr_mean)), note)


def econml_estimates(
    y: np.ndarray, d: np.ndarray, X: pd.DataFrame, n_folds: int, seed: int, iters: int,
    forest_trees: int, nuisance: str = "fixed", n_jobs: int = 2,
) -> list[Estimate]:
    from econml.dml import CausalForestDML, LinearDML
    from econml.dr import LinearDRLearner

    Xn = X.to_numpy()
    out = []
    # Constant-effect partially-linear DML: confounders enter as W only.
    dml = LinearDML(
        model_y=_reg(seed, iters, nuisance), model_t=_clf(seed, iters, nuisance), discrete_treatment=True,
        cv=n_folds, random_state=seed,
    )
    dml.fit(y, d, X=None, W=Xn)
    inf = dml.effect_inference().population_summary()
    out.append(
        _est("LinearDML", float(np.squeeze(inf.mean_point)), float(np.squeeze(inf.stderr_mean)),
             "constant effect; targets Var(D|X)-weighted ATE")
    )
    dml_h = LinearDML(
        model_y=_reg(seed, iters, nuisance), model_t=_clf(seed, iters, nuisance), discrete_treatment=True,
        cv=n_folds, random_state=seed,
    )
    out.append(_econml_ate(dml_h, Xn, y, d, "LinearDML-het", "linear CATE in X, averaged"))
    dr = LinearDRLearner(
        model_propensity=_clf(seed, iters, nuisance), model_regression=_reg(seed, iters, nuisance), cv=n_folds,
        random_state=seed,
    )
    out.append(_econml_ate(dr, Xn, y, d, "DRLearner", "doubly robust, linear final stage"))
    cf = CausalForestDML(
        model_y=_reg(seed, iters, nuisance), model_t=_clf(seed, iters, nuisance), discrete_treatment=True,
        n_estimators=_trees(forest_trees), min_samples_leaf=40, max_samples=0.2, cv=n_folds,
        random_state=seed, n_jobs=n_jobs,
    )
    out.append(_econml_ate(cf, Xn, y, d, "CausalForestDML", "honest causal forest, averaged"))
    return out


def _cate_factories(seed: int, iters: int, forest_trees: int) -> dict:
    from econml.dml import CausalForestDML
    from econml.metalearners import TLearner, XLearner

    return {
        "T-learner": lambda: TLearner(models=_reg(seed, iters)),
        "X-learner": lambda: XLearner(
            models=_reg(seed, iters), propensity_model=_clf(seed, iters)
        ),
        "CausalForestDML": lambda: CausalForestDML(
            model_y=_reg(seed, iters), model_t=_clf(seed, iters), discrete_treatment=True,
            n_estimators=_trees(forest_trees), min_samples_leaf=40, max_samples=0.2, cv=3,
            random_state=seed, n_jobs=2,
        ),
    }


def cross_fit_cate(factory, y, d, X: np.ndarray, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Out-of-sample CATE: fit on training folds, predict the held-out fold."""
    tau = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        est = factory()
        est.fit(y[tr], d[tr], X=X[tr])
        tau[te] = np.squeeze(est.effect(X[te]))
    return tau


# ------------------------------------------------------------ CATE evaluation helpers
def calibration_table(tau_hat: np.ndarray, psi: np.ndarray, tau_true: np.ndarray, bins: int = 10):
    """Decile calibration: mean predicted CATE vs. DR-estimated and true mean effect per decile."""
    q = pd.qcut(pd.Series(tau_hat).rank(method="first"), bins, labels=False)
    tab = pd.DataFrame({"q": q, "pred": tau_hat, "dr": psi, "true": tau_true}).groupby("q")
    out = tab.agg(pred=("pred", "mean"), dr=("dr", "mean"), true=("true", "mean"), n=("pred", "size"))
    out["dr_se"] = tab["dr"].std(ddof=1) / np.sqrt(out["n"])
    return out.reset_index()


def calibration_slope(tau_hat: np.ndarray, psi: np.ndarray) -> tuple[float, float]:
    """Best-linear-predictor slope of DR scores on predicted CATE (1 = well calibrated)."""
    fit = sm.OLS(psi, sm.add_constant(tau_hat)).fit(cov_type="HC3")
    return float(fit.params[1]), float(fit.bse[1])


def uplift_curve(score: np.ndarray, gain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative per-unit incremental gain when targeting the top-q fraction by ``score``.

    Returns (fraction targeted, cumulative gain / n). The random-targeting curve is q * mean(gain).
    """
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(gain[order]) / len(gain)
    return np.arange(1, len(gain) + 1) / len(gain), cum


def uplift_auc_gain(score: np.ndarray, gain: np.ndarray) -> float:
    """Mean gap between the targeting curve and random targeting (normalised by the ATE)."""
    q, cum = uplift_curve(score, gain)
    return float(np.mean(cum - q * gain.mean()) / abs(gain.mean()))


# --------------------------------------------------------------------------- DoWhy
def dowhy_analysis(df: pd.DataFrame, n: int = 6000, seed: int = 0) -> dict:
    """Explicit graph -> backdoor identification -> estimates -> refutation, with DoWhy.

    The causal graph has eight observed pre-treatment nodes (store, product, day of week, two
    yearly Fourier terms, holiday, log price, time trend), each a common cause of promo and
    demand, plus the edge promo -> units. Estimates go through the identified backdoor estimand
    (propensity-score stratification, and DML via DoWhy's EconML bridge). Refuters run on the
    stratification estimate. Runs on a random subsample of ``n`` rows to keep refuters fast.
    """
    import networkx as nx

    if not hasattr(nx.algorithms, "d_separated"):  # renamed in newer networkx
        nx.algorithms.d_separated = nx.algorithms.is_d_separator
    from dowhy import CausalModel

    ang = 2 * np.pi * df["date"].dt.dayofyear / 365.25
    data = pd.DataFrame(
        {
            "store": df["store"].astype("category"),
            "product": df["product"].astype("category"),
            "dow": df["date"].dt.dayofweek.astype("category"),
            "sin1": np.sin(ang),
            "cos1": np.cos(ang),
            "holiday": df["holiday"].astype(float),
            "log_price": np.log(df["price"]),
            "t": (df["date"] - df["date"].min()).dt.days / 365.0,
            "promo": df["promo"].astype(int),
            "units": df["units"],
        }
    )
    data = data.sample(n=min(n, len(data)), random_state=seed).reset_index(drop=True)
    nodes = ["store", "product", "dow", "sin1", "cos1", "holiday", "log_price", "t"]
    g = nx.DiGraph()
    for c in nodes:  # every observed covariate is a common cause of promo and demand
        g.add_edge(c, "promo")
        g.add_edge(c, "units")
    g.add_edge("promo", "units")
    model = CausalModel(data=data, treatment="promo", outcome="units", graph=g)
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    est = model.estimate_effect(estimand, method_name="backdoor.propensity_score_stratification")
    res = {
        "n": len(data),
        "graph_nodes": len(nodes) + 2,
        "backdoor_set": list(estimand.get_backdoor_variables()),
        "estimate_stratification": float(est.value),
        "refutations": {},
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dml = model.estimate_effect(
                estimand,
                method_name="backdoor.econml.dml.LinearDML",
                method_params={
                    "init_params": {
                        "model_y": _reg(seed, 60), "model_t": _clf(seed, 60),
                        "discrete_treatment": True, "cv": 3,
                    },
                    "fit_params": {},
                },
            )
        res["estimate_econml_dml"] = float(dml.value)
    except Exception as exc:  # noqa: BLE001
        res["estimate_econml_dml_error"] = f"{type(exc).__name__}: {exc}"
    for label, method, kw in [
        ("placebo_treatment", "placebo_treatment_refuter",
         {"placebo_type": "permute", "num_simulations": 20}),
        ("random_common_cause", "random_common_cause", {"num_simulations": 20}),
        ("data_subset", "data_subset_refuter", {"subset_fraction": 0.8, "num_simulations": 20}),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = model.refute_estimate(estimand, est, method_name=method, random_seed=seed, **kw)
            p = r.refutation_result.get("p_value") if isinstance(
                r.refutation_result, dict) else None
            res["refutations"][label] = {
                "new_effect": float(r.new_effect),
                "p_value": None if p is None or not np.isfinite(p) else float(p),
            }
        except Exception as exc:  # noqa: BLE001
            res["refutations"][label] = {"error": f"{type(exc).__name__}: {exc}"}
    return res


# ---------------------------------------------------------------- ATE estimates only
def ate_estimates(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 0, iters: int = 100,
    forest_trees: int = 100, nuisance: str = "fixed", n_jobs: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """All ATE estimators on one panel. Returns the estimate table and the AIPW DR scores."""
    y = df["units"].to_numpy(dtype=float)
    d = df["promo"].to_numpy(dtype=int)
    X = covariates(df)
    truth = true_effects(df)
    ests = [naive_difference(y, d), regression_adjustment(y, d, X)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ests += econml_estimates(y, d, X, n_folds, seed, iters, forest_trees, nuisance, n_jobs)
        aipw, psi = cross_fit_aipw(
            y, d, X.to_numpy(), n_folds=n_folds, seed=seed, iters=iters, nuisance=nuisance
        )
    ests.append(aipw)
    tab = pd.DataFrame([e.__dict__ for e in ests])
    tab["bias"] = tab["estimate"] - truth["ate"]
    tab["covers_ate"] = (tab["ci_low"] <= truth["ate"]) & (truth["ate"] <= tab["ci_high"])
    return tab, psi


# ---------------------------------------------------------------- coverage study
def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z**2 / n
    mid = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    lo = 0.0 if k == 0 else max(0.0, mid - half)
    hi = 1.0 if k == n else min(1.0, mid + half)
    return float(lo), float(hi)


def coverage_study(
    seeds, n_stores: int = 8, n_products: int = 5, n_days: int = 1095, n_folds: int = 5,
    iters: int = 100, forest_trees: int = 100, nuisance: str = "fixed", n_jobs: int = 2,
    model_seed: int = 0, callback=None,
) -> pd.DataFrame:
    """Re-generate the synthetic panel for every data seed and run every ATE estimator.

    Returns one row per (seed, method) with the estimate, CI, the seed's true ATE and the
    ``Var(D|X)``-weighted estimand. Nuisance hyperparameters never look at the truth.
    ``callback(raw_so_far)`` is called after each seed (used for checkpointing).
    """
    import time

    from .data import generate

    rows = []
    for s in seeds:
        t0 = time.perf_counter()
        df = generate(n_stores, n_products, n_days, seed=int(s))
        truth = true_effects(df)
        tab, _ = ate_estimates(df, n_folds=n_folds, seed=model_seed, iters=iters,
                               forest_trees=forest_trees, nuisance=nuisance, n_jobs=n_jobs)
        secs = time.perf_counter() - t0
        for r in tab.itertuples(index=False):
            rows.append({
                "seed": int(s), "nuisance": nuisance, "method": r.method,
                "estimate": r.estimate, "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "true_ate": truth["ate"], "true_plr": truth["plr_weighted"], "seconds": secs,
            })
        if callback is not None:
            callback(pd.DataFrame(rows))
    return pd.DataFrame(rows)


def summarize_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (nuisance, method): bias, RMSE, empirical 95% CI coverage (+ Wilson CI), CI width.

    ``coverage_ate`` is coverage of each seed's true ATE. ``coverage_plr`` is coverage of the
    ``Var(D|X)``-weighted effect, the estimand constant-effect LinearDML actually targets.
    ``sd_err / mean_se`` near 1 means the standard error matches the real spread of the error.
    """
    out = []
    for (nuis, method), g in raw.groupby(["nuisance", "method"], sort=False):
        err = g["estimate"] - g["true_ate"]
        cov = (g["ci_low"] <= g["true_ate"]) & (g["true_ate"] <= g["ci_high"])
        cov_p = (g["ci_low"] <= g["true_plr"]) & (g["true_plr"] <= g["ci_high"])
        n, k = len(g), int(cov.sum())
        lo, hi = wilson_interval(k, n)
        lo_p, hi_p = wilson_interval(int(cov_p.sum()), n)
        out.append({
            "nuisance": nuis, "method": method, "n_seeds": n,
            "mean_bias": float(err.mean()), "rel_bias": float((err / g["true_ate"]).mean()),
            "rmse": float(np.sqrt(np.mean(err**2))), "sd_err": float(err.std(ddof=1)),
            "mean_se": float(g["se"].mean()),
            "se_ratio": float(err.std(ddof=1) / g["se"].mean()),
            "coverage_ate": k / n, "coverage_ate_lo": lo, "coverage_ate_hi": hi,
            "coverage_plr": float(cov_p.mean()), "coverage_plr_lo": lo_p,
            "coverage_plr_hi": hi_p,
            "mean_width": float((g["ci_high"] - g["ci_low"]).mean()),
            "mean_seconds_per_seed": float(g["seconds"].mean()),
        })
    return pd.DataFrame(out)


# ------------------------------------------------------------------------ orchestrator
@dataclass
class CausalResults:
    truth: dict
    estimates: pd.DataFrame
    cate: pd.DataFrame  # out-of-sample CATE predictions per learner + true_tau
    cate_metrics: pd.DataFrame
    calibration: pd.DataFrame  # decile table for the best learner
    best_learner: str = ""
    uplift: dict = field(default_factory=dict)
    dowhy: dict = field(default_factory=dict)


def run_causal(
    df: pd.DataFrame, n_folds: int = 5, seed: int = 0, iters: int = 100, forest_trees: int = 100,
    use_dowhy: bool = True, dowhy_n: int = 6000, nuisance: str = "fixed",
) -> CausalResults:
    y = df["units"].to_numpy(dtype=float)
    d = df["promo"].to_numpy(dtype=int)
    Xn = covariates(df).to_numpy()
    truth = true_effects(df)
    tab, psi = ate_estimates(df, n_folds=n_folds, seed=seed, iters=iters,
                             forest_trees=forest_trees, nuisance=nuisance)

    tau_true = df["true_tau"].to_numpy()
    cate = pd.DataFrame({"true_tau": tau_true})
    metrics = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, factory in _cate_factories(seed, iters, forest_trees).items():
            th = cross_fit_cate(factory, y, d, Xn, seed=seed)
            cate[name] = th
            slope, slope_se = calibration_slope(th, psi)
            metrics.append(
                {
                    "learner": name,
                    "corr_true": float(np.corrcoef(th, tau_true)[0, 1]),
                    "rmse": float(np.sqrt(np.mean((th - tau_true) ** 2))),
                    "calib_slope": slope,
                    "calib_slope_se": slope_se,
                    "uplift_auc_gain": uplift_auc_gain(th, psi),
                }
            )
    cm = pd.DataFrame(metrics)
    best = str(cm.sort_values("corr_true", ascending=False).iloc[0]["learner"])
    q, cum_best = uplift_curve(cate[best].to_numpy(), psi)
    _, cum_oracle = uplift_curve(tau_true, psi)
    uplift = {
        "q": q, "model": cum_best, "oracle": cum_oracle, "random": q * psi.mean(),
        "learner": best,
    }
    res = CausalResults(
        truth=truth, estimates=tab, cate=cate, cate_metrics=cm,
        calibration=calibration_table(cate[best].to_numpy(), psi, tau_true),
        best_learner=best, uplift=uplift,
    )
    if use_dowhy:
        try:
            res.dowhy = dowhy_analysis(df, n=dowhy_n, seed=seed)
        except Exception as exc:  # noqa: BLE001
            res.dowhy = {"error": f"{type(exc).__name__}: {exc}"}
    return res
