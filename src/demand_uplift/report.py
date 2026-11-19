"""Markdown report and matplotlib figures for backtests and causal analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .causal import CausalResults
from .forecast import BacktestResult


def _calib_view(c: pd.DataFrame) -> pd.DataFrame:
    out = c.assign(decile=c["q"].astype(int) + 1)
    return out[["decile", "pred", "dr", "dr_se", "true", "n"]]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    kinds = {c: df[c].dtype.kind for c in cols}
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row, strict=True):
            if kinds[c] == "b":
                cells.append("yes" if v else "no")
            elif kinds[c] in "iu":
                cells.append(str(int(v)))
            elif kinds[c] == "f":
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def fig_overview(df: pd.DataFrame, path: Path) -> None:
    sid = min(df["series_id"].unique())
    s = df[df["series_id"] == sid].iloc[:365]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(s["date"], s["units"], lw=0.9, label="observed units")
    ax.plot(s["date"], s["true_baseline"], lw=1.2, color="black", label="true no-promo baseline")
    p = s[s["promo"] == 1]
    ax.scatter(p["date"], p["units"], s=10, color="tab:red", label="promo day", zorder=3)
    ax.set_title(f"Synthetic series {sid}: promos land on high-baseline days (confounding)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_backtest(res: BacktestResult, path_metrics: Path, path_fc: Path) -> None:
    s = res.summary
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].barh(s["model"], s["mase"], color="tab:blue")
    axes[0].axvline(1.0, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel("MASE (lower is better)")
    axes[0].invert_yaxis()
    axes[1].barh(s["model"], s["coverage"], color="tab:green")
    axes[1].axvline(1 - 0.2, color="black", ls="--", lw=0.8, label="nominal 80%")
    axes[1].set_xlabel("empirical 80% interval coverage")
    axes[1].set_xlim(0, 1.05)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_metrics, dpi=130)
    plt.close(fig)

    fc = res.forecasts
    sid = min(fc["series_id"].unique())
    origin = fc["origin"].max()
    d = fc[(fc["series_id"] == sid) & (fc["origin"] == origin)]
    best = [m for m in s["model"] if m != "SeasonalNaive"][:2] + ["SeasonalNaive"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    first = d[d["model"] == d["model"].iloc[0]].sort_values("date")
    ax.plot(first["date"], first["y"], color="black", lw=2, label="actual")
    for m in best:
        g = d[d["model"] == m].sort_values("date")
        if g.empty:
            continue
        (line,) = ax.plot(g["date"], g["yhat"], lw=1.3, label=m)
        if m != "SeasonalNaive":
            ax.fill_between(g["date"], g["lo"], g["hi"], alpha=0.15, color=line.get_color())
    ax.set_title(f"Held-out 14-day forecast, series {sid} (80% intervals)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_fc, dpi=130)
    plt.close(fig)


def fig_causal(res: CausalResults, path_est: Path, path_cate: Path) -> None:
    t = res.estimates.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ypos = np.arange(len(t))
    ax.errorbar(
        t["estimate"], ypos, xerr=[t["estimate"] - t["ci_low"], t["ci_high"] - t["estimate"]],
        fmt="o", color="tab:blue", capsize=3,
    )
    ax.axvline(res.truth["ate"], color="red", ls="--", label=f"true ATE = {res.truth['ate']:.2f}")
    ax.axvline(res.truth["plr_weighted"], color="orange", ls=":",
               label=f"true PLR-weighted effect = {res.truth['plr_weighted']:.2f}")
    ax.set_yticks(ypos)
    ax.set_yticklabels(t["method"])
    ax.set_xlabel("incremental units per promo day (95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_est, dpi=130)
    plt.close(fig)

    best = res.best_learner
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sub = res.cate.sample(n=min(4000, len(res.cate)), random_state=0)
    axes[0].scatter(sub["true_tau"], sub[best], s=4, alpha=0.3)
    lim = [sub["true_tau"].min(), sub["true_tau"].max()]
    axes[0].plot(lim, lim, color="red", lw=1)
    axes[0].set_xlabel("true CATE")
    axes[0].set_ylabel(f"{best} CATE (out-of-sample)")
    axes[0].set_title("Recovery of heterogeneous effects")
    c = res.calibration
    axes[1].errorbar(c["pred"], c["dr"], yerr=1.96 * c["dr_se"], fmt="o", capsize=2,
                     label="DR estimate per decile")
    axes[1].plot(c["pred"], c["true"], "s", color="black", ms=4, label="true mean CATE")
    axes[1].plot(lim, lim, color="red", lw=1)
    axes[1].set_xlabel("mean predicted CATE (decile)")
    axes[1].set_ylabel("observed effect")
    axes[1].set_title("Calibration by predicted-uplift decile")
    axes[1].legend(fontsize=8)
    u = res.uplift
    axes[2].plot(u["q"], u["model"], label=f"{u['learner']} targeting")
    axes[2].plot(u["q"], u["oracle"], label="oracle (true CATE)", ls="--")
    axes[2].plot(u["q"], u["random"], label="random targeting", ls=":", color="gray")
    axes[2].set_xlabel("fraction of promo-days targeted")
    axes[2].set_ylabel("cumulative incremental units / n")
    axes[2].set_title("Uplift curve (DR scores)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_cate, dpi=130)
    plt.close(fig)


ML_METHODS = ("LinearDML", "LinearDML-het", "DRLearner", "CausalForestDML", "AIPW")
MODE_LABEL = {"fixed": "fixed budget (<=100 rounds)", "es": "early stopping (<=1000 rounds)"}


def fig_coverage(summary: pd.DataFrame, raw: pd.DataFrame, path: Path) -> None:
    """Left: empirical 95% CI coverage of the true ATE (Wilson 95% band) per estimator and
    nuisance setting. Right: per-seed relative error of the ML estimators."""
    methods = list(dict.fromkeys(summary["method"]))[::-1]
    modes = list(dict.fromkeys(summary["nuisance"]))
    off = np.linspace(-0.18, 0.18, len(modes)) if len(modes) > 1 else [0.0]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for k, mode in enumerate(modes):
        s = summary[summary["nuisance"] == mode].set_index("method").reindex(methods)
        y = np.arange(len(methods)) + off[k]
        ax.errorbar(s["coverage_ate"], y,
                    xerr=[(s["coverage_ate"] - s["coverage_ate_lo"]).clip(lower=0),
                          (s["coverage_ate_hi"] - s["coverage_ate"]).clip(lower=0)],
                    fmt="o", capsize=3, color=colors[k % 3], label=MODE_LABEL.get(mode, mode))
    ax.axvline(0.95, color="black", ls="--", lw=0.8, label="nominal 95%")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlim(-0.03, 1.03)
    n = int(summary["n_seeds"].max())
    ax.set_xlabel(f"share of {n} seeds whose 95% CI covers the true ATE")
    ax.set_title("CI coverage (bars: Wilson 95% interval)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax = axes[1]
    ml = [m for m in methods if m in ML_METHODS]
    for k, mode in enumerate(modes):
        g = raw[raw["nuisance"] == mode]
        for i, m in enumerate(ml):
            r = g[g["method"] == m]
            rel = 100 * (r["estimate"] - r["true_ate"]) / r["true_ate"]
            jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(rel))
            ax.scatter(rel, i + off[k] + jitter, s=7, alpha=0.5, color=colors[k % 3])
            ax.plot([rel.mean()] * 2, [i + off[k] - 0.12, i + off[k] + 0.12], color="black", lw=2)
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(ml)))
    ax.set_yticklabels(ml)
    ax.set_xlabel("estimate vs true ATE (%), one dot per seed; bar = mean")
    ax.set_title("Bias across data seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def write_report(
    out_dir: Path,
    meta: dict,
    bt: BacktestResult | None,
    ca: CausalResults | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["# Demand forecasting and causal uplift report", ""]
    L.append(
        f"Synthetic data, seed {meta['seed']}: {meta['n_stores']} stores x {meta['n_products']} "
        f"products x {meta['n_days']} days = {meta['n_rows']:,} rows; promo rate "
        f"{meta['promo_rate']:.1%}."
    )
    L.append("")
    if bt is not None:
        c = bt.config
        L += [
            "## Forecast backtest", "",
            (f"Rolling origin: {c.n_origins} origins, step {c.step} days, horizon {c.horizon} days, "
            f"{c.n_series} series. MASE scaled by in-sample seasonal-naive (m=7) MAE; "
            "interval nominal level 80%."), "",
            md_table(bt.summary.rename(columns={
                "mase": "MASE", "smape": "sMAPE (%)", "coverage": "80% coverage",
                "width": "mean width"})), "",
            "![metrics](backtest_metrics.png)", "", "![forecast](backtest_forecast.png)", "",
        ]
    if ca is not None:
        t = ca.truth
        L += [
            "## Causal uplift", "",
            (f"Ground truth: ATE = **{t['ate']:.3f}** incremental units per promo day "
            f"(ATT = {t['att']:.3f}; Var(D|X)-weighted partially-linear estimand = "
            f"{t['plr_weighted']:.3f})."), "",
        ]
        e = ca.estimates[["method", "estimate", "se", "ci_low", "ci_high", "bias", "covers_ate"]]
        L += [md_table(e), "", "![estimates](causal_estimates.png)", ""]
        L += [
            "### CATE learners (out-of-sample, 2-fold cross-fitting)", "",
            md_table(ca.cate_metrics), "",
            (f"Best learner by correlation with true CATE: **{ca.best_learner}**. "
            "`calib_slope` is the regression slope of doubly-robust scores on predicted CATE "
            "(1 = calibrated); `uplift_auc_gain` is the mean gap between the targeting curve and "
            "random targeting, as a multiple of the ATE."), "",
            md_table(_calib_view(ca.calibration)), "", "![cate](cate_diagnostics.png)", "",
        ]
        if ca.dowhy:
            L += ["### DoWhy identification and refutation", ""]
            if "error" in ca.dowhy:
                L += [f"DoWhy step failed: `{ca.dowhy['error']}`", ""]
            else:
                d = ca.dowhy
                L.append(
                    f"Graph: {d['graph_nodes']} nodes; every observed covariate "
                    f"({', '.join(d['backdoor_set'])}) is a common cause of promo and demand. "
                    f"On a {d['n']:,}-row subsample the identified backdoor estimand gives "
                    f"{d['estimate_stratification']:.3f} (propensity-score stratification)"
                    + (f" and {d['estimate_econml_dml']:.3f} (EconML LinearDML via DoWhy)."
                       if "estimate_econml_dml" in d else ".")
                )
                L.append("")
                rr = []
                for k, v in ca.dowhy["refutations"].items():
                    rr.append({"refuter": k, "new_effect": v.get("new_effect", float("nan")),
                               "p_value": v.get("p_value") if v.get("p_value") is not None
                               else float("nan"), "error": v.get("error", "")})
                L += [md_table(pd.DataFrame(rr)), ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")

    payload = {"meta": meta}
    if bt is not None:
        payload["backtest_summary"] = bt.summary.to_dict(orient="records")
        payload["xgb_early_stopping"] = bt.config.xgb_early_stopping
        payload["xgb_trees"] = bt.xgb_trees
    if ca is not None:
        payload["truth"] = ca.truth
        payload["estimates"] = ca.estimates.to_dict(orient="records")
        payload["cate_metrics"] = ca.cate_metrics.to_dict(orient="records")
        payload["dowhy"] = ca.dowhy
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    return path
