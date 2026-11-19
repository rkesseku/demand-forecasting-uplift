"""Builds the architecture and confounding diagrams for brief.qmd.
Other figures: backtest_metrics and cate_diagnostics are copies of ../*.png
produced by `python -m demand_uplift report --seed 42`; coverage_study is a copy of
../results/coverage_study.png from `python -m demand_uplift coverage`."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ACC = "#1f4e9e"; LIGHT = "#eef3fb"; GREY = "#555555"; RED = "#b03a2e"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=ACC, fs=9, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)


def arrow(ax, p, q, color=GREY, style="-|>", lw=1.4, ls="-"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


# ---------- architecture ----------
fig, ax = plt.subplots(figsize=(10, 4.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
box(ax, 0.2, 2.0, 2.0, 1.0, "data.py\nSynthetic generator\n(confounded promos,\nknown true effect)", fc=ACC, tc="white", bold=True, fs=8.5)
box(ax, 2.7, 3.45, 2.2, 1.1, "features.py\nLeakage-safe lags\n(every lag >= horizon)", fs=8.5)
box(ax, 5.3, 3.45, 2.6, 1.1, "forecast.py\nSeasonalNaive, AutoETS,\nAutoARIMA, Prophet,\nXGBoost+CQR, Ensemble", fs=7.5)
box(ax, 8.2, 3.45, 1.6, 1.1, "Rolling-origin\nbacktest\nMASE, sMAPE,\ncoverage", fs=8)
box(ax, 2.7, 0.4, 2.2, 1.5, "causal.py\nNaive, regression, DML,\nDR learner, causal forest,\nown AIPW, T-/X-learner", fs=8.5)
box(ax, 5.4, 0.4, 2.4, 1.5, "Checks against truth\nCATE calibration,\nuplift curve,\nDoWhy graph + 3 refuters", fs=8.5)
box(ax, 8.2, 0.4, 1.6, 1.5, "report.py\nreport.md,\nresults.json,\nfigures", fc="#dbe6f7", bold=True, fs=8.5)
arrow(ax, (2.1, 2.7), (2.7, 3.9)); arrow(ax, (4.9, 4.0), (5.3, 4.0)); arrow(ax, (7.9, 4.0), (8.2, 4.0))
arrow(ax, (2.1, 2.3), (2.7, 1.3)); arrow(ax, (4.9, 1.15), (5.4, 1.15)); arrow(ax, (7.8, 1.15), (8.2, 1.15))
arrow(ax, (9.0, 3.45), (9.0, 1.9))
ax.text(0.2, 4.5, "Question 1:\nhow much will\nwe sell?", color=ACC, fontsize=9.5, fontweight="bold", va="top")
ax.text(0.2, 3.5, "(forecasting, top row)", color=GREY, fontsize=8.5, va="top")
ax.text(0.2, 1.55, "Question 2:\nwhat did the\npromo cause?\n(causal, bottom row)", color=ACC, fontsize=9, fontweight="bold", va="top")
ax.text(5.0, 0.05, "CLI: python -m demand_uplift {generate, backtest, causal, report}", ha="center", fontsize=8.5, color=GREY, style="italic")
fig.savefig("figures/architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- confounding ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axs[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
box(ax, 2.5, 4.2, 5.0, 1.3, "Expected demand\n(season, weekday, store, price)", fc="#fdf0ee", ec=RED, fs=9)
box(ax, 0.3, 0.6, 2.8, 1.2, "Promotion\nrunning?", fs=9.5, bold=True)
box(ax, 6.9, 0.6, 2.8, 1.2, "Units sold", fs=9.5, bold=True)
arrow(ax, (4.0, 4.2), (2.0, 1.8), color=RED, lw=2); arrow(ax, (6.0, 4.2), (8.2, 1.8), color=RED, lw=2)
arrow(ax, (3.1, 1.2), (6.9, 1.2), color=ACC, lw=2.6)
ax.text(5.0, 1.45, "true effect", ha="center", fontsize=8.5, color=ACC, fontweight="bold")
ax.text(1.0, 3.2, "promos are\nscheduled when\ndemand is high", fontsize=8, color=RED, ha="center")
ax.text(9.0, 3.2, "high demand\nraises sales\nanyway", fontsize=8, color=RED, ha="center")
ax.set_title("Why the naive comparison misleads", fontsize=10.5, color="#2d2d2d")
ax = axs[1]
labs = ["True\neffect", "Naive\ncomparison"]; vals = [23.989, 56.829]
ax.bar(labs, vals, color=[ACC, RED], width=0.55)
for i, v in enumerate(vals): ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
ax.text(1, 30, "+137%\ntoo high", ha="center", color="white", fontsize=10, fontweight="bold")
ax.set_ylim(0, 66); ax.set_ylabel("extra units per promo day"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Seed-42 synthetic data", fontsize=10.5, color="#2d2d2d")
fig.tight_layout(); fig.savefig("figures/confounding.png", dpi=200, bbox_inches="tight"); plt.close(fig)
