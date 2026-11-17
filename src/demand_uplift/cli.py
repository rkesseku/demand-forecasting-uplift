"""Command line interface: ``python -m demand_uplift {generate,backtest,causal,report}``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import generate


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stores", type=int, default=8)
    p.add_argument("--products", type=int, default=5)
    p.add_argument("--days", type=int, default=1095)
    p.add_argument("--out", type=Path, default=Path("output"))


def _add_bt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--origins", type=int, default=4)
    p.add_argument("--step", type=int, default=28)
    p.add_argument("--series", type=int, default=6, help="series scored by every model")
    p.add_argument(
        "--models", default="SeasonalNaive,AutoETS,AutoARIMA_x,Prophet,XGBoost",
        help="comma-separated subset of SeasonalNaive,AutoETS,AutoARIMA_x,Prophet,XGBoost",
    )


def _add_causal_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iters", type=int, default=100, help="boosting iterations for nuisances")
    p.add_argument("--trees", type=int, default=100, help="causal forest trees")
    p.add_argument("--no-dowhy", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="demand_uplift", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_data_args(sub.add_parser("generate", help="write synthetic panel to CSV"))
    p = sub.add_parser("backtest", help="rolling-origin forecast backtest")
    _add_data_args(p)
    _add_bt_args(p)
    p = sub.add_parser("causal", help="causal uplift analysis")
    _add_data_args(p)
    _add_causal_args(p)
    p = sub.add_parser("report", help="run everything; write report.md + figures")
    _add_data_args(p)
    _add_bt_args(p)
    _add_causal_args(p)
    return ap


def _make(args) -> pd.DataFrame:
    return generate(args.stores, args.products, args.days, args.seed)


def _bt_cfg(args):
    from .forecast import BacktestConfig

    return BacktestConfig(
        horizon=args.horizon, n_origins=args.origins, step=args.step, n_series=args.series,
        models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
    )


def _meta(args, df: pd.DataFrame) -> dict:
    return {
        "seed": args.seed, "n_stores": args.stores, "n_products": args.products,
        "n_days": args.days, "n_rows": len(df), "promo_rate": float(df["promo"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = _make(args)

    if args.cmd == "generate":
        path = args.out / "synthetic_demand.csv"
        df.to_csv(path, index=False)
        print(f"wrote {len(df):,} rows to {path}")
        return 0

    from . import report
    from .causal import run_causal
    from .forecast import backtest

    bt = ca = None
    if args.cmd in ("backtest", "report"):
        bt = backtest(df, _bt_cfg(args))
        print(report.md_table(bt.summary))
    if args.cmd in ("causal", "report"):
        ca = run_causal(
            df, n_folds=args.folds, iters=args.iters, forest_trees=args.trees,
            use_dowhy=not args.no_dowhy, seed=0,
        )
        print(f"true ATE = {ca.truth['ate']:.3f}")
        print(report.md_table(ca.estimates[["method", "estimate", "ci_low", "ci_high"]]))
    path = report.write_report(args.out, _meta(args, df), bt, ca)
    report.fig_overview(df, args.out / "demand_overview.png")
    if bt is not None:
        report.fig_backtest(bt, args.out / "backtest_metrics.png", args.out / "backtest_forecast.png")
    if ca is not None:
        report.fig_causal(ca, args.out / "causal_estimates.png", args.out / "cate_diagnostics.png")
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
