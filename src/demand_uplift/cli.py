"""Command line interface: ``python -m demand_uplift {generate,backtest,causal,coverage,report}``."""

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
    p.add_argument(
        "--no-xgb-es", action="store_true",
        help="XGBoost with the fixed 300 trees instead of time-ordered early stopping",
    )


def _add_causal_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iters", type=int, default=100, help="boosting iterations for nuisances")
    p.add_argument("--trees", type=int, default=100, help="causal forest trees")
    p.add_argument(
        "--nuisance", choices=["fixed", "es"], default="fixed",
        help="fixed: <= --iters boosting rounds; es: <= 1000 rounds, early stopping on a 10%% "
        "validation split of each training fold",
    )


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
    p.add_argument("--no-dowhy", action="store_true")
    p = sub.add_parser(
        "coverage", help="ATE bias / RMSE / 95%% CI coverage over many re-generated data seeds",
    )
    _add_data_args(p)
    _add_causal_args(p)
    p.add_argument("--n-seeds", type=int, default=50, help="data seeds --seed .. --seed+n-1")
    p.add_argument(
        "--modes", default="fixed,es", help="comma-separated nuisance settings to compare",
    )
    p.add_argument("--jobs", type=int, default=2, help="causal forest worker processes")
    p.add_argument("--resume", action="store_true", help="skip seeds already in coverage_raw.csv")
    p.add_argument("--from-raw", nargs="+", type=Path, help="only summarise these raw CSVs")
    p = sub.add_parser("report", help="run everything; write report.md + figures")
    _add_data_args(p)
    _add_bt_args(p)
    _add_causal_args(p)
    p.add_argument("--no-dowhy", action="store_true")
    return ap


def _make(args) -> pd.DataFrame:
    return generate(args.stores, args.products, args.days, args.seed)


def _bt_cfg(args):
    from .forecast import BacktestConfig

    return BacktestConfig(
        horizon=args.horizon, n_origins=args.origins, step=args.step, n_series=args.series,
        models=tuple(m.strip() for m in args.models.split(",") if m.strip()),
        xgb_early_stopping=not args.no_xgb_es,
    )


def _meta(args, df: pd.DataFrame) -> dict:
    return {
        "seed": args.seed, "n_stores": args.stores, "n_products": args.products,
        "n_days": args.days, "n_rows": len(df), "promo_rate": float(df["promo"].mean()),
    }


def run_coverage(args) -> int:
    import json

    from . import report
    from .causal import coverage_study, summarize_coverage

    raw_path = args.out / "coverage_raw.csv"
    if args.from_raw:  # summarise finished runs (e.g. modes run as separate processes)
        raw = pd.concat([pd.read_csv(f) for f in args.from_raw], ignore_index=True)
        raw = raw.drop_duplicates(["nuisance", "seed", "method"], keep="last")
    else:
        seeds = list(range(args.seed, args.seed + args.n_seeds))
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        done = pd.read_csv(raw_path) if args.resume and raw_path.exists() else None
        frames = [] if done is None else [done]

        def ckpt(r: pd.DataFrame) -> None:  # frames holds only earlier results at this point
            pd.concat([*frames, r]).to_csv(raw_path, index=False)
            last = r.iloc[-1]
            print(f"[{last['nuisance']}] seed {int(last['seed'])} done "
                  f"({last['seconds']:.1f}s)", flush=True)

        for mode in modes:
            todo = seeds if done is None else [
                s for s in seeds if not ((done["nuisance"] == mode) & (done["seed"] == s)).any()
            ]
            if todo:
                frames.append(coverage_study(
                    todo, args.stores, args.products, args.days, n_folds=args.folds,
                    iters=args.iters, forest_trees=args.trees, nuisance=mode, n_jobs=args.jobs,
                    callback=ckpt,
                ))
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(raw_path, index=False)
    summ = summarize_coverage(raw)
    per_seed = raw.drop_duplicates(["nuisance", "seed"]).groupby("nuisance")["seconds"]
    payload = {
        "config": {
            "seeds": sorted(int(s) for s in raw["seed"].unique()),
            "n_stores": args.stores, "n_products": args.products, "n_days": args.days,
            "folds": args.folds, "iters": args.iters, "trees": args.trees,
            "modes": list(dict.fromkeys(raw["nuisance"])), "level": 0.95,
            "coverage_interval": "Wilson 95%",
        },
        "seconds_per_seed": {k: {"mean": float(v.mean()), "total": float(v.sum())}
                             for k, v in per_seed},
        "summary": summ.to_dict(orient="records"),
    }
    (args.out / "coverage_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.fig_coverage(summ, raw, args.out / "coverage_study.png")
    cols = ["nuisance", "method", "mean_bias", "rmse", "se_ratio", "coverage_ate",
            "coverage_ate_lo", "coverage_ate_hi", "mean_width"]
    print(report.md_table(summ[cols]))
    print(f"wrote {args.out / 'coverage_study.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "coverage":
        return run_coverage(args)
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
            use_dowhy=not args.no_dowhy, seed=0, nuisance=args.nuisance,
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
