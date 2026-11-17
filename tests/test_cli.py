from demand_uplift.cli import main


def test_generate_cli(tmp_path):
    rc = main(["generate", "--stores", "2", "--products", "2", "--days", "60",
               "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "synthetic_demand.csv").exists()


def test_report_cli_end_to_end(tmp_path):
    rc = main([
        "report", "--stores", "3", "--products", "2", "--days", "420", "--out", str(tmp_path),
        "--origins", "1", "--series", "2", "--models", "SeasonalNaive,AutoETS,XGBoost",
        "--folds", "2", "--iters", "30", "--trees", "30", "--no-dowhy",
    ])
    assert rc == 0
    for f in ("report.md", "results.json", "backtest_metrics.png", "cate_diagnostics.png",
              "causal_estimates.png", "backtest_forecast.png", "demand_overview.png"):
        assert (tmp_path / f).exists(), f
    assert "ATE" in (tmp_path / "report.md").read_text(encoding="utf-8")
