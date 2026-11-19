import pandas as pd

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


def test_coverage_cli(tmp_path):
    import json

    rc = main(["coverage", "--stores", "2", "--products", "2", "--days", "200", "--n-seeds", "2",
               "--folds", "2", "--iters", "15", "--trees", "8", "--jobs", "1",
               "--out", str(tmp_path)])
    assert rc == 0
    payload = json.loads((tmp_path / "coverage_study.json").read_text(encoding="utf-8"))
    assert payload["config"]["modes"] == ["fixed", "es"]
    assert len(payload["summary"]) == 14
    assert (tmp_path / "coverage_study.png").exists()
    assert (tmp_path / "coverage_raw.csv").exists()
    out2 = tmp_path / "merged"
    rc = main(["coverage", "--from-raw", str(tmp_path / "coverage_raw.csv"), "--out", str(out2)])
    assert rc == 0
    merged = json.loads((out2 / "coverage_study.json").read_text(encoding="utf-8"))
    a, b = pd.DataFrame(merged["summary"]), pd.DataFrame(payload["summary"])
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)  # CSV round trip
