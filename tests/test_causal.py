import numpy as np
import pandas as pd
import pytest

from demand_uplift.causal import calibration_slope, uplift_auc_gain


def _row(res, method):
    return res.estimates.set_index("method").loc[method]


# Cross-fitted GBM nuisances leave a small finite-sample bias (a few percent of the ATE) that
# the sandwich / influence-function SEs do not capture. "Honest tolerance" below therefore pads
# the 95% CI by a bias allowance of 15% of the true effect. The naive estimate misses by >100%.
BIAS_ALLOWANCE = 0.15


def _covers(r, target, ate):
    pad = BIAS_ALLOWANCE * ate
    return r["ci_low"] - pad <= target <= r["ci_high"] + pad


def test_naive_estimate_is_badly_biased(causal_results):
    r = _row(causal_results, "naive")
    ate = causal_results.truth["ate"]
    assert r["estimate"] > 2.0 * ate
    assert not r["covers_ate"]
    # adjusting only linearly for covariates still leaves large bias
    assert _row(causal_results, "regression")["bias"] > 0.3 * ate


def test_aipw_and_dr_cover_true_ate(causal_results):
    ate = causal_results.truth["ate"]
    for m in ("AIPW", "DRLearner", "LinearDML-het"):
        r = _row(causal_results, m)
        assert _covers(r, ate, ate), (m, r.to_dict(), ate)
        assert abs(r["bias"]) < 0.15 * ate, m


def test_dml_covers_its_estimand_and_is_near_ate(causal_results):
    r = _row(causal_results, "LinearDML")
    t = causal_results.truth
    # Partially-linear DML targets the Var(D|X)-weighted effect; it must cover that and be
    # far closer to the ATE than the naive difference.
    assert _covers(r, t["plr_weighted"], t["ate"])
    assert abs(r["bias"]) < 0.3 * t["ate"]
    assert abs(r["bias"]) < abs(_row(causal_results, "naive")["bias"]) / 4


def test_causal_forest_ci_covers_ate(causal_results):
    r = _row(causal_results, "CausalForestDML")
    assert r["covers_ate"], r.to_dict()


def test_xlearner_cate_correlates_with_truth(causal_results):
    cm = causal_results.cate_metrics.set_index("learner")
    assert cm.loc["X-learner", "corr_true"] > 0.6
    assert cm.loc["X-learner", "calib_slope"] > 0.5


def test_uplift_helpers_on_known_scores():
    rng = np.random.default_rng(0)
    tau = rng.normal(1.0, 1.0, 5000)
    psi = tau + rng.normal(0, 0.1, 5000)
    slope, _ = calibration_slope(tau, psi)
    assert abs(slope - 1) < 0.05
    assert uplift_auc_gain(tau, psi) > 0.2
    assert abs(uplift_auc_gain(rng.normal(size=5000), psi)) < 0.1


def test_dowhy_identification_estimate_and_refutation(panel):
    pytest.importorskip("dowhy")
    from demand_uplift.causal import dowhy_analysis

    out = dowhy_analysis(panel, n=800)
    assert len(out["backdoor_set"]) == 8
    assert np.isfinite(out["estimate_stratification"])
    assert set(out["refutations"]) == {"placebo_treatment", "random_common_cause", "data_subset"}
    placebo = out["refutations"]["placebo_treatment"]
    assert abs(placebo["new_effect"]) < 0.25 * abs(out["estimate_stratification"])


def test_wilson_interval_known_values():
    from demand_uplift.causal import wilson_interval

    lo, hi = wilson_interval(19, 20)  # 95% observed with n=20
    assert 0.763 < lo < 0.765 and 0.990 < hi < 0.992
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0 and 0.11 < hi < 0.12
    lo, hi = wilson_interval(15, 30)
    assert np.isclose(0.5 - lo, hi - 0.5)


def test_coverage_study_on_tiny_panels():
    from demand_uplift.causal import coverage_study, summarize_coverage

    kw = {"n_stores": 2, "n_products": 2, "n_days": 200, "n_folds": 2, "iters": 15,
          "forest_trees": 8, "n_jobs": 1}
    raw = pd.concat([coverage_study([1, 2], nuisance=m, **kw) for m in ("fixed", "es")])
    assert len(raw) == 2 * 2 * 7
    assert raw.groupby(["nuisance", "seed"])["true_ate"].nunique().eq(1).all()
    summ = summarize_coverage(raw)
    assert len(summ) == 2 * 7 and (summ["n_seeds"] == 2).all()
    for c in ("coverage_ate", "coverage_ate_lo", "coverage_ate_hi", "coverage_plr"):
        assert summ[c].between(0, 1).all(), c
    assert (summ["coverage_ate_lo"] <= summ["coverage_ate"]).all()
    assert (summ["coverage_ate"] <= summ["coverage_ate_hi"]).all()
    assert (summ["mean_width"] > 0).all() and (summ["rmse"] >= summ["mean_bias"].abs()).all()
    naive = summ[summ["method"] == "naive"]
    assert (naive["mean_bias"] > 0).all()  # confounding biases the naive contrast upward


def test_nuisance_modes():
    from demand_uplift.causal import ES_MAX_ITER, _reg

    assert _reg(0, 100, "fixed").max_iter == 100
    es = _reg(0, 100, "es")
    assert es.max_iter == ES_MAX_ITER and es.early_stopping is True
    with pytest.raises(ValueError):
        _reg(0, 100, "tuned")
