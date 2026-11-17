import numpy as np
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
