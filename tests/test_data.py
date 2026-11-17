import numpy as np

from demand_uplift.data import generate, true_effects


def test_reproducible_and_seed_sensitive():
    a = generate(3, 2, 120, seed=1)
    b = generate(3, 2, 120, seed=1)
    c = generate(3, 2, 120, seed=2)
    assert a.equals(b)
    assert not a["units"].equals(c["units"])


def test_shape_and_columns(panel):
    assert len(panel) == 6 * 4 * 540
    assert panel["series_id"].nunique() == 24
    assert (panel["units"] >= 0).all()
    assert 0.08 < panel["promo"].mean() < 0.4


def test_promo_is_confounded_with_baseline_demand(panel):
    # Promo days have systematically higher no-promo baseline demand.
    base = panel["true_baseline"]
    assert base[panel["promo"] == 1].mean() > 1.15 * base[panel["promo"] == 0].mean()


def test_effect_is_heterogeneous_and_truth_consistent(panel):
    tau = panel["true_tau"]
    assert tau.std() / tau.mean() > 0.3
    te = true_effects(panel)
    assert np.isclose(te["ate"], tau.mean())
    assert te["att"] > te["ate"]  # promos land where effects are larger
