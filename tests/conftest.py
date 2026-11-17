import pytest

from demand_uplift.causal import run_causal
from demand_uplift.data import generate


@pytest.fixture(scope="session")
def panel():
    return generate(n_stores=6, n_products=4, n_days=540, seed=7)


@pytest.fixture(scope="session")
def causal_results(panel):
    return run_causal(panel, n_folds=3, iters=60, forest_trees=40, use_dowhy=False, seed=0)
