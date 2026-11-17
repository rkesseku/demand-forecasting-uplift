import numpy as np
import pandas as pd

from demand_uplift.features import build_features, feature_columns


def test_features_do_not_use_future_target(panel):
    h = 14
    cutoff = panel["date"].min() + pd.Timedelta(days=400)
    base = build_features(panel, h)
    tampered = panel.copy()
    tampered.loc[tampered["date"] > cutoff, "units"] = 1e6  # destroy the future
    alt = build_features(tampered, h)
    # Features for any row dated <= cutoff + h may only depend on y up to cutoff.
    safe = base["date"] <= cutoff + pd.Timedelta(days=h)
    cols = feature_columns(h)
    pd.testing.assert_frame_equal(base.loc[safe, cols], alt.loc[safe, cols])
    # ...and later rows do change, so the test has power.
    later = base["date"] > cutoff + pd.Timedelta(days=h + 30)
    assert not np.allclose(
        base.loc[later, f"lag_{h}"].fillna(0), alt.loc[later, f"lag_{h}"].fillna(0)
    )


def test_no_truth_columns_in_features():
    assert not any(c.startswith("true_") or c == "units" for c in feature_columns(14))
