"""Daten-Invarianten: Leakage-Freiheit, Onset-Labeling, Shapes. Marker ``data``."""

from __future__ import annotations

import numpy as np
import pytest

from automl_ad import config
from automl_ad.data import load_split, load_validation, make_labels


@pytest.mark.data
def test_split_shapes_and_features():
    s = load_split(faults=[1, 3], n_train_good_runs=3, n_test_good_runs=2, n_test_fault_runs=2, seed=0)
    assert s.X_train_good.shape[1] == len(config.FEATURE_COLS) == 52
    assert s.X_test.shape[1] == 52
    assert s.X_test.shape[0] == len(s.y_test) == len(s.meta_test)


@pytest.mark.data
def test_onset_labeling_pre_onset_is_normal():
    s = load_split(faults=[1, 3], n_train_good_runs=3, n_test_good_runs=2, n_test_fault_runs=2, seed=0)
    m = s.meta_test
    pre_onset_faulty = (m["faultNumber"] != 0) & (m["sample"] <= config.ONSET_TESTING)
    assert (s.y_test[pre_onset_faulty.to_numpy()] == 0).all()
    post = (m["faultNumber"] != 0) & (m["sample"] > config.ONSET_TESTING)
    assert (s.y_test[post.to_numpy()] == 1).all()


@pytest.mark.data
def test_validation_runs_disjoint_from_test():
    s = load_split(faults=[1, 3], n_train_good_runs=3, n_test_good_runs=3, n_test_fault_runs=3, seed=0)
    load_validation(s, n_good_runs=3, n_fault_runs=3, source="testing")
    assert len(set(s.test_fault_runs)) == len(s.test_fault_runs)


@pytest.mark.data
def test_scaler_fitted_on_good_only():
    s = load_split(faults=[1], n_train_good_runs=4, n_test_good_runs=2, n_test_fault_runs=2, seed=0)
    assert np.allclose(s.X_train_good.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(s.X_train_good.std(axis=0), 1, atol=1e-2)


def test_make_labels_helper():
    import pandas as pd

    df = pd.DataFrame({"faultNumber": [0, 1, 1], "sample": [500, 100, 200]})
    y = make_labels(df, onset=config.ONSET_TESTING)
    assert list(y) == [0, 0, 1]  # gut; faulty pre-onset; faulty post-onset
