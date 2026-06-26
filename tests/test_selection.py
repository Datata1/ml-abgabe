"""Selektions-Tests: Oracle vs. label-frei liefern gültige Kandidaten. Marker ``data``."""

from __future__ import annotations

import pytest

from automl_ad.data import load_split, load_validation
from automl_ad.selection import DEFAULT_CANDIDATES, select_internal, select_oracle

_NAMES = {name for name, _ in DEFAULT_CANDIDATES}


@pytest.mark.data
def test_select_oracle_returns_valid_candidate():
    s = load_split(faults=[1, 3], n_train_good_runs=4, n_test_good_runs=3, n_test_fault_runs=3, seed=0)
    X_val, y_val, _ = load_validation(s, n_good_runs=3, n_fault_runs=3, source="testing")
    best, aucs = select_oracle(DEFAULT_CANDIDATES, s.X_train_good, X_val, y_val)
    assert best in _NAMES
    assert set(aucs) == _NAMES
    assert all(0.0 <= v <= 1.0 for v in aucs.values())


@pytest.mark.data
def test_select_internal_returns_valid_candidate():
    s = load_split(faults=[1, 3], n_train_good_runs=4, n_test_good_runs=3, n_test_fault_runs=3, seed=0)
    best, centrality = select_internal(DEFAULT_CANDIDATES, s.X_train_good, s.X_test)
    assert best in _NAMES
    assert set(centrality) == _NAMES
