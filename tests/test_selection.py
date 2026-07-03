"""Selektions-Tests: Score-Dict-API (schnell, ohne Daten) + i.i.d.-Wrapper (Marker ``data``)."""

from __future__ import annotations

import numpy as np
import pytest

from automl_ad.data import load_split, load_validation
from automl_ad.selection import (
    DEFAULT_CANDIDATES,
    agreement,
    consensus_centrality,
    ensemble_consensus,
    oracle_best,
    per_fault_breakdown,
    select_internal,
    select_oracle,
)

_NAMES = {name for name, _ in DEFAULT_CANDIDATES}


def _toy_scores(n=200, seed=0):
    """Drei einige Detektoren (a,b,c korreliert) + ein Außenseiter (d)."""
    rng = np.random.default_rng(seed)
    base = rng.random(n)
    return {
        "a": base + 0.05 * rng.random(n),
        "b": base + 0.05 * rng.random(n),
        "c": base + 0.05 * rng.random(n),
        "d": rng.random(n),  # Außenseiter
    }


def test_consensus_centrality_prefers_mainstream():
    scores = _toy_scores()
    best, centrality = consensus_centrality(scores)
    assert set(centrality) == set(scores)
    assert best in {"a", "b", "c"}          # der Außenseiter d gewinnt nicht
    assert centrality["d"] < centrality[best]


def test_ensemble_consensus_shapes_and_methods():
    scores = _toy_scores()
    for method in ("average", "maximization", "median"):
        out = ensemble_consensus(scores, method)
        assert out.shape == (200,)
        assert np.isfinite(out).all()
    with pytest.raises(ValueError):
        ensemble_consensus(scores, "unknown")


def test_agreement_high_for_correlated_low_for_random():
    corr = {"a": np.arange(100.0), "b": np.arange(100.0), "c": np.arange(100.0)}
    assert agreement(corr) > 0.99
    rng = np.random.default_rng(1)
    rand = {k: rng.random(100) for k in "abc"}
    assert agreement(rand) < 0.5


def test_per_fault_breakdown_separates_easy_and_hard():
    # 50 Gutpunkte + Fehler 1 (perfekt trennbar) + Fehler 2 (Scores wie Gutdaten = unsichtbar).
    rng = np.random.default_rng(0)
    fault_no = np.r_[np.zeros(50), np.full(30, 1), np.full(30, 2)].astype(int)
    y = (fault_no != 0).astype(int)
    easy_hard = np.r_[rng.random(50), 1.0 + rng.random(30), rng.random(30)]
    scores = {"a": easy_hard, "b": easy_hard + 0.01 * rng.random(110)}
    out = per_fault_breakdown(scores, easy_hard, y, fault_no)
    assert set(out) == {1, 2}
    assert out[1]["roc_auc"] == 1.0                    # leichter Fehler: perfekt
    assert out[2]["roc_auc"] < 0.7                     # schwerer Fehler: nahe Zufall
    assert all(v["agreement"] > 0.9 for v in out.values())  # Schwarm trotzdem einig


def test_oracle_best_picks_highest_auc():
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    scores = {"good": np.r_[np.zeros(50), np.ones(50)], "bad": np.r_[np.ones(50), np.zeros(50)]}
    best, aucs = oracle_best(scores, y)
    assert best == "good"
    assert aucs["good"] == 1.0 and aucs["bad"] == 0.0


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
