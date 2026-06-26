"""Tests für EM/MV (Goix 2016) — **ohne Daten/Netz**, kleine synthetische Eingaben."""

from __future__ import annotations

import numpy as np

from automl_ad.internal_metrics import em_mv_for_detector, select_emmv
from automl_ad.selection import DEFAULT_CANDIDATES


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((400, 6))          # nur "Gutdaten"
    X_eval = np.r_[rng.standard_normal((200, 6)),    # normal + klare Ausreißer
                   rng.standard_normal((20, 6)) * 6 + 9]
    return X_train, X_eval


def test_em_mv_returns_finite_scores():
    X_train, X_eval = _toy()
    em, mv = em_mv_for_detector(
        "iforest", {}, X_train, X_eval,
        n_features_sub=4, n_subspaces=2, n_eval=300, n_generated=2000,
    )
    assert np.isfinite(em) and em >= 0.0
    assert np.isfinite(mv) and mv >= 0.0


def test_select_emmv_returns_valid_candidate():
    X_train, X_eval = _toy()
    names = {name for name, _ in DEFAULT_CANDIDATES}
    best, scores = select_emmv(
        DEFAULT_CANDIDATES, X_train, X_eval,
        n_features_sub=4, n_subspaces=2, n_eval=300, n_generated=2000,
    )
    assert best in names
    assert set(scores) == names
    assert all(np.isfinite(em) and np.isfinite(mv) for em, mv in scores.values())
