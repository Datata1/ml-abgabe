"""Tests für die zeitbewusste Fensterung (automl_ad.ts) — synthetisch, ohne TEP-Daten."""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl_ad.ts import iter_run_sequences, windowed_scores


def _toy_meta(runs, length):
    """Baut (X, meta) mit ``runs`` Läufen à ``length`` Zeitschritten (faultNumber=0)."""
    rows = []
    for r in range(1, runs + 1):
        for s in range(length):
            rows.append((0, r, s))
    meta = pd.DataFrame(rows, columns=["faultNumber", "simulationRun", "sample"])
    rng = np.random.default_rng(runs)
    X = rng.standard_normal((len(meta), 4))
    return X, meta


def test_iter_run_sequences_groups_by_run_without_crossing():
    X, meta = _toy_meta(runs=3, length=10)
    segments = list(iter_run_sequences(X, meta))
    assert len(segments) == 3                       # ein Segment je Lauf
    for rows, _X_run in segments:
        assert len(rows) == 10                      # keine Run-Grenzen überschritten
        assert len({meta["simulationRun"].iloc[i] for i in rows}) == 1


def test_iter_run_sequences_sorts_by_sample():
    X, meta = _toy_meta(runs=1, length=5)
    shuffled = meta.iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)
    X_shuf = X[[3, 0, 4, 1, 2]]
    (rows, _), = list(iter_run_sequences(X_shuf, shuffled))
    assert list(shuffled["sample"].to_numpy()[rows]) == [0, 1, 2, 3, 4]


def test_windowed_scores_length_and_finite():
    X_train, meta_train = _toy_meta(runs=4, length=120)
    X_eval, meta_eval = _toy_meta(runs=2, length=120)
    scores = windowed_scores(
        "iforest", X_train, meta_train, X_eval, meta_eval, window_size=20, step=1
    )
    assert scores.shape == (len(X_eval),)
    assert np.isfinite(scores).all()


def test_windowed_scores_handles_short_runs():
    X_train, meta_train = _toy_meta(runs=3, length=80)
    # Eval-Lauf kürzer als window_size → darf nicht crashen, bleibt endlich (aufgefüllt).
    X_eval, meta_eval = _toy_meta(runs=1, length=10)
    scores = windowed_scores(
        "iforest", X_train, meta_train, X_eval, meta_eval, window_size=30, step=1
    )
    assert scores.shape == (10,)
    assert np.isfinite(scores).all()
