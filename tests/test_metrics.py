"""Metrik-Tests auf synthetischen Scores — **ohne Datenzugriff**, laufen immer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from automl_ad.metrics import best_f1, detection_delay, false_alarm_rate, summarize


def test_summarize_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])  # höher = anomaler, perfekt getrennt
    m = summarize(y, scores)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["f1"] == 1.0


def test_summarize_random_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    m = summarize(y, scores)
    assert 0.3 < m["roc_auc"] < 0.7  # nahe Zufall


def test_best_f1_returns_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    f1, thr = best_f1(y, scores)
    assert f1 == 1.0
    assert np.isfinite(thr)


def test_false_alarm_rate():
    y = np.array([0, 0, 0, 1])
    scores = np.array([0.0, 1.0, 1.0, 1.0])  # 2 von 3 Normalen über Threshold 0.5
    assert false_alarm_rate(y, scores, threshold=0.5) == 2 / 3


def test_detection_delay_basic():
    meta = pd.DataFrame(
        {"faultNumber": [1] * 6, "simulationRun": [1] * 6, "sample": [1, 2, 3, 4, 5, 6]}
    )
    scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    delay = detection_delay(meta, scores, threshold=0.5, onset=2)
    assert delay == 4 - 2  # erster Alarm bei sample 4, Onset 2
