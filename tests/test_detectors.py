"""Detektor-Kontrakt: tabellarische, unüberwachte Detektoren. Marker ``data``."""

from __future__ import annotations

import numpy as np
import pytest

from automl_ad.data import load_split
from automl_ad.detectors import available_detectors, make_detector

CORE_DETECTORS = ["ecod", "iforest", "ocsvm", "pca"]


@pytest.mark.data
@pytest.mark.parametrize("name", CORE_DETECTORS)
def test_detector_contract(name):
    s = load_split(faults=[1], n_train_good_runs=3, n_test_good_runs=2, n_test_fault_runs=2, seed=0)
    # Kleines Trainings-Subsample für Tempo (z. B. OCSVM).
    det = make_detector(name).fit(s.X_train_good[:1200])
    scores = det.decision_function(s.X_test)
    assert scores.shape[0] == s.X_test.shape[0]
    assert np.all(np.isfinite(scores))
    assert hasattr(det, "threshold_") and np.isfinite(det.threshold_)
    assert hasattr(det, "decision_scores_")


def test_registry_contains_core_detectors():
    av = available_detectors()
    for name in CORE_DETECTORS:
        assert name in av


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        make_detector("gibt_es_nicht")
