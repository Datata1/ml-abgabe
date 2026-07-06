"""Einheitliches Detektor-Interface + Registry der klassischen PyOD-Detektoren.

Wir standardisieren auf die PyOD-``BaseDetector``-Signatur (``fit`` / ``decision_function`` mit
„höher = anomaler" / ``decision_scores_`` / ``threshold_``). Jede Auswahlstrategie spricht nur
gegen dieses Protokoll und ist damit unabhängig vom konkreten Detektor austauschbar.

Bewusst schlank: nur vier klassische, dependency-leichte Detektoren (kein torch/Deep Learning).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np
from pyod.models.hdbscan import HDBSCAN
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA

from . import config


@runtime_checkable
class AnomalyDetector(Protocol):
    """Minimal-Kontrakt (von allen PyOD-Detektoren erfüllt)."""

    decision_scores_: np.ndarray
    threshold_: float

    def fit(self, X: np.ndarray) -> AnomalyDetector: ...
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...  # höher = anomaler


# --- Factories ---------------------------------------------------------------------
def make_iforest(**hp):
    params = {"random_state": config.RANDOM_SEED, "contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return IForest(**params)


def make_pca(**hp):
    params = {"random_state": config.RANDOM_SEED, "contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return PCA(**params)


def make_knn(**hp):
    # Score = Abstand zum k-nächsten Nachbarn (Standard: k=5, method='largest').
    params = {"contamination": config.DEFAULT_CONTAMINATION, "n_jobs": -1}
    params.update(hp)
    return KNN(**params)


def make_hdbscan(**hp):
    # Dichte-basiert (GLOSH-Outlier-Score); scored neue Punkte über approximate_predict.
    params = {"contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return HDBSCAN(**params)


# Registry: name -> factory(**hyperparams) -> Detector
REGISTRY: dict[str, Callable] = {
    "knn": make_knn,
    "pca": make_pca,
    "hdbscan": make_hdbscan,
    "iforest": make_iforest,
}


def make_detector(name: str, **hp) -> AnomalyDetector:
    """Erzeugt einen Detektor über die Registry."""
    if name not in REGISTRY:
        raise KeyError(f"Unbekannter Detektor '{name}'. Verfügbar: {sorted(REGISTRY)}")
    return REGISTRY[name](**hp)


def available_detectors() -> list[str]:
    return sorted(REGISTRY)
