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
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
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


def make_ocsvm(**hp):
    params = {"contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return OCSVM(**params)


def make_pca(**hp):
    params = {"random_state": config.RANDOM_SEED, "contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return PCA(**params)


def make_ecod(**hp):
    # ECOD ist parameterfrei (außer contamination) — robuster, statistischer Default-Anker.
    params = {"contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return ECOD(**params)


# Registry: name -> factory(**hyperparams) -> Detector
REGISTRY: dict[str, Callable] = {
    "ecod": make_ecod,
    "iforest": make_iforest,
    "ocsvm": make_ocsvm,
    "pca": make_pca,
}


def make_detector(name: str, **hp) -> AnomalyDetector:
    """Erzeugt einen Detektor über die Registry."""
    if name not in REGISTRY:
        raise KeyError(f"Unbekannter Detektor '{name}'. Verfügbar: {sorted(REGISTRY)}")
    return REGISTRY[name](**hp)


def available_detectors() -> list[str]:
    return sorted(REGISTRY)
