"""Interne, label-freie Bewertungsmaße: Excess-Mass (EM) & Mass-Volume (MV).

Nach Goix (2016), *„How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?"*
(arXiv:1607.01152). Beide bewerten eine Anomalie-Scoring-Funktion **ohne Labels** über die
Geometrie ihrer Level-Sets: ein guter Scorer konzentriert viel „Normal-Masse" auf **kleinem
Volumen**.

- **EM**: höher = besser.   - **MV**: niedriger = besser.

Das Volumen wird per Monte-Carlo geschätzt (uniforme Punkte im Feature-Bounding-Box). Weil das
in hoher Dimension unzuverlässig wird, mittelt man über **zufällige niedrigdimensionale
Feature-Subräume** (Standardvorgehen aus dem Paper). Damit ist EM/MV eine zweite interne Metrik
neben dem Konsens (`selection.select_internal`).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc

from .detectors import make_detector


def _em_auc(t, t_max, volume_support, s_unif, s_X, n_generated) -> float:
    """Fläche unter der Excess-Mass-Kurve (höher = besser)."""
    em_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    em_t[0] = 1.0
    for u in np.unique(s_X):
        em_t = np.maximum(
            em_t,
            (s_X > u).sum() / n_samples
            - t * (s_unif > u).sum() / n_generated * volume_support,
        )
    amax = int(np.argmax(em_t <= t_max)) + 1
    if amax == 1:  # t_max nie erreicht -> über die ganze Kurve integrieren
        amax = t.shape[0]
    return float(auc(t[:amax], em_t[:amax]))


def _mv_auc(axis_alpha, volume_support, s_unif, s_X, n_generated) -> float:
    """Fläche unter der Mass-Volume-Kurve (niedriger = besser)."""
    n_samples = s_X.shape[0]
    order = s_X.argsort()
    mass = 0.0
    cpt = 0
    u = s_X[order[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        while mass < axis_alpha[i] and cpt < n_samples:
            cpt += 1
            u = s_X[order[-cpt]]
            mass = cpt / n_samples
        mv[i] = (s_unif >= u).sum() / n_generated * volume_support
    return float(auc(axis_alpha, mv))


def em_mv_for_detector(
    name: str,
    hp: dict,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    n_features_sub: int = 5,
    n_subspaces: int = 5,
    n_eval: int = 2000,
    n_generated: int = 10000,
    t_max: float = 0.9,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
    seed: int = 0,
) -> tuple[float, float]:
    """EM- und MV-Score eines Detektors, gemittelt über zufällige Feature-Subräume.

    Der Detektor wird je Subraum auf den (Gut-)Trainingsdaten gefittet; Scores werden als
    „Normalität" = ``-decision_function`` interpretiert (höher = normaler). Rückgabe ``(em, mv)``.
    """
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    k = min(n_features_sub, d)

    # Eval-Subsample für Tempo (EM/MV iterieren über die eindeutigen Scores).
    if X_eval.shape[0] > n_eval:
        X_eval = X_eval[rng.choice(X_eval.shape[0], size=n_eval, replace=False)]

    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    # volume_support = 1: jeder Subraum wird auf den Einheitswürfel [0,1]^k normalisiert.
    # Sonst sprengen Ausreißer die Bounding-Box und EM kollabiert numerisch (Goix, hohe Dim.).
    volume_support = 1.0
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)

    ems: list[float] = []
    mvs: list[float] = []
    for _ in range(n_subspaces):
        feats = rng.choice(d, size=k, replace=False)
        lo = X_eval[:, feats].min(axis=0)
        hi = X_eval[:, feats].max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)

        def _scale(a, lo=lo, span=span):
            return (a - lo) / span

        det = make_detector(name, **hp).fit(_scale(X_train[:, feats]))
        s_X = -det.decision_function(_scale(X_eval[:, feats]))
        unif = rng.uniform(0.0, 1.0, size=(n_generated, k))
        s_unif = -det.decision_function(unif)

        ems.append(_em_auc(t, t_max, volume_support, s_unif, s_X, n_generated))
        mvs.append(_mv_auc(axis_alpha, volume_support, s_unif, s_X, n_generated))

    return float(np.mean(ems)), float(np.mean(mvs))
