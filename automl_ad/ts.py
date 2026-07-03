"""Zeitbewusste Detektion: PyOD-Detektoren über gleitende Fenster (pro Simulationslauf).

TEP ist eine **Zeitreihe** — statt jede Zeile isoliert (i.i.d.) zu bewerten, betrachten wir
gleitende **Fenster**. Wir nutzen dieselben PyOD-Primitiven wie ``pyod.models.TimeSeriesOD``
(``sliding_windows`` / ``map_scores_to_timestamps``).

**Warum nicht direkt ``TimeSeriesOD``?** Der Adapter arbeitet auf **einer** zusammenhängenden
Sequenz. TEP besteht aber aus **vielen Simulationsläufen**: (1) fittet man ``TimeSeriesOD`` auf allen
Läufen am Stück, entstehen Fenster, die **Run-Grenzen überschreiten** (Ende Lauf A + Anfang Lauf B);
(2) für ein robustes „Normal" wollen wir **einen** Detektor auf den Fenstern **aller** Gut-Läufe
fitten — ``TimeSeriesOD`` fittet aber pro Aufruf nur auf einer Sequenz (kein Pooling). Dieses Modul
liefert genau diese **Mehr-Lauf-Orchestrierung** (pro Lauf fenstern, Gut-Fenster poolen, je Eval-Lauf
zurückmappen) um dieselben Primitiven — also „TimeSeriesOD, aber run-aware".

Ergebnis von :func:`windowed_scores`: ein Anomalie-Score **pro Zeitstempel**, in derselben
Reihenfolge wie ``X_eval`` — damit läuft die Score-Dict-API aus :mod:`automl_ad.selection`
unverändert auf zeitbewussten Scores.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from pyod.models._ts_utils import map_scores_to_timestamps, sliding_windows

from .detectors import make_detector
from .selection import Candidate, Scores

# Ein zusammenhängender Lauf ist durch (faultNumber, simulationRun) eindeutig; innerhalb nach sample.
RUN_KEYS = ["faultNumber", "simulationRun"]


def iter_run_sequences(X: np.ndarray, meta: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Iteriert ``(row_index, X_run)`` je zusammenhängendem Lauf ``(faultNumber, simulationRun)``.

    ``row_index`` sind die Positionszeilen in ``X`` (nach ``sample`` sortiert), ``X_run`` das
    zugehörige, zeitlich geordnete Segment. So können Scores exakt zurückgemappt werden.
    """
    X = np.asarray(X, dtype=float)
    ordered = (
        meta.reset_index(drop=True)
        .assign(_pos=np.arange(len(meta)))
        .sort_values([*RUN_KEYS, "sample"])
    )
    for _, sub in ordered.groupby(RUN_KEYS, sort=False):
        rows = sub["_pos"].to_numpy()
        yield rows, X[rows]


def windowed_scores(
    name: str,
    X_train_good: np.ndarray,
    meta_train: pd.DataFrame,
    X_eval: np.ndarray,
    meta_eval: pd.DataFrame,
    *,
    window_size: int = 50,
    step: int = 1,
    aggregation: str = "max",
    **hp,
) -> np.ndarray:
    """Zeitbewusster Score eines Detektors: fenstern → fitten (gepoolt) → scoren → zurückmappen.

    - **Fit:** Fenster je Gut-Lauf bilden, alle poolen, inneren Detektor darauf fitten.
    - **Score:** je Eval-Lauf fenstern, Fenster scoren, per ``max``/``mean`` auf Zeitstempel mappen.

    Rückgabe: Score-Vektor der Länge ``len(X_eval)`` (höher = anomaler). Läufe kürzer als
    ``window_size`` bekommen den minimalen (am wenigsten anomalen) Score.
    """
    train_windows = [
        sliding_windows(X_run, window_size, step)
        for _, X_run in iter_run_sequences(X_train_good, meta_train)
        if len(X_run) >= window_size
    ]
    if not train_windows:
        raise ValueError(f"Kein Trainingslauf ist lang genug für window_size={window_size}.")
    detector = make_detector(name, **hp).fit(np.vstack(train_windows))

    scores = np.full(len(X_eval), np.nan)
    for rows, X_run in iter_run_sequences(X_eval, meta_eval):
        if len(X_run) < window_size:
            continue  # zu kurz zum Fenstern → bleibt NaN, wird unten gefüllt
        window_scores = detector.decision_function(sliding_windows(X_run, window_size, step))
        mapped, _ = map_scores_to_timestamps(
            window_scores, window_size, step, len(X_run), aggregation=aggregation
        )
        scores[rows] = mapped

    missing = np.isnan(scores)
    if missing.any():  # kurze Läufe / Randlücken: neutral (Minimum) auffüllen
        fill = float(np.nanmin(scores)) if not missing.all() else 0.0
        scores[missing] = fill
    return scores


def windowed_candidate_scores(
    candidates: list[Candidate],
    split,
    *,
    window_size: int = 50,
    step: int = 1,
    aggregation: str = "max",
) -> Scores:
    """Zeitbewusster Producer: ``{detektorname: score_vektor}`` für :mod:`automl_ad.selection`.

    Gegenstück zu :func:`automl_ad.selection.fit_scores` (i.i.d.), aber gefenstert auf ``split``.
    """
    return {
        name: windowed_scores(
            name,
            split.X_train_good,
            split.meta_train,
            split.X_test,
            split.meta_test,
            window_size=window_size,
            step=step,
            aggregation=aggregation,
            **hp,
        )
        for name, hp in candidates
    }
