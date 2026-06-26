"""Plots für die Präsentation (Matplotlib). Speichern nach ``reports/``.

Alle Funktionen geben die Matplotlib-Figure zurück (für marimo-Anzeige) und speichern
optional zusätzlich auf Platte.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def _save(fig, save_as: str | None):
    if save_as:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORTS_DIR / save_as if not Path(save_as).is_absolute() else Path(save_as)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def score_timeseries(
    meta: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    fault: int,
    run: int | None = None,
    onset: int = config.ONSET_TESTING,
    save_as: str | None = None,
):
    """Anomaly-Score über die Zeit für einen Beispiel-Lauf, mit Onset + Threshold."""
    df = meta.copy()
    df["score"] = scores
    sub = df[df["faultNumber"] == fault]
    if run is None:
        run = int(sub["simulationRun"].iloc[0])
    sub = sub[sub["simulationRun"] == run].sort_values("sample")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(sub["sample"], sub["score"], lw=1.0, label="Anomaly-Score")
    ax.axhline(threshold, color="tab:red", ls="--", lw=1, label="Threshold")
    ax.axvline(onset, color="tab:green", ls=":", lw=1.5, label=f"Fehler-Onset (>{onset})")
    ax.set(xlabel="sample", ylabel="Score", title=f"Fehler {fault}, Lauf {run}")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, save_as)


def comparison_bars(
    results: dict[str, dict],
    metric: str = "roc_auc",
    save_as: str | None = None,
):
    """Balkenvergleich einer Metrik über mehrere (Methode/Strategie)-Ergebnisse."""
    names = list(results)
    values = [results[n].get(metric, float("nan")) for n in names]

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(names)), 3.5))
    bars = ax.bar(names, values, color="tab:blue")
    ax.set(ylabel=metric, title=f"Vergleich: {metric}")
    ax.set_ylim(0, 1 if metric in {"roc_auc", "pr_auc", "f1"} else None)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, save_as)


def selection_comparison(
    strategies: dict[str, tuple[str, float]],
    oracle_auc: float | None = None,
    save_as: str | None = None,
):
    """Vergleich der Auswahlstrategien: je Strategie der gewählte Detektor + dessen ROC-AUC.

    ``strategies``: ``{Strategie-Label: (gewählter_detektor, roc_auc)}``.
    ``oracle_auc``: optionale horizontale Referenzlinie (Obergrenze).
    """
    labels = list(strategies)
    picks = [strategies[s][0] for s in labels]
    values = [strategies[s][1] for s in labels]

    colors = ["tab:gray", "tab:blue", "tab:green", "tab:orange", "tab:purple"][: len(labels)]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(labels)), 4.0))
    bars = ax.bar(labels, values, color=colors)
    if oracle_auc is not None:
        ax.axhline(oracle_auc, color="tab:red", ls="--", lw=1.2,
                   label=f"Oracle (Obergrenze) = {oracle_auc:.3f}")
        ax.legend(fontsize=8, loc="lower right")
    ax.set(ylabel="ROC-AUC", title="Modellauswahl ohne Labels: Strategien im Vergleich")
    ax.set_ylim(0, 1)
    for b, pick, v in zip(bars, picks, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{pick}\n{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    return _save(fig, save_as)
