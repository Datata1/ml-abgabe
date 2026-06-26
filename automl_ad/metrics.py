"""Evaluationsmetriken für Anomalieerkennung.

Threshold-unabhängig: ROC-AUC, PR-AUC (Average Precision), bester F1.
Betriebsspezifisch (Threshold nötig): Detection-Delay, False-Alarm-Rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import config


def best_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Bester F1 über alle Thresholds + zugehöriger Threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    # precision_recall_curve liefert len(thresholds) == len(precision) - 1
    thr = float(thresholds[min(best, len(thresholds) - 1)])
    return float(f1[best]), thr


def detection_delay(
    meta: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    onset: int = config.ONSET_TESTING,
) -> float:
    """Mittlerer Detection-Delay über Fehlerläufe (Samples vom Onset bis zum 1. Alarm).

    Läufe ohne Alarm gehen mit der maximal möglichen Verzögerung ein, damit „nie erkannt"
    bestraft wird.
    """
    df = meta.copy()
    df["score"] = scores
    delays: list[int] = []
    fault_runs = df[df["faultNumber"] != 0].groupby(["faultNumber", "simulationRun"])
    for _, g in fault_runs:
        g = g.sort_values("sample")
        post = g[g["sample"] > onset]
        if post.empty:
            continue
        alarms = post[post["score"] > threshold]
        if alarms.empty:
            delays.append(int(post["sample"].max() - onset))  # nie erkannt
        else:
            delays.append(int(alarms["sample"].iloc[0] - onset))
    return float(np.mean(delays)) if delays else float("nan")


def false_alarm_rate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Anteil fälschlich als Anomalie markierter normaler Punkte."""
    normal = y_true == 0
    if normal.sum() == 0:
        return float("nan")
    return float((scores[normal] > threshold).mean())


def summarize(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    meta: pd.DataFrame | None = None,
    threshold: float | None = None,
    onset: int = config.ONSET_TESTING,
) -> dict:
    """Kompakte Kennzahlensammlung. Threshold/Meta optional für Betriebsmetriken."""
    f1, f1_thr = best_f1(y_true, scores)
    out = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": f1,
    }
    thr = threshold if threshold is not None else f1_thr
    out["false_alarm_rate"] = false_alarm_rate(y_true, scores, thr)
    if meta is not None:
        out["detection_delay"] = detection_delay(meta, scores, thr, onset=onset)
    return out
