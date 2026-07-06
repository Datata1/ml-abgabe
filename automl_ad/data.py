"""Daten-Loading, Run-Level-Split, Skalierung und Onset-korrektes Labeling (TEP).

Konventionen:
- Split **nach simulationRun** (nie zeilenweise), Train-/Test-Läufe disjunkt → kein Leakage.
- Unüberwachte AD: Training nur auf Gutdaten; StandardScaler nur auf Gutdaten gefittet.
- Punktweises Label berücksichtigt den Fehler-Onset (faultNumber!=0 ≠ jede Zeile anomal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------------------
# Roh-Loading (mit Predicate-Pushdown auf simulationRun / faultNumber)
# --------------------------------------------------------------------------------------
def _read_runs(
    path,
    runs: list[int],
    faults: list[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Liest nur die gewünschten Läufe (und optional Fehlertypen) aus einer Parquet-Datei."""
    filters = [("simulationRun", "in", list(runs))]
    if faults is not None:
        filters.append(("faultNumber", "in", list(faults)))
    cols = columns if columns is not None else config.META_COLS + config.FEATURE_COLS
    return pd.read_parquet(path, columns=cols, filters=filters)


def _pick_runs(n: int, seed: int, exclude: set[int] | None = None) -> list[int]:
    """Wählt reproduzierbar n Lauf-IDs aus 1..N_RUNS_TOTAL (optional unter Ausschluss)."""
    rng = np.random.default_rng(seed)
    pool = [r for r in range(1, config.N_RUNS_TOTAL + 1) if not exclude or r not in exclude]
    if n > len(pool):
        raise ValueError(f"{n} Läufe angefragt, aber nur {len(pool)} verfügbar.")
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def make_labels(df: pd.DataFrame, onset: int) -> np.ndarray:
    """Punktweises Anomalie-Label (1=anomal) unter Berücksichtigung des Fehler-Onsets."""
    return ((df["faultNumber"] != 0) & (df["sample"] > onset)).to_numpy().astype(int)


# --------------------------------------------------------------------------------------
# Split-Container
# --------------------------------------------------------------------------------------
@dataclass
class Split:
    """Ein fertig vorbereiteter, skalierter Train/Test-Split für unüberwachte AD."""

    X_train_good: np.ndarray          # nur Gutdaten, skaliert
    X_test: np.ndarray                # gemischt (normal + Fehler), skaliert
    y_test: np.ndarray                # punktweises Label (1=anomal, Onset-korrekt)
    meta_test: pd.DataFrame           # faultNumber/simulationRun/sample (für Detection-Delay etc.)
    meta_train: pd.DataFrame          # dito für die Trainings-Gutdaten
    scaler: StandardScaler
    faults: list[int]                 # im Test enthaltene Fehlertypen
    train_runs: list[int]             # Gutdaten-Läufe im Training (für disjunkte Validierung)
    test_good_runs: list[int]         # Gutdaten-Läufe im Test (disjunkt zu Validierung)
    test_fault_runs: list[int]        # Fehler-Läufe im Test (disjunkt zu Validierung)

    def __repr__(self) -> str:  # kompakte Übersicht
        return (
            f"Split(train_good={self.X_train_good.shape}, test={self.X_test.shape}, "
            f"anomaly_rate={self.y_test.mean():.3f})"
        )


def load_split(
    faults: list[int] | None = None,
    n_train_good_runs: int = 25,
    n_test_good_runs: int = 20,
    n_test_fault_runs: int = 20,
    seed: int = config.RANDOM_SEED,
) -> Split:
    """Baut einen unüberwachten-AD-Split aus den TEP-Parquet-Dateien.

    - Training: Gutdaten aus ``fault_free_training`` (``n_train_good_runs`` Läufe).
    - Test: Gutdaten aus ``fault_free_testing`` (``n_test_good_runs`` Läufe) + Fehlerdaten aus
      ``faulty_testing`` (``n_test_fault_runs`` Läufe je Fehlertyp in ``faults``).
    - Scaler wird nur auf den Trainings-Gutdaten gefittet.

    Train- und Test-Gutdaten stammen aus unterschiedlichen Dateien (verschiedene Simulationen)
    → kein Leakage trotz evtl. gleicher Lauf-Nummern.
    """
    faults = faults if faults is not None else config.DEMO_FAULTS

    train_runs = _pick_runs(n_train_good_runs, seed=seed)
    test_good_runs = _pick_runs(n_test_good_runs, seed=seed + 1)
    test_fault_runs = _pick_runs(n_test_fault_runs, seed=seed + 2)

    df_train = _read_runs(config.FAULT_FREE_TRAINING, train_runs)

    df_test_good = _read_runs(config.FAULT_FREE_TESTING, test_good_runs)
    df_test_fault = _read_runs(config.FAULTY_TESTING, test_fault_runs, faults=faults)
    df_test = pd.concat([df_test_good, df_test_fault], ignore_index=True)

    # Skalierung (nur auf Gutdaten gefittet).
    scaler = StandardScaler().fit(df_train[config.FEATURE_COLS].to_numpy())
    X_train_good = scaler.transform(df_train[config.FEATURE_COLS].to_numpy())
    X_test = scaler.transform(df_test[config.FEATURE_COLS].to_numpy())

    y_test = make_labels(df_test, onset=config.ONSET_TESTING)
    meta_test = df_test[config.META_COLS].reset_index(drop=True)
    meta_train = df_train[config.META_COLS].reset_index(drop=True)

    return Split(
        X_train_good, X_test, y_test, meta_test, meta_train, scaler,
        faults, train_runs, test_good_runs, test_fault_runs,
    )


def load_validation(
    split: Split,
    n_good_runs: int = 10,
    n_fault_runs: int = 10,
    seed: int = 99,
    source: str = "testing",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Gelabeltes Validierungsset für Oracle-Auswahl/HPO — **disjunkt vom Test**.

    Im echten unüberwachten Betrieb gäbe es diese Labels nicht; das Validierungsset dient hier
    nur dazu, die **Obergrenze** (``select_oracle``) und HPO mit Zielmetrik zu demonstrieren.

    - ``source="testing"`` (Default): aus den Testdateien (gleiche Verteilung wie der Test),
      aber mit Läufen, die **disjunkt** zu den Test-Läufen sind, Onset 160.
    - ``source="training"``: aus den **Trainings**-Dateien (``faulty_training``, Onset 20 +
      held-out ``fault_free_training``). Alternative, wenn die Testdateien unberührt bleiben sollen.

    Skaliert mit dem Scaler aus ``split``.
    """
    if source == "testing":
        good_runs = _pick_runs(n_good_runs, seed=seed, exclude=set(split.test_good_runs))
        fault_runs = _pick_runs(n_fault_runs, seed=seed + 1, exclude=set(split.test_fault_runs))
        df_good = _read_runs(config.FAULT_FREE_TESTING, good_runs)
        df_fault = _read_runs(config.FAULTY_TESTING, fault_runs, faults=split.faults)
        onset = config.ONSET_TESTING
    elif source == "training":
        good_runs = _pick_runs(n_good_runs, seed=seed, exclude=set(split.train_runs))
        fault_runs = _pick_runs(n_fault_runs, seed=seed + 1)
        df_good = _read_runs(config.FAULT_FREE_TRAINING, good_runs)
        df_fault = _read_runs(config.FAULTY_TRAINING, fault_runs, faults=split.faults)
        onset = config.ONSET_TRAINING
    else:
        raise ValueError("source muss 'testing' oder 'training' sein.")

    df_val = pd.concat([df_good, df_fault], ignore_index=True)
    X_val = split.scaler.transform(df_val[config.FEATURE_COLS].to_numpy())
    y_val = make_labels(df_val, onset=onset)
    meta_val = df_val[config.META_COLS].reset_index(drop=True)
    return X_val, y_val, meta_val
