"""Hyperparameter-Optimierung mit Optuna — das klassische AutoML-Werkzeug.

Demonstriert Bayesian Optimization (TPE-Sampler) auf den AD-Detektoren. Die Zielmetrik wird auf
einem **gelabelten Validierungsset** ausgewertet (Oracle-HPO). Im echten unüberwachten Betrieb
wäre dafür eine label-freie Metrik nötig — genau deshalb dreht sich der rote Faden um die
**Auswahl ohne Labels** (siehe ``automl_ad/selection.py`` und ``automl_ad/llm.py``).
"""

from __future__ import annotations

import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .detectors import make_detector

optuna.logging.set_verbosity(optuna.logging.WARNING)

_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    """Suchraum je Detektor."""
    if name == "iforest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_samples": trial.suggest_int("max_samples", 128, 1024),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    if name == "ocsvm":
        return {
            "nu": trial.suggest_float("nu", 0.01, 0.5, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        }
    if name == "pca":
        return {"n_components": trial.suggest_int("n_components", 1, len(config.FEATURE_COLS) - 1)}
    if name == "ecod":
        return {}  # parameterfrei
    raise KeyError(f"Kein Suchraum für '{name}' definiert.")


def run_optuna(
    name: str,
    X_train_good,
    X_val,
    y_val,
    n_trials: int = 30,
    metric: str = "roc_auc",
    seed: int = config.RANDOM_SEED,
) -> tuple[dict, optuna.Study]:
    """Tunt einen Detektor; Ziel = ``metric`` auf dem gelabelten Validierungsset.

    Gibt die besten Hyperparameter und die Optuna-Study (für Verlaufs-Plots) zurück.
    """
    score_fn = _METRICS[metric]

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_params(trial, name)
        det = make_detector(name, **hp).fit(X_train_good)
        return float(score_fn(y_val, det.decision_function(X_val)))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study
