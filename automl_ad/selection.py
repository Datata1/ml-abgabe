"""Label-freie Auswahl & Kombination von Detektoren — auf Basis fertiger Score-Dicts.

Roter Faden: *Wie wählt/kombiniert man ohne Labels?* Kernidee: nicht ein einzelnes Modell ist die
Wahrheit, aber der **Konsens** vieler Modelle ist robust.

Die Kernfunktionen arbeiten auf einem ``scores``-Dict ``{detektorname: score_vektor}`` und sind damit
unabhängig davon, **wie** die Scores entstanden sind (Producer: :func:`fit_scores`):

- :func:`consensus_centrality` — **Modus A**: wähle das zum Konsens zentralste **eine** Modell.
- :func:`ensemble_consensus`  — **Modus B**: der Konsens-Score eines **Ensembles** ist die Vorhersage.
- :func:`agreement`           — label-freies Vertrauensmaß (mittlere paarweise Spearman-Korrelation).
- :func:`per_fault_breakdown` — differenzierte Auswertung pro Fehlertyp (AUC + agreement je Fehler).
- :func:`oracle_best`         — label-**basierte** Obergrenze (nur Referenz; real nicht verfügbar).

Die dünnen Wrapper :func:`select_internal` / :func:`select_oracle` bündeln „fitten + auswählen"
für den naiven Startpunkt.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from .detectors import make_detector

# Kandidat = (Detektorname, Hyperparameter-Dict)
Candidate = tuple[str, dict]
Scores = dict[str, np.ndarray]

DEFAULT_CANDIDATES: list[Candidate] = [
    ("knn", {}),
    ("pca", {}),
    ("hdbscan", {}),
    ("iforest", {}),
]

# Ensemble-Kombinatoren (identisch zu pyod.models.combination, aber ohne die optionale
# ``combo``-Abhängigkeit): über z-normierte Detektor-Scores aggregieren.
_COMBINERS = {
    "average": lambda m: m.mean(axis=1),
    "maximization": lambda m: m.max(axis=1),
    "median": lambda m: np.median(m, axis=1),
}


def fit_scores(candidates: list[Candidate], X_train, X_eval) -> Scores:
    """Score-Producer: fittet alle Kandidaten auf Gutdaten, gibt ihre Eval-Scores zurück."""
    return {
        name: make_detector(name, **hp).fit(X_train).decision_function(X_eval)
        for name, hp in candidates
    }


def _rank_matrix(scores: Scores) -> tuple[list[str], np.ndarray]:
    """Namen + Matrix der Ränge (Spalte = Detektor). Ränge sind über Detektoren vergleichbar."""
    names = list(scores)
    ranks = np.column_stack([rankdata(scores[n]) for n in names])
    return names, ranks


def consensus_centrality(scores: Scores) -> tuple[str, dict[str, float]]:
    """**Modus A** — wähle den Detektor, dessen Ranking am stärksten mit dem Konsens korreliert.

    Rang-Konsens = Mittel aller Rang-Vektoren; Centrality = Korrelation jedes Detektors zum Konsens.
    Rückgabe ``(bester_name, {name: centrality})``. Benötigt keine Labels.
    """
    names, ranks = _rank_matrix(scores)
    consensus = ranks.mean(axis=1)
    centrality = {n: float(np.corrcoef(ranks[:, i], consensus)[0, 1]) for i, n in enumerate(names)}
    return max(centrality, key=centrality.get), centrality


def ensemble_consensus(scores: Scores, method: str = "average") -> np.ndarray:
    """**Modus B** — kombiniere alle Detektoren zu **einem** Score-Vektor (die Vorhersage).

    Scores werden je Detektor z-normiert (vergleichbare Skala), dann kombiniert
    (``average`` | ``maximization`` | ``median`` — wie ``pyod.models.combination``).
    """
    if method not in _COMBINERS:
        raise ValueError(f"method muss aus {sorted(_COMBINERS)} sein, war '{method}'.")
    matrix = np.column_stack([np.asarray(scores[n], dtype=float) for n in scores])
    mu = matrix.mean(axis=0)
    sigma = np.where(matrix.std(axis=0) > 0, matrix.std(axis=0), 1.0)
    return _COMBINERS[method]((matrix - mu) / sigma)


def agreement(scores: Scores) -> float:
    """Label-freies Vertrauensmaß: mittlere paarweise Spearman-Korrelation der Detektor-Rankings.

    Hoch = der „Schwarm" ist sich einig (verlässlicher). Spiegelt das ``agreement`` der ADEngine.
    """
    _, ranks = _rank_matrix(scores)
    corr = np.corrcoef(ranks, rowvar=False)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return float(np.mean(off_diag))


def per_fault_breakdown(scores: Scores, prediction, y, fault_numbers) -> dict[int, dict[str, float]]:
    """Differenzierte Auswertung **pro Fehlertyp**: Gutdaten + jeweils *ein* Fehlertyp.

    Für jeden Fehlertyp ``f`` wird der Test auf (Gutdaten ∪ Fehler ``f``) eingeschränkt und
    berechnet:

    - ``roc_auc`` der übergebenen Vorhersage (label-basiert → nur zur Illustration),
    - ``agreement`` des Detektor-Schwarms auf demselben Ausschnitt (label-frei).

    So sieht man, welche Fehlerarten die Erkennung trägt — und ob das label-freie Signal die
    schweren Fälle überhaupt anzeigt (Einigkeit ≠ Richtigkeit).
    """
    prediction = np.asarray(prediction, dtype=float)
    y = np.asarray(y)
    fault_numbers = np.asarray(fault_numbers)
    good = fault_numbers == 0
    out: dict[int, dict[str, float]] = {}
    for f in sorted({int(v) for v in np.unique(fault_numbers)} - {0}):
        m = good | (fault_numbers == f)
        out[f] = {
            "roc_auc": float(roc_auc_score(y[m], prediction[m])),
            "agreement": agreement({n: np.asarray(s)[m] for n, s in scores.items()}),
        }
    return out


def oracle_best(scores: Scores, y) -> tuple[str, dict[str, float]]:
    """Label-**basierte** Obergrenze (Referenz): bester ROC-AUC auf gelabeltem Set.

    Nur zum Vergleich — im echten unüberwachten Betrieb nicht verfügbar.
    """
    aucs = {name: float(roc_auc_score(y, s)) for name, s in scores.items()}
    return max(aucs, key=aucs.get), aucs


# --------------------------------------------------------------------------------------
# Dünne Wrapper (naiver Startpunkt: fitten + auswählen in einem Aufruf)
# --------------------------------------------------------------------------------------
def select_internal(candidates, X_train, X_eval) -> tuple[str, dict[str, float]]:
    """Bequemlichkeit: ``consensus_centrality(fit_scores(...))`` (Modus A)."""
    return consensus_centrality(fit_scores(candidates, X_train, X_eval))


def select_oracle(candidates, X_train, X_val, y_val) -> tuple[str, dict[str, float]]:
    """Bequemlichkeit: ``oracle_best(fit_scores(...), y_val)`` (label-Obergrenze)."""
    return oracle_best(fit_scores(candidates, X_train, X_val), y_val)
