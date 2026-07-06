"""Reproduzierbare Zahlen (ALLE 20 Fehlertypen): Baselines, Konsens A/B, Oracle
→ ``reports/results.csv`` + differenzierte Pro-Fehler-Auswertung → ``reports/results_per_fault.csv``.

Für die **Grafiken** (Folien) siehe ``scripts/make_figures.py`` / ``make figures``. Alle ROC/AUC sind
**nur zur Illustration** (im echten unüberwachten Betrieb nicht verfügbar).

Start:  uv run python scripts/run_experiment.py
"""

from __future__ import annotations

import csv

from sklearn.metrics import roc_auc_score

from automl_ad import config
from automl_ad.data import load_split
from automl_ad.selection import (
    DEFAULT_CANDIDATES,
    agreement,
    consensus_centrality,
    ensemble_consensus,
    fit_scores,
    oracle_best,
    per_fault_breakdown,
)

NAIVE = "iforest"  # der typische "Standard-Griff" ohne Labels


def main() -> None:
    split = load_split(**config.SPLIT_KW)
    y = split.y_test

    def auc(scores):
        return float(roc_auc_score(y, scores))

    # Detektor-Scores (einmal), darauf alle label-freien Strategien.
    scores = fit_scores(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    best_a, _ = consensus_centrality(scores)
    ens = ensemble_consensus(scores, "average")
    oracle_pick, oracle_aucs = oracle_best(scores, y)

    strategies = {
        "naiv (fix iforest)": (NAIVE, auc(scores[NAIVE])),  # blinde Einzelwahl
        "Konsens A": (best_a, auc(scores[best_a])),         # zentralstes Modell (label-frei)
        "Konsens B (avg)": ("ensemble", auc(ens)),          # Ensemble-Konsens als Vorhersage
        "Oracle (Referenz)": (oracle_pick, oracle_aucs[oracle_pick]),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.REPORTS_DIR / "results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["strategie", "gewaehlter_detektor", "roc_auc_illustrativ"])
        for label, (pick, val) in strategies.items():
            w.writerow([label, pick, f"{val:.4f}"])

    # Differenzierte Auswertung: wie verändert sich das Ergebnis mit der Fehlerart?
    fault_no = split.meta_test["faultNumber"].to_numpy()
    bd_b = per_fault_breakdown(scores, ens, y, fault_no)
    bd_a = per_fault_breakdown(scores, scores[best_a], y, fault_no)
    with (config.REPORTS_DIR / "results_per_fault.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fault", "schwer", "roc_auc_konsens_a", "roc_auc_konsens_b", "agreement"])
        for f in sorted(bd_b):
            w.writerow([
                f, int(f in config.HARD_FAULTS),
                f"{bd_a[f]['roc_auc']:.4f}", f"{bd_b[f]['roc_auc']:.4f}",
                f"{bd_b[f]['agreement']:.4f}",
            ])

    print(f"label-freies agreement (Schwarm-Einigkeit): {agreement(scores):.3f}")
    for label, (pick, val) in strategies.items():
        print(f"  {label:20s} -> {pick:10s} ROC-AUC(illustr.)={val:.3f}")
    hard = {f: bd_b[f]["roc_auc"] for f in config.HARD_FAULTS if f in bd_b}
    print(f"\nschwere Fehler (Konsens B): " + ", ".join(f"F{f}={v:.3f}" for f, v in hard.items()))
    print(f"geschrieben: {config.REPORTS_DIR/'results.csv'} + results_per_fault.csv  (Grafiken: make figures)")


if __name__ == "__main__":
    main()
