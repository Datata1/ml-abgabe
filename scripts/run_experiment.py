"""End-to-End-Experiment: Modellauswahl ohne Labels auf dem Tennessee-Eastman-Prozess.

Vergleicht die drei label-freien Auswahlstrategien gegen die Oracle-Obergrenze:

  naiv (fixer Default)  ·  statistisch (Konsens)  ·  pyOD ADEngine  ·  Oracle (Labels)

Stufe 3 ist die echte, library-native AutoML-AD (``pyod.utils.ad_engine.ADEngine``):
benchmark-gestützte Detektorwahl + Consensus + label-freie Qualitätsdiagnostik. Unsere
selbstgebaute LLM-Variante (``automl_ad/llm.py``) wird nur noch als „unter der Haube"-Randnotiz
gezeigt.

Ausgabe: ``reports/results.csv`` + ``reports/auswahl_vergleich.png``.

Aufruf:  uv run python scripts/run_experiment.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from automl_ad import config
from automl_ad.data import load_split, load_validation
from automl_ad.detectors import make_detector
from automl_ad.plots import selection_comparison
from automl_ad.pyod_engine import benchmark_ranking, run_engine, run_engine_llm_routed
from automl_ad.selection import (
    DEFAULT_CANDIDATES,
    select_internal,
    select_llm,
    select_oracle,
)

NAIVE_DETECTOR = "ecod"  # willkürlich gewählter, fixer Default (Stufe 1)
ENGINE_MAX_ROWS = 8000   # ADEngine-Subsample (Tempo/Speicher; z. B. KNN skaliert quadratisch)


def _data_available() -> bool:
    return all(
        p.exists()
        for p in (
            config.FAULT_FREE_TRAINING,
            config.FAULT_FREE_TESTING,
            config.FAULTY_TESTING,
        )
    )


def main() -> int:
    if not _data_available():
        print(
            "TEP-Parquet-Dateien fehlen unter data/.\n"
            "Bitte die vier Dateien (TEP_FaultFree_Training/Testing, TEP_Faulty_Training/Testing)\n"
            "wie in der README beschrieben nach data/ legen.",
            file=sys.stderr,
        )
        return 1

    print("Lade TEP-Split ...")
    split = load_split()

    # Pro Kandidat: einmal auf Gutdaten fitten, Test-Scores -> Test-ROC-AUC (gemeinsame Basis).
    print("Berechne Test-ROC-AUC je Detektor ...")
    test_auc: dict[str, float] = {}
    for name, hp in DEFAULT_CANDIDATES:
        det = make_detector(name, **hp).fit(split.X_train_good)
        test_auc[name] = float(roc_auc_score(split.y_test, det.decision_function(split.X_test)))
    for name, auc in sorted(test_auc.items(), key=lambda kv: kv[1]):
        print(f"  {name:8s} ROC-AUC={auc:.3f}")

    strategies: dict[str, tuple[str, float]] = {}
    reasons: dict[str, str] = {}

    # Stufe 1 — naiv: fixer Default.
    strategies["naiv\n(fixer Default)"] = (NAIVE_DETECTOR, test_auc[NAIVE_DETECTOR])

    # Stufe 2 — statistisch: Konsens (label-frei, auf den Testdaten).
    pick_internal, _ = select_internal(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    strategies["statistisch\n(Konsens)"] = (pick_internal, test_auc[pick_internal])

    # Stufe 3 — pyOD ADEngine (echte, library-native AutoML-AD; label-frei).
    # ADEngine fittet+bewertet im Consensus-Workflow auf demselben X; daher auf einem
    # Subsample der Testdaten. Die Bewertung macht ADEngine selbst (validate()).
    print("\nBenchmark-Rangliste (pyOD-Wissensbasis, ADBench):")
    for name, rank in benchmark_ranking():
        print(f"  {name:8s} rank={rank}")
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = np.sort(rng.choice(len(split.X_test), size=min(ENGINE_MAX_ROWS, len(split.X_test)),
                             replace=False))
    X_sub, y_sub = split.X_test[idx], split.y_test[idx]
    print(f"\nStufe 3 — pyOD ADEngine (Subsample {len(idx)} Punkte) ...")
    out = run_engine(X_sub, y_sub)
    val = out["validation"]
    engine_auc = val["consensus_roc_auc"]
    engine_pick = out["best_detector"] or "Consensus"
    strategies["pyOD ADEngine\n(Consensus)"] = (str(engine_pick), engine_auc)
    reasons[str(engine_pick)] = (
        f"benchmark-gestützt: {', '.join(out['detectors'])}; "
        f"label-freie Qualität {out['quality_verdict']} ({out['quality_overall']:.2f}); "
        f"consensus_helped={val['consensus_helped']}"
    )
    print(f"  gewählte Detektoren: {', '.join(out['detectors'])}")
    print(f"  bester laut Consensus: {engine_pick} | Übereinstimmung: {out['agreement']:.2f}")
    print(f"  label-freie Qualität: {out['quality_verdict']} ({out['quality_overall']:.2f})")
    print(f"  ADEngine-Validierung: Consensus-ROC-AUC={engine_auc:.3f}, "
          f"bester Detektor={val['best_detector_roc_auc']}, consensus_helped={val['consensus_helped']}")

    # Bonus — Brücke „unter der Haube" → „echt": UNSER LLM steuert pyODs Detektorwahl
    # (plan_detection(llm_client=...)). LLM wählt aus pyODs 61-Detektoren-Wissensbasis.
    try:
        routed = run_engine_llm_routed(split.X_train_good, X_sub)
        routed_auc = float(roc_auc_score(y_sub, routed["scores_test"]))
        strategies["ADEngine + LLM\n(LLM-Routing)"] = (str(routed["detector"]), routed_auc)
        reasons[str(routed["detector"])] = (
            f"LLM-geroutet (pyOD-KB): + {', '.join(routed['alternatives'])}; {routed['reason']}"
        )
        print(f"\nBonus — ADEngine mit LLM-Routing: {routed['detector']} "
              f"(+ {', '.join(routed['alternatives'])}) | ROC-AUC={routed_auc:.3f}")
        print(f"  LLM-Begründung: {routed['reason']}")
    except Exception as exc:  # noqa: BLE001 - optionaler Bonus: nie den Gesamtlauf abbrechen
        print(f"\nBonus (LLM-Routing) übersprungen: {exc}", file=sys.stderr)

    # Randnotiz „unter der Haube": unsere vereinfachte LLM-Eigenbauvariante (4 Detektoren).
    try:
        pick_llm, info = select_llm(DEFAULT_CANDIDATES, split.X_train_good, X_sub)
        print(f"\n[unter der Haube] LLM-Eigenbau (4 Detektoren) wählt: {pick_llm}"
              f"{'  [Fallback]' if info['fallback'] else ''} — {info['reason']}")
    except RuntimeError as exc:
        print(f"\n[unter der Haube] LLM-Eigenbau übersprungen (kein Provider): {exc}",
              file=sys.stderr)

    # Obergrenze — Oracle: Auswahl auf gelabeltem Validierungsset, Bewertung auf Test.
    X_val, y_val, _ = load_validation(split)
    pick_oracle, _ = select_oracle(DEFAULT_CANDIDATES, split.X_train_good, X_val, y_val)
    oracle_auc = test_auc[pick_oracle]
    print(f"\nOracle-Auswahl (mit Labels): {pick_oracle}  ROC-AUC={oracle_auc:.3f}")

    # --- Ausgabe: Tabelle + Plot ---------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "strategie": label.replace("\n", " "),
            "gewaehlter_detektor": pick,
            "roc_auc": auc,
            "begruendung": reasons.get(pick, ""),
        }
        for label, (pick, auc) in strategies.items()
    ]
    rows.append(
        {"strategie": "oracle (Obergrenze)", "gewaehlter_detektor": pick_oracle,
         "roc_auc": oracle_auc, "begruendung": "Auswahl mit Labels"}
    )
    df = pd.DataFrame(rows)
    csv_path = config.REPORTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)

    selection_comparison(strategies, oracle_auc=oracle_auc, save_as="auswahl_vergleich.png")

    print(f"\nGeschrieben:\n  {csv_path}\n  {config.REPORTS_DIR / 'auswahl_vergleich.png'}")
    print("\n" + df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
