"""Erzeugt alle slide-fertigen Grafiken nach ``reports/figures/`` (für die PowerPoint-Folien).

Rechnet einmal die ganze Pipeline und ruft die ``fig_*``-Funktionen aus ``automl_ad.figures``.
Headless (Agg-Backend), keine Anzeige nötig.

Start:  uv run python scripts/make_figures.py   (oder: make figures)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - vor pyplot-Import (via figures) festlegen

from sklearn.metrics import roc_auc_score  # noqa: E402

from automl_ad import config, figures  # noqa: E402
from automl_ad.data import load_split  # noqa: E402
from automl_ad.pyod_engine import benchmark_ranking, run_engine  # noqa: E402
from automl_ad.selection import (  # noqa: E402
    DEFAULT_CANDIDATES,
    agreement,
    consensus_centrality,
    ensemble_consensus,
    fit_scores,
    per_fault_breakdown,
)

NAIVE = "iforest"  # der typische "Standard-Griff" ohne Labels


def main() -> None:
    figures.use_slide_style()
    split = load_split(**config.SPLIT_KW)

    def auc(s):
        return float(roc_auc_score(split.y_test, s))

    # --- Rechnen ---
    scores = fit_scores(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    det_auc = {n: auc(s) for n, s in scores.items()}
    best_a, centrality = consensus_centrality(scores)
    modeB_auc = {m: auc(ensemble_consensus(scores, m)) for m in ("average", "maximization", "median")}
    breakdown = per_fault_breakdown(
        scores, ensemble_consensus(scores, "average"), split.y_test,
        split.meta_test["faultNumber"].to_numpy()
    )

    import numpy as np
    _rng = np.random.default_rng(0)
    _idx = np.sort(_rng.choice(len(split.X_test), size=min(3000, len(split.X_test)), replace=False))
    engine_out = run_engine(split.X_test[_idx], split.y_test[_idx], data_type="tabular")

    top5 = [(n, int(rk.get("ADBench_overall"))) for n, rk in benchmark_ranking("tabular", top_k=5)
            if rk.get("ADBench_overall") is not None]

    cache_path = config.REPORTS_DIR / "llm_cache.json"
    llm_cache = json.loads(cache_path.read_text()) if cache_path.exists() else None

    fault = split.faults[0]
    strategies = {
        "naiv (fix iforest)": (NAIVE, det_auc[NAIVE]),
        "Konsens A": (best_a, det_auc[best_a]),
        "Konsens B (avg)": ("ensemble", modeB_auc["average"]),
    }
    _ev = engine_out.get("validation") or {}
    if _ev.get("consensus_roc_auc") is not None:
        strategies["ADEngine"] = ("Consensus", float(_ev["consensus_roc_auc"]))
    if llm_cache and llm_cache.get("roc_auc") is not None:
        strategies["ADEngine\n+ LLM"] = (llm_cache.get("detector"), float(llm_cache["roc_auc"]))

    # --- Grafiken schreiben ---
    # Modell-Steckbriefe (je Baseline-Detektor eine Karte — für die Vorstellungs-Folie,
    # frei kombinierbar; bewusst ohne Nummer im Namen).
    for det_name in ("knn", "pca", "hdbscan", "iforest"):
        figures.fig_model_card(det_name, save_as=f"modell_{det_name}.png")

    figures.fig_detector_spread(det_auc, save_as="01_detektor_streuung.png")
    figures.fig_score_timeline(split, scores["pca"], fault=fault, onset=config.ONSET_TESTING,
                               save_as="02_score_zeitreihe.png")
    figures.fig_consensus(scores, centrality, best_a, agreement(scores), save_as="03_konsens_modusA.png")
    figures.fig_ensemble(modeB_auc, max(det_auc.values()), save_as="04_konsens_modusB.png")
    figures.fig_per_fault(breakdown, save_as="05_pro_fehler.png")
    figures.fig_benchmark(top5, save_as="06_benchmark.png")
    figures.fig_engine_report(engine_out, save_as="07_adengine_report.png")
    figures.fig_llm_routing(llm_cache, save_as="08_llm_routing.png")
    figures.fig_final_comparison(strategies, save_as="09_fazit_vergleich.png")

    print(f"13 Grafiken geschrieben nach {config.REPORTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
