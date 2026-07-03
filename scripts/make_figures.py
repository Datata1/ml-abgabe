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
from automl_ad.ts import windowed_candidate_scores  # noqa: E402

WINDOW, STEP, AGG = config.WINDOW, config.STEP, config.AGGREGATION


def main() -> None:
    figures.use_slide_style()
    split = load_split(**config.SPLIT_KW)

    def auc(s):
        return float(roc_auc_score(split.y_test, s))

    # --- Rechnen ---
    iid_auc = {n: auc(s) for n, s in fit_scores(DEFAULT_CANDIDATES, split.X_train_good, split.X_test).items()}
    tw = windowed_candidate_scores(DEFAULT_CANDIDATES, split, window_size=WINDOW, step=STEP, aggregation=AGG)
    tw_auc = {n: auc(s) for n, s in tw.items()}
    best_a, centrality = consensus_centrality(tw)
    modeB_auc = {m: auc(ensemble_consensus(tw, m)) for m in ("average", "maximization", "median")}
    breakdown = per_fault_breakdown(
        tw, ensemble_consensus(tw, "average"), split.y_test, split.meta_test["faultNumber"].to_numpy()
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
        "naiv (i.i.d.)": (min(iid_auc, key=iid_auc.get), min(iid_auc.values())),
        "Konsens A": (best_a, tw_auc[best_a]),
        "Konsens B (avg)": ("ensemble", modeB_auc["average"]),
    }
    _ev = engine_out.get("validation") or {}
    if _ev.get("consensus_roc_auc") is not None:
        strategies["ADEngine"] = ("Consensus", float(_ev["consensus_roc_auc"]))
    if llm_cache and llm_cache.get("roc_auc") is not None:
        strategies["ADEngine\n+ LLM"] = (llm_cache.get("detector"), float(llm_cache["roc_auc"]))

    # --- Grafiken schreiben ---
    figures.fig_detector_spread(iid_auc, save_as="01_detektor_streuung.png")
    figures.fig_windowing_mechanism(split, tw["pca"], fault=fault, window=WINDOW, step=STEP,
                                    save_as="02_fensterung.png")
    figures.fig_iid_vs_windowed(iid_auc, tw_auc, save_as="03_iid_vs_zeitbewusst.png")
    figures.fig_score_timeline(split, tw["pca"], fault=fault, onset=config.ONSET_TESTING,
                               save_as="04_score_zeitreihe.png")
    figures.fig_consensus(tw, centrality, best_a, agreement(tw), save_as="05_konsens_modusA.png")
    figures.fig_ensemble(modeB_auc, max(tw_auc.values()), save_as="06_konsens_modusB.png")
    figures.fig_per_fault(breakdown, save_as="11_pro_fehler.png")
    figures.fig_benchmark(top5, save_as="07_benchmark.png")
    figures.fig_engine_report(engine_out, save_as="08_adengine_report.png")
    figures.fig_llm_routing(llm_cache, save_as="09_llm_routing.png")
    figures.fig_final_comparison(strategies, max(tw_auc.values()), save_as="10_fazit_vergleich.png")

    print(f"11 Grafiken geschrieben nach {config.REPORTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
