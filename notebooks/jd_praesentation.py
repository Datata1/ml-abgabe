"""Interaktive Präsentation (marimo): Outlier Detection OHNE Labels — zeitbewusst.

**Grafik-zentriert:** jede Zelle = knapper Kontext + eine slide-fertige Grafik (aus ``automl_ad.figures``).
Dieselben Grafiken schreibt ``make figures`` als PNGs nach ``reports/figures/`` — direkt für die Folien.

Start:  uv run marimo edit notebooks/jd_praesentation.py   ·   Präsentieren: make present
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Outlier Detection **ohne Labels** — zeitbewusst
        ### Tennessee-Eastman-Prozess (52 Sensoren, Zeitreihe)

        Jede Zelle liefert **eine slide-fertige Grafik** (auch als PNG via `make figures`).
        ROC/AUC erscheint nur als markierter „Wenn wir spicken würden"-Block — real hätte man es nicht.
        """
    )
    return (mo,)


@app.cell
def _():
    # --- Setup: Stil + Daten + gemeinsame Scores (einmal) ---
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from automl_ad import config, figures
    from automl_ad.data import load_split
    from automl_ad.selection import (
        DEFAULT_CANDIDATES,
        agreement,
        consensus_centrality,
        ensemble_consensus,
        fit_scores,
        per_fault_breakdown,
    )
    from automl_ad.ts import windowed_candidate_scores

    figures.use_slide_style()
    # EIN Setup für alle Zahlen (config.SPLIT_KW): alle 20 Fehlertypen — auch die schweren,
    # sonst wird die Auswertung zu optimistisch.
    split = load_split(**config.SPLIT_KW)
    WINDOW, STEP, AGG = config.WINDOW, config.STEP, config.AGGREGATION

    def _auc(s):
        return float(roc_auc_score(split.y_test, s))

    iid_auc = {n: _auc(s) for n, s in fit_scores(DEFAULT_CANDIDATES, split.X_train_good, split.X_test).items()}
    tw = windowed_candidate_scores(DEFAULT_CANDIDATES, split, window_size=WINDOW, step=STEP, aggregation=AGG)
    tw_auc = {n: _auc(s) for n, s in tw.items()}
    best_a, centrality = consensus_centrality(tw)
    modeB_auc = {m: _auc(ensemble_consensus(tw, m)) for m in ("average", "maximization", "median")}
    breakdown = per_fault_breakdown(
        tw, ensemble_consensus(tw, "average"), split.y_test, split.meta_test["faultNumber"].to_numpy()
    )
    return (STEP, WINDOW, agreement, best_a, breakdown, centrality, config, figures, iid_auc,
            modeB_auc, np, split, tw, tw_auc)


@app.cell
def _(figures, iid_auc, mo):
    mo.vstack([
        mo.md("## 1 — Das Problem: ohne Labels ist die Detektorwahl ein Blindflug"),
        figures.fig_detector_spread(iid_auc),
    ])
    return


@app.cell
def _(WINDOW, STEP, figures, mo, split, tw):
    mo.vstack([
        mo.md("## Mechanismus: wie ein Detektor **zeitbewusst** wird (Fenster-Adapter)"),
        figures.fig_windowing_mechanism(split, tw["pca"], fault=split.faults[0], window=WINDOW, step=STEP),
    ])
    return


@app.cell
def _(figures, iid_auc, mo, tw_auc):
    mo.vstack([
        mo.md("## Der Effekt: zeitbewusst schlägt den i.i.d.-Blick"),
        figures.fig_iid_vs_windowed(iid_auc, tw_auc),
    ])
    return


@app.cell
def _(config, figures, mo, split, tw):
    mo.vstack([
        mo.md("## Anschauung: Score über die Zeit eines Fehlerlaufs"),
        figures.fig_score_timeline(split, tw["pca"], fault=split.faults[0], onset=config.ONSET_TESTING),
    ])
    return


@app.cell
def _(agreement, best_a, centrality, figures, mo, tw):
    mo.vstack([
        mo.md("## 2 — Konsens · Modus A: das zentralste Modell (label-frei)"),
        figures.fig_consensus(tw, centrality, best_a, agreement(tw)),
    ])
    return


@app.cell
def _(figures, mo, modeB_auc, tw_auc):
    mo.vstack([
        mo.md("## Konsens · Modus B: das Ensemble ist die Vorhersage"),
        figures.fig_ensemble(modeB_auc, max(tw_auc.values())),
    ])
    return


@app.cell
def _(breakdown, figures, mo):
    mo.vstack([
        mo.md("## Differenzierung: wie verändert sich das Ergebnis mit der Fehlerart?"),
        figures.fig_per_fault(breakdown),
    ])
    return


@app.cell
def _(figures, mo):
    # ADEngine-Benchmark: vorberechnete Bestenliste (ADBench).
    from automl_ad.pyod_engine import benchmark_ranking

    top5 = [(n, int(rk.get("ADBench_overall"))) for n, rk in benchmark_ranking("tabular", top_k=5)
            if rk.get("ADBench_overall") is not None]
    mo.vstack([
        mo.md("## 3 — ADEngine: die Benchmark ist eine **vorberechnete** Bestenliste"),
        figures.fig_benchmark(top5),
    ])
    return


@app.cell
def _(figures, mo, np, split):
    # ADEngine tabular (sub-Sekunde, fair). time_series wäre langsam + schwach → nur erwähnt.
    from automl_ad.pyod_engine import run_engine

    _idx = np.sort(np.random.default_rng(0).choice(len(split.X_test), size=min(3000, len(split.X_test)), replace=False))
    engine_out = run_engine(split.X_test[_idx], split.y_test[_idx], data_type="tabular")
    mo.vstack([
        mo.md("## ADEngine: ein Aufruf → Auswahl + label-freies Verdikt"),
        figures.fig_engine_report(engine_out),
    ])
    return (engine_out,)


@app.cell
def _(config, figures, mo):
    # Natives LLM-Routing aus Cache (make cache). Ohne Cache: Hinweis in der Grafik.
    import json

    _p = config.REPORTS_DIR / "llm_cache.json"
    llm_cache = json.loads(_p.read_text()) if _p.exists() else None
    mo.vstack([
        mo.md("## ADEngine · natives LLM-Routing: abgesichert & nicht-deterministisch"),
        figures.fig_llm_routing(llm_cache),
    ])
    return (llm_cache,)


@app.cell
def _(best_a, engine_out, figures, iid_auc, llm_cache, mo, modeB_auc, tw_auc):
    _naiv = min(iid_auc, key=iid_auc.get)
    strategies = {
        "naiv (i.i.d.)": (_naiv, iid_auc[_naiv]),
        "Konsens A": (best_a, tw_auc[best_a]),
        "Konsens B (avg)": ("ensemble", modeB_auc["average"]),
    }
    _ev = engine_out.get("validation") or {}
    if _ev.get("consensus_roc_auc") is not None:
        strategies["ADEngine"] = ("Consensus", float(_ev["consensus_roc_auc"]))
    if llm_cache and llm_cache.get("roc_auc") is not None:
        strategies["ADEngine\n+ LLM"] = (llm_cache.get("detector"), float(llm_cache["roc_auc"]))
    mo.vstack([
        mo.md("## Fazit — alle Strategien im Vergleich (ohne Labels ≈ Oracle)"),
        figures.fig_final_comparison(strategies, max(tw_auc.values())),
    ])
    return


if __name__ == "__main__":
    app.run()
