"""Interaktive Demo (marimo): Modellauswahl ohne Labels in drei Stufen.

Start:  uv run marimo edit notebooks/demo.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # AutoML für Anomalieerkennung
        ## Wie wählt man **ohne Labels** das passende Modell?

        Drei Stufen am Tennessee-Eastman-Prozess (TEP):

        1. **naiv** — ein fixer pyOD-Detektor (Default)
        2. **statistisch** — Konsens / Model-Centrality (label-frei)
        3. **echte AutoML-AD** — pyODs eingebaute ADEngine (benchmark-gestützt, Consensus)
           — und „unter der Haube" der vereinfachte LLM-Eigenbau (PyOD-2-Idee)

        Als Obergrenze dient die **Oracle**-Auswahl (mit Labels — in echt nicht verfügbar).
        """
    )
    return (mo,)


@app.cell
def _():
    from sklearn.metrics import roc_auc_score

    from automl_ad.data import load_split
    from automl_ad.detectors import make_detector
    from automl_ad.selection import DEFAULT_CANDIDATES

    split = load_split()

    # Test-ROC-AUC je Detektor (gemeinsame Bewertungsbasis).
    test_auc = {}
    for name, hp in DEFAULT_CANDIDATES:
        det = make_detector(name, **hp).fit(split.X_train_good)
        test_auc[name] = float(roc_auc_score(split.y_test, det.decision_function(split.X_test)))
    test_auc
    return DEFAULT_CANDIDATES, split, test_auc


@app.cell
def _(mo, test_auc):
    mo.md(
        "### Stufe 1 — naiv\n"
        f"Fixer Default **ecod**: ROC-AUC = **{test_auc['ecod']:.3f}**. "
        "Ohne Labels weiß man aber nicht, ob das eine gute Wahl ist — die Detektoren streuen:\n\n"
        + "\n".join(f"- `{n}`: {a:.3f}" for n, a in sorted(test_auc.items()))
    )
    return


@app.cell
def _(DEFAULT_CANDIDATES, mo, split, test_auc):
    from automl_ad.selection import select_internal

    pick_internal, centrality = select_internal(
        DEFAULT_CANDIDATES, split.X_train_good, split.X_test
    )
    mo.md(
        "### Stufe 2 — statistisch (Konsens)\n"
        f"Label-freie Auswahl wählt **{pick_internal}** "
        f"(ROC-AUC = **{test_auc[pick_internal]:.3f}**) — ganz ohne Labels."
    )
    return


@app.cell
def _(mo, split):
    import numpy as np

    from automl_ad.pyod_engine import benchmark_ranking, run_engine

    # ADEngine fittet+bewertet auf demselben X -> Subsample der Testdaten.
    _rng = np.random.default_rng(0)
    _idx = np.sort(_rng.choice(len(split.X_test), size=min(8000, len(split.X_test)), replace=False))
    eng = run_engine(split.X_test[_idx], split.y_test[_idx])
    _val = eng["validation"]
    mo.md(
        "### Stufe 3 — die echte AutoML-AD: pyOD ADEngine\n"
        f"Benchmark-Rangliste (ADBench): {benchmark_ranking()}\n\n"
        f"**Gewählte Detektoren:** {', '.join(eng['detectors'])}\n\n"
        f"**Label-freie Qualität:** {eng['quality_verdict']} ({eng['quality_overall']:.2f})\n\n"
        f"**Selbst-Validierung:** Consensus-ROC-AUC = **{_val['consensus_roc_auc']:.3f}**, "
        f"bester Einzeldetektor = {_val['best_detector_roc_auc']:.3f}, "
        f"consensus_helped = **{_val['consensus_helped']}**"
    )
    return


@app.cell
def _(DEFAULT_CANDIDATES, mo, split):
    # "Unter der Haube": wie wissensbasierte Auswahl intern funktioniert (vereinfacht).
    from automl_ad.selection import select_llm

    try:
        pick_llm, info = select_llm(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
        out = mo.md(
            "### Unter der Haube — LLM-Eigenbau (PyOD-2-Idee)\n"
            f"Das Sprachmodell wählt **{pick_llm}**.\n\n"
            f"**Datensatz-Profil:** {info['profile']['tags']}\n\n"
            f"**Begründung:** {info['reason']}"
        )
    except RuntimeError as exc:
        out = mo.md(
            "### Unter der Haube — LLM-Eigenbau (PyOD-2-Idee)\n"
            f"*Kein LLM-Provider erreichbar* — `ANTHROPIC_API_KEY` setzen oder Ollama starten.\n\n"
            f"> {exc}"
        )
    out
    return


@app.cell
def _(DEFAULT_CANDIDATES, mo, split, test_auc):
    from automl_ad.data import load_validation
    from automl_ad.selection import select_oracle

    X_val, y_val, _ = load_validation(split)
    pick_oracle, _aucs = select_oracle(DEFAULT_CANDIDATES, split.X_train_good, X_val, y_val)
    mo.md(
        "### Obergrenze — Oracle (mit Labels)\n"
        f"Mit Labels wäre **{pick_oracle}** optimal (ROC-AUC = **{test_auc[pick_oracle]:.3f}**).\n\n"
        "**Kernaussage:** Die label-freien Strategien kommen sehr nah an diese Obergrenze heran."
    )
    return


if __name__ == "__main__":
    app.run()
