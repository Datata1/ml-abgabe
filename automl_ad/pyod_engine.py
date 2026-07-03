"""PyODs eingebaute AutoML-AD (``ADEngine``) — die native, benchmark-gestützte Pipeline.

``pyod.utils.ad_engine.ADEngine`` liefert eine fertige, **benchmark-gestützte**
Anomalie-Erkennungs-Pipeline:

- **profiliert** den Datensatz,
- **wählt** aus 60+ Detektoren benchmark-gestützt (ADBench für tabular, TSB-AD für time_series)
  die passenden aus — Meta-Learning, fertig in der Library,
- bildet **Multi-Detektor-Consensus** (kontinuierlicher Score, label-frei),
- liefert **label-freie Qualitätsdiagnostik** (``verdict``/``overall``),
- kann hinterher mit ``validate(state, y)`` optional gegen echte Labels auswerten
  (Consensus vs. bester Einzeldetektor, inkl. ``consensus_helped``).

Das **LLM-Routing** ist ebenfalls PyOD-nativ (``plan_detection(llm_client=...)`` +
``pyod.utils._llm``): PyOD baut Prompt & parst/validiert die Antwort selbst; wir liefern via
``run_engine_llm_routed`` nur den Provider-Transport (``llm.llm_router``).
"""

from __future__ import annotations

import numpy as np

from . import config


def run_engine(
    X,
    y=None,
    *,
    data_type: str = "tabular",
    priority: str = "balanced",
    random_state: int = config.RANDOM_SEED,
) -> dict:
    """Führt eine ADEngine-Untersuchung auf ``X`` aus (label-frei) und gibt die Kernergebnisse.

    ADEngine fittet und bewertet im Session-Workflow (``investigate``) auf demselben ``X``.
    Rückgabe (Dict): ``consensus_scores`` (kontinuierlich, höher = anomaler), ``agreement``,
    ``detectors`` (benchmark-gewählte Detektoren), ``best_detector``, ``quality_verdict`` /
    ``quality_overall`` (label-frei). Wird ``y`` übergeben, ergänzt ADEngines eigene
    ``validate``-Methode ``validation`` (Consensus-ROC-AUC/F1, bester Detektor, ``consensus_helped``).
    """
    from pyod.utils.ad_engine import ADEngine

    X = np.asarray(X, dtype=float)
    engine = ADEngine(random_state=random_state)
    state = engine.investigate(X, data_type=data_type, priority=priority)

    consensus = state.consensus or {}
    quality = state.quality or {}
    analysis = state.analysis or {}
    successful = [r for r in state.results if r.get("status") == "success"]

    out = {
        "consensus_scores": np.asarray(consensus.get("scores"), dtype=float),
        "agreement": float(consensus.get("agreement", float("nan"))),
        "detectors": [r.get("detector_name") for r in successful],
        "best_detector": analysis.get("best_detector"),
        "quality_verdict": quality.get("verdict"),
        "quality_overall": float(quality.get("overall", float("nan"))),
        "validation": None,
    }

    if y is not None:
        val = engine.validate(state, np.asarray(y))
        best = val.get("best_detector")
        vs_best = val.get("consensus_vs_best", {})
        out["validation"] = {
            "consensus_roc_auc": float(val["consensus"]["roc_auc"]),
            "consensus_f1": float(val["consensus"]["f1"]),
            "best_detector_roc_auc": float(best["roc_auc"]) if isinstance(best, dict) else None,
            "consensus_helped": vs_best.get("consensus_helped"),
        }
    return out


def run_engine_llm_routed(
    X_train,
    X_test,
    *,
    chat_fn=None,
    data_type: str = "tabular",
    top_k: int = 3,
    random_state: int = config.RANDOM_SEED,
) -> dict:
    """Bonus: **unser** LLM steuert pyODs Detektorwahl (`plan_detection(llm_client=...)`).

    Verbindet „unter der Haube" (unser LLM) mit „echt" (pyOD-Engine): pyOD baut den Routing-Prompt
    aus seiner Wissensbasis (61 Detektoren), unser ``chat_fn`` (Default ``llm.llm_router``) liefert
    die Antwort, pyOD parst sie zu einem Plan und führt ihn aus (induktiv: fit auf ``X_train``,
    Score auf ``X_test``). Bei ungültiger LLM-Antwort fällt pyOD selbst auf Regel-Routing zurück.

    Rückgabe: ``detector`` (LLM-Primärwahl), ``alternatives``, ``reason``, ``scores_test``.
    """
    from pyod.utils.ad_engine import ADEngine

    from .llm import llm_router

    chat = chat_fn if chat_fn is not None else llm_router
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    engine = ADEngine(random_state=random_state)
    profile = engine.profile_data(X_train, data_type=data_type)
    # Deep-Learning-Detektoren ausschließen (Repo ist bewusst torch-frei) — dynamisch aus dem
    # ``requires``-Feld der Wissensbasis, damit das LLM nur lauffähige Detektoren wählen kann.
    deep = [
        d["name"]
        for d in engine.list_detectors(data_type=data_type)
        if any("torch" in str(r).lower() for r in (d.get("requires") or []))
    ]
    plan = engine.plan_detection(
        profile, llm_client=chat, top_k=top_k,
        constraints={"exclude_detectors": deep},
    )
    result = engine.run_detection(X_train, plan, X_test=X_test)

    # ``source`` = "llm", wenn PyODs LLM-Plan genutzt wurde; "rule", wenn PyOD wegen ungültiger
    # LLM-Antwort auf Regel-Routing zurückgefallen ist (Note wird in _plan_via_llm gesetzt).
    return {
        "detector": plan.get("detector_name"),
        "alternatives": [a.get("detector_name") for a in plan.get("alternatives", [])],
        "reason": plan.get("reason"),
        "source": "llm" if str(plan.get("note", "")).startswith("llm-driven") else "rule",
        "scores_test": np.asarray(result.get("scores_test"), dtype=float),
    }


def benchmark_ranking(data_type: str = "tabular", top_k: int = 5) -> list[tuple[str, object]]:
    """Benchmark-Rangliste der Detektoren aus pyODs Wissensbasis (z. B. ADBench für 'tabular').

    Zeigt die „Meta-Learning"-Grundlage, auf der ADEngine seine Auswahl trifft.
    Rückgabe: Liste ``(Detektorname, benchmark_rank)``.
    """
    from pyod.utils.ad_engine import ADEngine

    engine = ADEngine()
    ranked = engine.compare_detectors(data_type=data_type, top_k=top_k)
    return [(d.get("name"), d.get("benchmark_rank")) for d in ranked]
