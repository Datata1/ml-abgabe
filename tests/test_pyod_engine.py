"""Test für die LLM-gesteuerte pyOD-Routing-Brücke — **ohne Netz**, Provider gemockt."""

from __future__ import annotations

import numpy as np

from automl_ad.pyod_engine import run_engine_llm_routed


def _fake_router(_prompt: str) -> str:
    # pyOD erwartet ein JSON-Array {detector, justification}.
    return (
        '[{"detector":"IForest","justification":"robuster Allrounder"},'
        '{"detector":"ECOD","justification":"schnell"},'
        '{"detector":"KNN","justification":"lokal"}]'
    )


def test_llm_routed_returns_detector_and_scores():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((300, 8))
    X_test = np.r_[rng.standard_normal((150, 8)), rng.standard_normal((15, 8)) * 5 + 8]

    out = run_engine_llm_routed(X_train, X_test, chat_fn=_fake_router)

    assert out["detector"] == "IForest"          # LLM-Primärwahl wurde übernommen
    assert "ECOD" in out["alternatives"]
    assert out["scores_test"].shape[0] == X_test.shape[0]
    assert np.all(np.isfinite(out["scores_test"]))
