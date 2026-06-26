"""Tests für die LLM-Auswahl — **ohne Netz/Daten**, Provider wird gemockt (``chat_fn``)."""

from __future__ import annotations

import numpy as np

from automl_ad.llm import _parse_choice, build_prompt, profile_dataset, select_llm
from automl_ad.selection import DEFAULT_CANDIDATES


def _synthetic(n=400, d=52, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def test_profile_dataset_keys():
    p = profile_dataset(_synthetic())
    assert p["n_features"] == 52
    assert p["dimensionalitaet"] == "hoch"
    assert {"mean_abs_skew", "mean_abs_correlation", "tags"} <= set(p)


def test_build_prompt_lists_all_candidates():
    p = profile_dataset(_synthetic())
    prompt = build_prompt(p, [n for n, _ in DEFAULT_CANDIDATES])
    for name in ("ecod", "iforest", "ocsvm", "pca"):
        assert name in prompt


def test_parse_choice_substring_and_case():
    names = ["ecod", "iforest", "ocsvm", "pca"]
    assert _parse_choice('{"selected": "IForest", "reason": "x"}', names)[0] == "iforest"
    assert _parse_choice('Antwort: {"selected":"pca","reason":"y"}', names)[0] == "pca"
    assert _parse_choice("kein json", names)[0] is None


def test_select_llm_with_mocked_provider():
    def fake_chat(_prompt: str) -> str:
        return '{"selected": "ocsvm", "reason": "kompakter Normalbereich"}'

    name, info = select_llm(DEFAULT_CANDIDATES, _synthetic(), _synthetic(), chat_fn=fake_chat)
    assert name == "ocsvm"
    assert info["fallback"] is False
    assert "kompakt" in info["reason"]


def test_select_llm_fallback_on_garbage():
    name, info = select_llm(
        DEFAULT_CANDIDATES, _synthetic(), _synthetic(), chat_fn=lambda _p: "völlig unbrauchbar"
    )
    assert name == "ecod"  # robuster Default
    assert info["fallback"] is True
