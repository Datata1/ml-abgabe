"""Einmalig **vor dem Vortrag** ausführen: das LLM-Routing cachen (offline-sicher).

Das Notebook lädt danach aus ``reports/llm_cache.json`` — so hängt der Vortrag nicht an einem
laufenden Ollama/Claude. (Die ADEngine selbst läuft im ``tabular``-Modus live und schnell, muss also
**nicht** gecacht werden.)

Ehrlich: ein kleines lokales llama3.1 liefert oft ungültiges JSON → PyOD fällt sicher auf
Regel-Routing zurück. ``source`` im Cache hält fest, ob es eine echte **LLM**-Wahl oder der
**Regel**-Fallback war.

Mit ``--replay`` wird **kein** LLM befragt: die zuletzt gecachte LLM-Wahl wird auf dem aktuellen
Split neu ausgeführt/gescort (nützlich, wenn sich das Experiment-Setup geändert hat und kein
Provider läuft — die Wahl bleibt dieselbe, nur die Zahl wird konsistent).

Start:  uv run python scripts/precompute.py   (oder: make cache)
"""

from __future__ import annotations

import json
import sys

from sklearn.metrics import roc_auc_score

from automl_ad import config
from automl_ad.data import load_split
from automl_ad.pyod_engine import run_engine_llm_routed

# Muss zum Notebook passen (gleicher Split → vergleichbare Zahlen).
SPLIT_KW = config.SPLIT_KW


def _force_rule(_prompt: str) -> str:
    """chat_fn, das absichtlich fehlschlägt → erzwingt PyODs Regel-Routing (robuster Fallback)."""
    raise RuntimeError("erzwinge Regel-Routing")


def _replay_chat(cache: dict):
    """chat_fn, das die gecachte LLM-Wahl wörtlich wiedergibt (kein Provider nötig).

    PyOD erwartet ein JSON-Array aus ``{"detector", "justification"}`` — genau das bauen wir aus
    dem Cache nach; Parsing/KB-Validierung/Ausführung laufen wie bei einer echten LLM-Antwort.
    """
    picks = [cache["detector"], *cache.get("alternatives", [])]
    payload = json.dumps([
        {"detector": d, "justification": cache.get("reason", "replay der gecachten Wahl")}
        for d in picks
    ])
    return lambda _prompt: payload


def _route(split, chat_fn):
    routed = run_engine_llm_routed(split.X_train_good, split.X_test, data_type="tabular", chat_fn=chat_fn)
    return routed, float(roc_auc_score(split.y_test, routed["scores_test"]))


def main() -> None:
    split = load_split(**SPLIT_KW)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "llm_cache.json"

    if "--replay" in sys.argv:
        old = json.loads(path.read_text())
        if old.get("source") != "llm":
            raise SystemExit("--replay braucht einen Cache mit source='llm' (sonst make cache).")
        print(f"Replay der gecachten LLM-Wahl ({old['detector']}) auf dem aktuellen Split …")
        routed, auc = _route(split, _replay_chat(old))
        routed["source"] = "llm"  # die Wahl stammt aus dem echten LLM-Lauf, nur neu gescort
        routed["reason"] = old.get("reason")
    else:
        print("LLM-Routing (Provider nötig) …")
        try:
            routed, auc = _route(split, None)  # None → Default llm_router (echtes LLM)
        except Exception as exc:  # noqa: BLE001 - kleines LLM kann inkompatible Detektoren wählen (z. B. MAD)
            print(f"  LLM-Plan ungültig/inkompatibel ({type(exc).__name__}) → Regel-Fallback")
            routed, auc = _route(split, _force_rule)
            routed["source"] = "rule"
            routed["reason"] = "kleines LLM lieferte eine ungültige/inkompatible Wahl → PyOD-Regel-Fallback"

    cache = {
        "detector": routed["detector"],
        "alternatives": routed["alternatives"],
        "reason": routed["reason"],
        "source": routed["source"],  # "llm" (echte Wahl) oder "rule" (Fallback bei kaputtem JSON)
        "roc_auc": auc,
    }
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"  -> {path}: source={cache['source']}, detector={cache['detector']} (ROC-AUC={auc:.3f})")


if __name__ == "__main__":
    main()
