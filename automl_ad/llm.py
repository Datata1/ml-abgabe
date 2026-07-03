"""Provider-Transport für PyODs natives LLM-Routing.

Wir bauen **keine** eigene LLM-Auswahl. PyODs ``ADEngine.plan_detection(llm_client=...)`` besitzt die
gesamte Routing-Logik selbst (Prompt-Bau + Antwort-Parsing + KB-Constraints, siehe
``pyod.utils._llm``). Diese Datei liefert nur den **Transport**: ein ``(prompt) -> str``-Callable, das
den von PyOD gebauten Prompt an einen Provider durchreicht.

Provider: primär Claude (``ANTHROPIC_API_KEY``), Fallback lokales Ollama. Genutzt von
``pyod_engine.run_engine_llm_routed`` (Default-``chat_fn`` = ``llm_router``).
"""

from __future__ import annotations

import json
import os

from . import config  # noqa: F401 - löst das Laden der optionalen .env aus

# Neutraler System-Prompt: die Formatvorgaben liefert PyODs Routing-Prompt selbst.
_ROUTER_SYSTEM = (
    "Du bist Experte für unüberwachte Anomalieerkennung. Befolge exakt die Formatvorgaben im "
    "folgenden Prompt und antworte ausschließlich mit dem geforderten JSON, ohne weiteren Text."
)

# Structured Output: erzwingt die Form der Routing-Antwort (Array aus {detector, justification}) per
# constrained decoding — dasselbe Prinzip wie MCP-Tool-Calling, nur schlank über Ollamas ``format``.
# Fixt die FORM (kein Objekt-statt-Array mehr), nicht die Qualität der Wahl.
_ROUTING_SCHEMA = {
    "type": "array",
    "maxItems": 3,  # ohne Limit generiert das Modell ein unbegrenzt langes Array → Timeout
    "items": {
        "type": "object",
        "properties": {
            "detector": {"type": "string"},
            "justification": {"type": "string"},
        },
        "required": ["detector", "justification"],
    },
}


def llm_router(prompt: str) -> str:
    """``(prompt) -> str``-Callable für PyODs ``ADEngine.plan_detection(llm_client=...)``.

    PyOD baut den Routing-Prompt aus seiner Wissensbasis und parst die Antwort selbst — diese Funktion
    reicht ihn an den Provider durch, **erzwingt aber die Antwortform** via Structured Output
    (:data:`_ROUTING_SCHEMA`), damit kleine Modelle nicht an der JSON-Form scheitern und PyOD unnötig
    auf Regel-Routing zurückfällt.
    """
    return _coerce_json_array(llm_chat(prompt, system=_ROUTER_SYSTEM, fmt=_ROUTING_SCHEMA))


def _coerce_json_array(raw: str) -> str:
    """Sicherheitsnetz: einzelnes JSON-Objekt → ein-elementiges Array. Sonst unverändert."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(obj, dict):
        return json.dumps([obj])
    return raw


def llm_chat(prompt: str, *, system: str = _ROUTER_SYSTEM, fmt: object = "json") -> str:
    """Schickt den Prompt an den verfügbaren Provider und gibt die Roh-Textantwort zurück.

    - ``ANTHROPIC_API_KEY`` gesetzt → Claude (Modell aus ``ANTHROPIC_MODEL``).
    - sonst → lokales Ollama (``OLLAMA_HOST`` / ``OLLAMA_MODEL``).
    - ist keiner erreichbar → klarer Fehler (kein stilles Raten).

    ``fmt`` = Ollamas ``format`` (``"json"`` oder ein JSON-Schema für Structured Output).
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return _chat_claude(prompt, system)
    return _chat_ollama(prompt, system, fmt)


def _chat_claude(prompt: str, system: str) -> str:
    from anthropic import Anthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
    client = Anthropic()  # liest ANTHROPIC_API_KEY aus der Umgebung
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _chat_ollama(prompt: str, system: str, fmt: object = "json") -> str:
    import httpx

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    try:
        resp = httpx.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": fmt,  # "json" oder ein JSON-Schema (Structured Output)
                "options": {"num_predict": 512},  # Sicherheitskappe gegen Endlos-Generierung
            },
            timeout=120.0,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - bewusst: klaren Hinweis statt stiller Fehler
        raise RuntimeError(
            "Kein LLM-Provider erreichbar. Entweder ANTHROPIC_API_KEY setzen (Claude) oder ein "
            f"lokales Ollama unter {host} starten (ollama pull {model}). Ursache: {exc}"
        ) from exc
    return resp.json()["message"]["content"]
