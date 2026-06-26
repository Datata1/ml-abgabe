"""LLM-gestützte Modellauswahl (vereinfachte PyOD-2-Idee, Chen et al. 2024).

Die dritte Stufe des roten Fadens: *wissensbasierte* Auswahl. Statt rein statistisch (Konsens)
beschreiben wir den Datensatz und die Detektoren in Worten und lassen ein Sprachmodell den
passendsten Detektor **mit Begründung** wählen. Pipeline (analog Paper, vereinfacht):

1. **Detektor-Steckbriefe** (Stärken/Schwächen) als symbolisches Wissen — ``DETECTOR_CATALOG``.
2. **Datensatz-Profiling** (statistische Kennzahlen → Beschreibung) — ``profile_dataset``.
3. **Auswahl per Reasoning**: Prompt aus 1+2 → LLM nennt Detektor + Begründung — ``select_llm``.

Provider: primär Claude (API-Key), Fallback lokales Ollama. Für Tests kann ``chat_fn`` injiziert
werden, sodass kein Netz nötig ist.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import numpy as np
from scipy.stats import kurtosis, skew

from . import config  # noqa: F401 - löst das Laden der optionalen .env aus

# --------------------------------------------------------------------------------------
# 1) Detektor-Steckbriefe (symbolisches Wissen, aus PyOD-Doku/Literatur)
# --------------------------------------------------------------------------------------
DETECTOR_CATALOG: dict[str, dict[str, list[str]]] = {
    "ecod": {
        "staerken": [
            "parameterfrei und schnell; robust bei schweren Verteilungs-Tails je Feature",
            "gut, wenn die Features weitgehend UNABHÄNGIG sind",
        ],
        "schwaechen": [
            "behandelt jedes Feature einzeln und nimmt Feature-Unabhängigkeit an",
            "ignoriert Korrelationen zwischen Features -> schwach bei GEKOPPELTEN Größen "
            "(typisch für Prozess-/Sensordaten mit ausgeprägter linearer Struktur)",
        ],
    },
    "iforest": {
        "staerken": [
            "skaliert gut auf viele Samples und Features, robuste Defaults",
            "wenige Annahmen, allrounder für hochdimensionale Tabellendaten",
        ],
        "schwaechen": [
            "schwächer bei sehr lokalen Anomalien / feinen Dichteunterschieden",
            "modelliert lineare Kopplungen nicht explizit; zufallsbasiert",
        ],
    },
    "ocsvm": {
        "staerken": [
            "kernbasiert: erfasst nichtlineare Entscheidungsgrenzen und Korrelationen",
            "stark, wenn der Normalbereich kompakt und gut abgegrenzt ist",
        ],
        "schwaechen": [
            "rechenintensiv und langsam bei vielen Datenpunkten",
            "empfindlich gegenüber den Hyperparametern nu und gamma",
        ],
    },
    "pca": {
        "staerken": [
            "nutzt explizit die Korrelationsstruktur (Rekonstruktion im linearen Unterraum)",
            "sehr stark bei GEKOPPELTEN Größen / ausgeprägter linearer Struktur "
            "(typisch für Prozess-/Sensordaten); schnell und interpretierbar",
        ],
        "schwaechen": [
            "erfasst nur lineare Zusammenhänge",
            "schwach bei rein nichtlinearen Mannigfaltigkeiten",
        ],
    },
}


# --------------------------------------------------------------------------------------
# 2) Datensatz-Profiling
# --------------------------------------------------------------------------------------
def profile_dataset(X: np.ndarray, *, max_rows: int = 5000, seed: int = 0) -> dict:
    """Berechnet beschreibende Statistiken eines (skalierten) Feature-Arrays.

    Liefert ein Dict mit Kennzahlen + abgeleiteten, gut lesbaren Tags (für den Prompt).
    Große Arrays werden zur Beschleunigung zeilenweise subsampelt.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    if n_samples > max_rows:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(n_samples, size=max_rows, replace=False)]

    mean_abs_skew = float(np.mean(np.abs(skew(X, axis=0, nan_policy="omit"))))
    mean_excess_kurtosis = float(np.mean(kurtosis(X, axis=0, fisher=True, nan_policy="omit")))

    # Mittlere absolute lineare Korrelation zwischen Features (Indikator für Kopplung/Redundanz).
    corr = np.corrcoef(X, rowvar=False)
    off_diag = corr[~np.eye(n_features, dtype=bool)]
    mean_abs_corr = float(np.nanmean(np.abs(off_diag)))
    share_high_corr = float(np.nanmean(np.abs(off_diag) > 0.7))

    # Lineare Struktur: Varianzanteil in den ersten Hauptkomponenten. Hoch = die Daten leben
    # in einem niederdimensionalen linearen Unterraum (spricht für korrelations-modellierende
    # Verfahren wie pca/ocsvm; ecod ignoriert solche Kopplungen).
    pca_var_topk = _pca_explained_var(X, k=min(5, n_features))

    dim_tag = "hoch" if n_features >= 50 else "mittel" if n_features >= 10 else "niedrig"
    coupling = mean_abs_corr > 0.25 or pca_var_topk > 0.6
    struktur_tag = (
        "stark gekoppelte Features (ausgeprägte lineare Struktur)" if coupling
        else "kaum lineare Kopplung (Features eher unabhängig)"
    )
    tail_tag = "schwere Tails / ausgeprägte Schiefe" if (
        mean_abs_skew > 1.0 or mean_excess_kurtosis > 3.0
    ) else "moderate Verteilung"

    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "dimensionalitaet": dim_tag,
        "mean_abs_skew": round(mean_abs_skew, 3),
        "mean_excess_kurtosis": round(mean_excess_kurtosis, 3),
        "mean_abs_correlation": round(mean_abs_corr, 3),
        "share_high_corr_pairs": round(share_high_corr, 3),
        "pca_var_top5": round(pca_var_topk, 3),
        "tags": [dim_tag + "-dimensional", struktur_tag, tail_tag],
    }


def _pca_explained_var(X: np.ndarray, k: int) -> float:
    """Varianzanteil der ersten k Hauptkomponenten (0..1). Robust gegen kleine Fehler."""
    try:
        from sklearn.decomposition import PCA

        p = PCA(n_components=k).fit(X)
        return float(p.explained_variance_ratio_.sum())
    except Exception:  # noqa: BLE001 - Profiling darf nie hart fehlschlagen
        return float("nan")


# --------------------------------------------------------------------------------------
# 3a) Prompt-Bau
# --------------------------------------------------------------------------------------
_SYSTEM = (
    "Du bist ein Experte für unüberwachte Anomalieerkennung. Aufgabe: aus einer festen Liste von "
    "Detektoren den auswählen, der am besten zu den statistischen Eigenschaften des Datensatzes passt. "
    "Gehe systematisch vor: bewerte ZUERST jeden Detektor einzeln (passt seine Stärke/Schwäche zu den "
    "Datensatz-Eigenschaften?), wähle ERST DANACH den besten. Achte besonders darauf, ob die Features "
    "gekoppelt (korreliert) sind: bei stark gekoppelten Größen sind korrelations-modellierende "
    "Verfahren im Vorteil, unabhängigkeitsannehmende im Nachteil. "
    "Antworte ausschließlich mit einem JSON-Objekt der Form "
    '{"assessment": {"<detektor>": "<kurze Eignung>"}, "selected": "<detektorname>", '
    '"reason": "<kurze deutsche Begründung>"}.'
)


def build_prompt(profile: dict, candidate_names: list[str]) -> str:
    """Baut den Nutzer-Prompt aus Datensatz-Profil und Detektor-Steckbriefen."""
    katalog_zeilen = []
    for name in candidate_names:
        info = DETECTOR_CATALOG.get(name, {"staerken": [], "schwaechen": []})
        katalog_zeilen.append(
            f"- {name}:\n"
            f"    Stärken: {'; '.join(info['staerken'])}\n"
            f"    Schwächen: {'; '.join(info['schwaechen'])}"
        )
    katalog = "\n".join(katalog_zeilen)

    return (
        "Datensatz-Profil (Features sind bereits standardisiert):\n"
        f"- Stichproben: {profile['n_samples']}, Features: {profile['n_features']} "
        f"({profile['dimensionalitaet']}-dimensional)\n"
        f"- mittlere |Schiefe|: {profile['mean_abs_skew']}, "
        f"mittlere Exzess-Kurtosis: {profile['mean_excess_kurtosis']}\n"
        f"- mittlere |Korrelation| zwischen Features: {profile['mean_abs_correlation']} "
        f"(Anteil stark korrelierter Paare: {profile['share_high_corr_pairs']})\n"
        f"- Varianzanteil der 5 größten Hauptkomponenten: {profile.get('pca_var_top5')} "
        "(hoch = ausgeprägte lineare Struktur / gekoppelte Features)\n"
        f"- Tags: {', '.join(profile['tags'])}\n\n"
        "Verfügbare Detektoren:\n"
        f"{katalog}\n\n"
        f"Bewerte jeden Detektor kurz und wähle dann genau einen aus dieser Liste: {candidate_names}."
    )


# --------------------------------------------------------------------------------------
# 3b) Provider (Claude primär, Ollama als Fallback)
# --------------------------------------------------------------------------------------
def llm_chat(prompt: str, *, system: str = _SYSTEM) -> str:
    """Schickt den Prompt an den verfügbaren Provider und gibt die Roh-Textantwort zurück.

    - Wenn ``ANTHROPIC_API_KEY`` gesetzt ist → Claude (Modell aus ``ANTHROPIC_MODEL``).
    - Sonst → lokales Ollama (``OLLAMA_HOST`` / ``OLLAMA_MODEL``).
    - Ist keiner erreichbar, wird ein klarer Fehler geworfen (kein stilles Raten).
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return _chat_claude(prompt, system)
    return _chat_ollama(prompt, system)


# Neutraler System-Prompt für pyODs eigenes Routing: pyOD liefert die Formatvorgaben selbst.
_ROUTER_SYSTEM = (
    "Du bist Experte für unüberwachte Anomalieerkennung. Befolge exakt die Formatvorgaben im "
    "folgenden Prompt und antworte ausschließlich mit dem geforderten JSON, ohne weiteren Text."
)


def llm_router(prompt: str) -> str:
    """``(prompt) -> str``-Callable für pyODs ``ADEngine.plan_detection(llm_client=...)``.

    pyOD baut den Routing-Prompt aus seiner Wissensbasis und parst die Antwort selbst — diese
    Funktion reicht den Prompt nur an den Provider (Claude/Ollama) durch. Damit kann **unser**
    LLM die Detektorwahl der echten pyOD-Engine steuern (Brücke „unter der Haube" → „echt").
    """
    return llm_chat(prompt, system=_ROUTER_SYSTEM)


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


def _chat_ollama(prompt: str, system: str) -> str:
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
                "format": "json",
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


# --------------------------------------------------------------------------------------
# 3c) Auswahl
# --------------------------------------------------------------------------------------
def _parse_choice(raw: str, candidate_names: list[str]) -> tuple[str | None, str]:
    """Extrahiert (selected, reason) aus der LLM-Antwort; tolerant gegenüber Drumherum-Text."""
    reason = ""
    selected = None
    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start : end + 1]) if start != -1 and end != -1 else {}
        selected = obj.get("selected")
        reason = str(obj.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        obj = {}
    # Robuste Zuordnung gegen die Kandidatenliste (Groß-/Kleinschreibung, Teilstring).
    if isinstance(selected, str):
        low = selected.strip().lower()
        for name in candidate_names:
            if name == low or name in low:
                return name, reason
    return None, reason


def select_llm(
    candidates,
    X_train,
    X_eval,
    *,
    chat_fn: Callable[[str], str] | None = None,
) -> tuple[str, dict]:
    """Label-freie, **wissensbasierte** Auswahl per Sprachmodell (PyOD-2-Idee).

    Signatur analog zu ``select_oracle``/``select_internal``. ``X_train`` wird nicht benötigt
    (die Auswahl basiert auf dem Profil von ``X_eval``), ist aber für eine einheitliche
    Schnittstelle vorhanden. ``chat_fn`` erlaubt das Injizieren eines Providers (Tests).

    Rückgabe: ``(gewählter_detektor, info)`` mit ``info = {profile, reason, raw, fallback}``.
    """
    candidate_names = [name for name, _ in candidates]
    profile = profile_dataset(X_eval)
    prompt = build_prompt(profile, candidate_names)

    chat = chat_fn if chat_fn is not None else llm_chat
    raw = chat(prompt)
    selected, reason = _parse_choice(raw, candidate_names)

    fallback = selected is None
    if fallback:
        # Ungültige/leere Antwort → konservativer, robuster Default (ECOD, sonst erster Kandidat).
        selected = "ecod" if "ecod" in candidate_names else candidate_names[0]
        reason = reason or "Fallback: keine gültige LLM-Antwort, robuster Default gewählt."

    return selected, {"profile": profile, "reason": reason, "raw": raw, "fallback": fallback}
