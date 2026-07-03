# 03 — PyODs ADEngine: native AutoML-AD (inkl. nativer LLM-Schicht)

> Drittes Thema von JD. Wir zeigen **PyODs eigene, fertige** AutoML-AD-Pipeline — nicht unseren
> Eigenbau. Die Konsens-Idee aus `02` taucht hier **industrialisiert** wieder auf: profilieren →
> benchmark-gestützt wählen → Consensus → **label-freie Qualität** melden. Optional steuert ein LLM die
> Detektorwahl — und zwar mit **PyODs nativer** LLM-Implementierung. Hält die [Narrativ-Regeln](00_narrativ.md) ein.
>
> ⚠️ **Wichtige Abgrenzung:** Wir bauen **keine** eigene LLM-Auswahl. Unser
> [`automl_ad/llm.py`](../../automl_ad/llm.py) ist **nur** der Provider-Transport (`llm_router` →
> Claude/Ollama). Die gesamte Routing-Logik — Prompt, Parsing, KB-Constraints — gehört PyOD:
> [`pyod/utils/_llm.py`](../../.venv/lib/python3.13/site-packages/pyod/utils/_llm.py) + `plan_detection`.

---

## Was ist die ADEngine?

`pyod.utils.ad_engine.ADEngine` (in PyOD 3.6.1) ist eine **eingebaute, benchmark-gestützte**
AD-Pipeline. Aus ihrem eigenen Docstring:

> *„Works as a standalone Python API (no LLM required) or as the backend for MCP/agent interfaces."*

Sie ist also **standalone und LLM-optional**. Der ganze Ablauf steckt in einem Aufruf:

```python
from pyod.utils.ad_engine import ADEngine
engine = ADEngine(random_state=0)
state  = engine.investigate(X, data_type="tabular", priority="balanced")
```

> **Warum tabular?** Wir demonstrieren die ADEngine im **tabular**-Modus (Detektoren aus **ADBench**:
> IForest/ECOD/KNN). Der `time_series`-Modus wählt **Subsequenz**-Detektoren (KShape, MatrixProfile),
> die **seltene Discords** suchen — TEP-Fehler sind aber **anhaltende Regimewechsel** (nach Onset ist
> fast alles anomal). Deshalb passt der Subsequenz-Ansatz hier nicht (Verdikt *low*, ROC-AUC ~0.46);
> **die richtige Zeitbewusstheit für TEP ist die Fensterung aus [01](01_baseline_pyod.md)** (Punkt-
> Detektor auf Fenstern), nicht Subsequenz-AD. Wir **erwähnen** den TS-Modus (er ist da), zeigen die
> Zahlen aber im tabular-Modus, wo die Engine fair funktioniert.

`investigate` durchläuft vier Phasen (auch einzeln aufrufbar: `start → plan → run → analyze`):

1. **profile_data** — charakterisiert die Daten (Modalität, Form, Statistik). Erkennt `data_type`
   **selbst** (`_sniff_data_type`, „unter der Haube") — man **kann** ihn aber explizit übergeben
   (z. B. `data_type='time_series'`), um eine Route zu erzwingen.
2. **plan** — wählt aus einer **Knowledge Base** (60+ Detektoren) die passenden per **Benchmark-Rang**
   (ADBench für tabular, **TSB-AD** für time_series). → **Meta-Learning**: „was auf ähnlichen Daten
   gut war, probiere zuerst".
3. **run** — fittet die geplanten Detektoren und bildet **Consensus-Scores** (genau die Idee aus `02`,
   Modus B — nur über eine benchmark-gewählte Auswahl).
4. **analyze** — bewertet die Qualität **label-frei** (s. u.) und benennt den `best_detector`.

**Repo-Glue:** [`run_engine`](../../automl_ad/pyod_engine.py#L26) verpackt das und gibt
`consensus_scores`, `agreement`, `detectors`, `best_detector`, `quality_verdict`, `quality_overall`
zurück; [`benchmark_ranking`](../../automl_ad/pyod_engine.py#L125) zeigt die KB-Rangliste.

---

## Wie sieht die Benchmark **konkret** aus?

Das ist der spannende Kern: Die „Benchmark" ist **keine Messung auf deinen Daten**, sondern eine
**vorab einprogrammierte Wissensbasis** (ein kuratiertes Leaderboard) aus großen, publizierten
Studien. Die Engine schaut dort nach *„was hat auf **vielen anderen** Datensätzen gut funktioniert"*
und nimmt das als **Startannahme** — reines **Meta-Learning / Transfer**, ohne Training auf TEP.

**1. Die Benchmark-Studien** (`engine.get_benchmarks()`):

| Benchmark | Scope | Datensätze × Algorithmen | Overall Top-5 | Kernbefund |
|---|---|---|---|---|
| **ADBench** (Han et al., NeurIPS 2022) | tabular | **57 × 30** | ECOD, IForest, KNN, COPOD, HBOS | „kein Verfahren dominiert; Ensemble der Top-5 ist robust" |
| **TSB-AD** (Liu & Paparrizos, NeurIPS 2024) | time_series | **1070 × 40** | IForest, LOF, POLY, KNN, KShapeAD | „klassisch ≈ deep; MatrixProfile stark bei Subsequenz-Anomalien" |

(+ NLP-ADBench für Text, BOND für Graphen.)

**2. Der Detektor-Steckbrief** je Verfahren (`engine.explain_detector('IForest')`), u. a.:

```
category:       ensemble
data_types:     ['tabular']
strengths:      ['Excellent overall benchmark performance', 'Linear time complexity', …]
weaknesses:     ['May struggle with local anomalies', 'Axis-aligned splits miss correlated features']
best_for:       General-purpose … large or high-dimensional datasets
avoid_when:     Anomalies are local density deviations or features are strongly correlated
benchmark_rank: {'ADBench_overall': 3}      ← der konkrete Rang
paper:          Liu et al., ICDM 2008
```

**So entsteht die Rangliste** — `engine.compare_detectors(data_type='tabular')` sortiert die KB nach
`benchmark_rank`:

```
ECOD (ADBench #5) · IForest (#3) · KNN (#4) · COPOD (#6) · HBOS (#7)
```

Für `data_type='time_series'` liest die Engine stattdessen die **TSB-AD**-Ränge → KShape,
MatrixProfile, … **Merksatz:** Die Engine benchmarkt nicht *deine* Daten — sie **profiliert** sie
(Modalität/Form) und **schlägt in der vorberechneten Bestenliste nach**.

---

## Das label-freie Herzstück: das Qualitätsverdikt

Die ADEngine **bewertet ihr eigenes Ergebnis ohne Labels** (`compute_quality`). Drei Diagnostiken,
gemittelt zu `overall` → Verdikt `high/medium/low`. Jede diagnostiziert **einen** Fehlermodus **und**
schlägt **eine** Korrektur vor:

| Diagnostik | Wie berechnet (konkret) | Niedrig → vorgeschlagene Korrektur |
|-----------|-------------------------|------------------------------------|
| **agreement** | paarweise **Spearman-Korrelation zwischen den Detektoren** | Detektoren uneinig → **schwächsten Detektor entfernen** & neu laufen |
| **stability** | standardisierte Score-Lücke am Rang-k-Schnitt: `(score[k]−score[k+1]) / std`, geclippt [0,1] | viele Scores „kleben" an der Schwelle → **`contamination` anpassen** |
| **separation** | Score-Abstand *geflaggt vs. Rest* — aus den **eigenen** Vorhersagen | *(nur beschreibend, s. Caveat)* |

### Wozu — der Zusammenhang: das Verdikt steuert die **nächste Aktion**

Genau hier greift die „agentische" AutoML-Idee. `analyze()` leitet aus `overall` ein
`state.next_action` ab:

```
overall ≥ 0.4  →  next_action = "report_to_user"   # gut genug: Ergebnis melden
overall < 0.4  →  next_action = "iterate"          # Vorschlag: schwächsten Detektor raus & neu
```

- In der **vollen agentischen Nutzung** (`iterate()`) dreht die Engine damit **selbst** eine
  Verbesserungsschleife — **ohne Labels**. Das ist die AutoML-Rückkopplung.
- In **unserer Ein-Schuss-Nutzung** (`run_engine` → `investigate`) lesen wir das Verdikt nur als
  **Vertrauens-Label** ab (`quality_verdict`/`quality_overall`); wir loopen nicht.

> ⚠️ **Ehrlichkeit (steht so in PyODs Docstring):** `separation` ist **zirkulär** — die Engine
> vergleicht Scores der Punkte, die sie *selbst* als anomal geflaggt hat, mit dem Rest; das ist
> **kein** unabhängiger Korrektheitsbeweis („descriptive, not a label-free quality signal"). Das
> einzige echte **cross-detector**-Evidenz-Signal ist **`agreement`** — genau dein Konsens aus `02`.

**Das** ist, was man in der Industrie in der Hand hat: die Engine sagt selbst, wie verlässlich (und
was sie sonst probieren würde) — label-frei; belastbar v. a. über `agreement`.

### Der optionale „Spick"-Check: `validate`

`engine.validate(state, y)` prüft **im Nachhinein mit Labels**, ob der Consensus besser war als der
beste Einzeldetektor (`consensus_helped`). Das gehört strikt in den **„Wenn wir spicken würden"-Block**.

**Ergebnis (caveated):**
> ⚠️ *ROC/AUC nur zur Illustration.* Im **tabular**-Modus wählt die ADEngine benchmark-gestützt
> **IForest + ECOD + KNN**, label-freie Qualität **„medium" (~0.64)**, Consensus-ROC-AUC **~0.85** —
> ein faires, sinnvolles Ergebnis. Der **time_series**-Modus dagegen wählt Subsequenz-Detektoren
> (KShape …) und liefert auf TEP **Verdikt *low* / ROC-AUC ~0.46** — ehrlicher Befund: diese
> Verfahren passen nicht zu **anhaltenden** Fehlern; die gefensterte Zeitbewusstheit aus [01](01_baseline_pyod.md)
> ist der bessere Weg.

---

## Die native LLM-Schicht (PyODs eigene Implementierung)

Die ADEngine kann die Detektorwahl statt per Regeln von einem **LLM** treffen lassen. Entscheidend:
**PyOD besitzt die gesamte Routing-Logik selbst** — der Aufrufer liefert nur den **Transport** zu
einem Provider. Es gibt **keinen** eingebauten Provider-Adapter.

### Was übernimmt das LLM überhaupt — und was **nicht**?

Genau **einen** Schritt: die **Auswahl** (`plan`). Ohne LLM macht das eine **hand-codierte Regel**
(Profil → Detektoren nach Benchmark-Rang). Mit LLM ersetzt dessen **Reasoning** diese Regel: es liest
das Datensatz-Profil **und** das annotierte Leaderboard und begründet die geordneten Top-k.

Das LLM bekommt als Prompt (PyOD baut ihn aus der KB) **das hier** — Profil + je Detektor
Rang/best_for/avoid_when/Stärken/Schwächen:

```
TASK PROFILE: data_type=tabular, n_samples=600, n_features=52, contamination_estimate=?
AVAILABLE DETECTORS (43):
- IForest (ensemble rank=3): best_for='… high-dimensional …'; avoid_when='… features strongly correlated';
    strengths=[Excellent overall benchmark performance; Linear time complexity]; weaknesses=[…]
- KNN (proximity rank=4): …
- ECOD (probabilistic rank=5): avoid_when='features heavily correlated …'; …
…  → "gib die Top-3 als JSON [{detector, justification}] zurück"
```

- **Das LLM übernimmt:** die **kontextuelle Selektion** — die *generische* Bestenliste an das
  *konkrete* Profil anpassen. Z. B. „Features stark korreliert → ECOD meiden (obwohl ADBench #5),
  eher korrelations-bewusst" — oder einen Detektor wählen, den die starre Regel nicht nähme (z. B. LOF).
- **Das LLM übernimmt NICHT:** Profiling, Modell-Fitting, Scoring, Consensus, Qualitäts-Verdikt,
  `validate` — **alles PyOD**. Und es kann **nicht ausbrechen**: `parse_routing_response` verwirft
  jeden Namen außerhalb der KB.

**Kurz:** Das LLM ist ein **belesener Auswähler** über der Benchmark — nicht mehr. Fällt es aus, greift
die Regel (Fallback).

### Was PyOD besitzt (`pyod/utils/_llm.py`)

- **`LLMCallable`** — ein Protokoll: *irgendein* `(prompt: str) -> str`. Der User wrappt sein
  eigenes SDK (Claude/GPT/Ollama) in genau diese Signatur. PyODs Docstring zeigt das Anthropic-Beispiel.
- **`build_routing_prompt(kb_context, top_k)`** — **PyOD baut den Prompt selbst**: Task-Profil +
  je Detektor `best_for` / `avoid_when` / `strengths` / `weaknesses` / Benchmark-`rank`, und die
  Anweisung, **genau `top_k`** Detektoren als **JSON-Array** `{"detector", "justification"}` zurückzugeben.
  Bewusst ohne Chain-of-Thought, damit der Prompt über viele LLMs hinweg funktioniert.
- **`parse_routing_response(response, kb, top_k)`** — **PyOD parst & validiert selbst**: extrahiert das
  JSON (tolerant ggü. Prosa/Markdown-Fences), und **verwirft** Namen, die nicht in der KB sind,
  nicht `shipped` sind oder Duplikate. So kann das LLM **nicht** aus der erlaubten Menge ausbrechen.
- **`RoutingParseError`** — schlägt das Parsen fehl, fällt die Engine auf **Regel-Routing** zurück
  (außer `PYOD3_LLM_STRICT=1`).

### Wie man es aufruft

```python
plan = engine.plan_detection(profile, llm_client=my_llm, top_k=3)
#  intern: get_kb_for_routing → build_routing_prompt → my_llm(prompt)
#          → parse_routing_response → KB-Constraints erzwingen → Plan
result = engine.run_detection(X_train, plan, X_test=X_test)
```

Zusätzlich gibt es **Agent-Surfaces** für MCP/Agenten: `get_kb_for_routing` (KB als strukturierter
Kontext) und `make_plan` (der Aufrufer/Agent wählt selbst) — die Grundlage, auf der ein Agent die
Engine steuern kann.

### Unsere Rolle = nur der Transport

Unser [`run_engine_llm_routed`](../../automl_ad/pyod_engine.py#L76) ist **genau dieser dünne
Transport**: Es reicht `llm_router` (ein `(prompt)->str`, das Claude bzw. lokales Ollama aufruft) als
`llm_client` rein und schließt torch-Detektoren aus (Repo ist torch-frei). **Die Routing-Logik —
Prompt, Parsing, KB-Enforcement — bleibt PyODs.** Genau das wollen wir zeigen.

**Ollama als Transport:** POST an `http://localhost:11434/api/chat`; Provider-Präzedenz im Repo:
Claude (falls `ANTHROPIC_API_KEY`) sonst Ollama.

### Format standardisieren: Structured Output (= die MCP-Idee, schlank)

Ein kleines Modell scheitert oft nicht an der *Wahl*, sondern an der *Form*: llama3.1 liefert mal
`{...}` statt `[{...}]`, mal verschachtelt `{"detectors": […]}`. PyODs Parser will ein Array → sonst
Regel-Fallback. **Die Lösung ist Schema-gebundene Ausgabe** — genau das Prinzip, für das PyOD auch die
**MCP-/Agent-Surfaces** hat (`get_kb_for_routing` + `make_plan`, wo ein MCP-Host die Tool-Argumente
gegen ein Schema erzwingt). Ohne einen MCP-Host holen wir dieselbe Garantie **schlank** über Ollamas
**`format`-Schema** (Structured Output / constrained decoding): wir übergeben ein JSON-Schema
(`[{detector, justification}]`, `maxItems: 3`), das llama3.1 zwingt, exakt diese Form zu erzeugen.
Code: [`llm._ROUTING_SCHEMA`](../../automl_ad/llm.py). **Wichtig:** das fixt die **Form**, nicht die
**Qualität** der Wahl.

### Die zwei Kernaussagen zum LLM-Ansatz

1. **Sicher — die LLM-Antwort kann nicht ausbrechen und nichts Ungültiges erzeugen.**
   Drei Schranken: **Form** (Structured Output erzwingt `[{detector, justification}]`), **Auswahl**
   (`parse_routing_response` lässt nur KB-Detektoren zu → kein Ausbruch), **Ausführung** (schlägt ein
   Pick doch fehl, greift der **Regel-Fallback**). Egal was das Modell sagt — das System bleibt stabil.

2. **Nicht-deterministisch — die Aussagen schwanken von Lauf zu Lauf.**
   Dieselbe Frage liefert mal **LOF** (gut, dichtebasiert → ROC-AUC ~0.99), mal **ROD/SOD**, teils mit
   sachlich falscher Begründung („ROD für low-dim" — TEP hat 52 Dim). Nicht reproduzierbar. **Deshalb
   cachen wir einen Lauf** (`make cache`) und fixieren eine gute Wahl für den Vortrag.

> Merksatz: Das LLM ist ein **austauschbarer, abgesicherter Auswähler** — robust *by design*, aber in
> der Güte modellabhängig (ein größeres Modell wählt konsistenter). Der Wert liegt in PyODs Schranken,
> nicht im einzelnen Pick.

---

## Flussdiagramm (für Slide/Notebook)

```
                 ┌──────────────── PyOD ADEngine (nativ) ────────────────┐
   X  ──►  profile_data ──►  get_kb_for_routing ──►  build_routing_prompt │
                 │                    (KB, 61 Detektoren)      │ (Prompt)  │
                 │                                             ▼           │
                 │                                    ┌──────────────┐     │
                 │        UNSER Transport  ◄──────────┤  prompt:str  │     │
                 │        (llm_router →               └──────────────┘     │
                 │         Claude / Ollama)                    │           │
                 │                 │  response:str (JSON)      ▼           │
                 │                 └──────►  parse_routing_response        │
                 │                          + KB-Constraints erzwingen     │
                 │                                    │ (Plan)             │
                 └──────────  run_detection  ◄────────┘                    │
                                     │  scores_test                        │
                 └─────────────────── analyze / quality-Verdikt ───────────┘
```

**Merksatz:** PyOD baut den Prompt, parst die Antwort und erzwingt die Regeln. Wir liefern nur die
**Leitung** zum Sprachmodell.

---

## Fazit dieses Teils

Die ADEngine ist die **industrialisierte** Konsens-Idee: sie profiliert (erkennt die Zeitreihe selbst,
wir geben `data_type` trotzdem explizit mit), wählt benchmark-gestützt, bildet Consensus und bewertet
sich **label-frei** selbst. Das LLM ist optional nur der *belesene Auswähler* über der Benchmark —
alles andere bleibt PyOD.
