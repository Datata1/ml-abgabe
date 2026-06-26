# AutoML für Anomalieerkennung — Modellauswahl ohne Labels

Abgabe-Projekt zum Thema **AutoML für unüberwachte Anomalieerkennung**.

Leitfrage: **Wie wählt man — ohne Labels — für einen gegebenen Datensatz ein passendes
Anomalie-Detektionsmodell?** Wir zeigen das in **drei Stufen** auf dem Tennessee-Eastman-Prozess
(TEP, simulierter Chemieprozess, 52 Sensoren, 20 Fehlertypen):

| Stufe | Idee | Funktion |
|-------|------|----------|
| 1. **naiv** | ein fixer pyOD-Detektor mit Defaults (willkürlich) | `make_detector` |
| 2. **statistisch** | Konsens / Model-Centrality, label-frei | `select_internal` |
| 3. **echte AutoML-AD** | pyODs eingebaute, benchmark-gestützte ADEngine (Consensus + label-freie Qualität) | `pyod_engine.run_engine` |
| (unter der Haube) | vereinfachter Eigenbau: LLM-gestützte Auswahl (PyOD-2-Idee) | `select_llm` |
| (Bonus) | **unser LLM steuert pyODs Routing** (`plan_detection(llm_client=…)`) — wählt aus 61 Detektoren | `run_engine_llm_routed` |
| (Obergrenze) | **Oracle** — Auswahl mit Labels (in echt nicht verfügbar) | `select_oracle` |

**Kernaussage:** Die label-freien Strategien (Stufe 2 & 3) kommen sehr nah an die Oracle-Obergrenze
heran — AutoML kann das AD-Modell praktisch ohne Labels nahezu optimal wählen.

> 📖 **Einfache Erklärung jeder Stufe** (was sie tut, wie sie es tut, mit Bezug auf die
> Ergebnisse): [`docs/erklaerung_3_stufen.md`](docs/erklaerung_3_stufen.md).

## Setup

```bash
uv sync
```

(Alternativ `pip install -e .` in einer Python-3.13-Umgebung.)

## Daten (TEP)

Die vier Parquet-Dateien werden **nicht** mitgeliefert. Sie nach `data/` legen:

```
data/TEP_FaultFree_Training.parquet
data/TEP_FaultFree_Testing.parquet
data/TEP_Faulty_Training.parquet
data/TEP_Faulty_Testing.parquet
```

Quelle: Tennessee-Eastman-Datensatz (Rieth et al., Harvard Dataverse) — RData nach Parquet
konvertieren (Spalten `faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11`).

## LLM-Selektor konfigurieren (Stufe 3)

`.env.example` nach `.env` kopieren und ausfüllen. Zwei Provider:

- **Claude (primär):** `ANTHROPIC_API_KEY` setzen (Modell via `ANTHROPIC_MODEL`).
- **Ollama (Fallback):** lokales Ollama starten und Modell ziehen, z. B. `ollama pull llama3.1`.

Ist keiner verfügbar, wird Stufe 3 sauber übersprungen — Stufe 1, 2 und Oracle laufen weiter.

## Ausführen

**Experiment (Tabelle + Plot):**

```bash
uv run python scripts/run_experiment.py
# -> reports/results.csv und reports/auswahl_vergleich.png
```

**Interaktive Demo (zum Vorführen):**

```bash
uv run marimo edit notebooks/demo.py
```

**Tests:**

```bash
uv run pytest                 # alles (Daten-Tests werden ohne Parquet übersprungen)
uv run pytest -m "not data"   # nur datenfreie Tests (Metriken + LLM, laufen immer)
```

## Struktur

```
automl_ad/
  config.py      # Pfade, Spalten, Onset-Logik, Seeds
  data.py        # TEP laden, Run-Level-Split, Onset-Labeling (load_split, load_validation)
  detectors.py   # 4 klassische pyOD-Detektoren (ecod, iforest, ocsvm, pca) + Registry
  selection.py   # select_oracle / select_internal / select_llm (einheitliche API)
  pyod_engine.py # Stufe 3: pyODs ADEngine (investigate + validate, benchmark-gestützt)
  llm.py         # 'unter der Haube': Dataset-Profiling + Detektor-Steckbriefe + Provider (Claude|Ollama)
  hpo.py         # Optuna-HPO (klassisches AutoML-Werkzeug)
  metrics.py     # ROC-AUC, PR-AUC, F1, Detection-Delay, False-Alarm-Rate
  plots.py       # Vergleichsgrafiken
scripts/run_experiment.py
notebooks/demo.py
docs/storyline.md   # Foliengerüst (3-Stufen-Faden) für den Vortrag
```

## Literatur

- Bahri et al., *AutoML: state of the art with a focus on anomaly detection, challenges, and
  research directions*, Int. J. Data Science and Analytics.
- Chen et al., *PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection*,
  2024 (arXiv:2412.12154) — Grundidee für Stufe 3.
