# AutoML für Anomalieerkennung ohne Labels (Tennessee-Eastman-Prozess)

Leitfrage: **Was tut man in der Outlier Detection, wenn man keine Labels hat?** — die Industrie-
Realität: man hat **keine** ROC/AUC, die zeigt, wie gut ein Modell Ausreißer trifft. Datensatz:
Tennessee-Eastman-Prozess (52 Sensoren, 20 Fehlertypen).

> ⚠️ TEP *hat* gelabelte Anomalien. ROC/AUC nutzen wir **nur zur Illustration** dessen, was man
> unüberwacht eben **nicht** hätte. Die real nutzbaren Signale sind **label-frei**: Konsens-Agreement
> und das ADEngine-Qualitätsverdikt.

## Roter Faden

1. **Baseline / PyOD** — 4 klassische Detektoren; blinde Einzelwahl ohne Labels ist Glückssache.
2. **Konsens** (label-frei, `automl_ad/selection.py`) — **Modus A** zentralstes Modell
   (`consensus_centrality`) **oder** **Modus B** Ensemble-Konsens als Vorhersage (`ensemble_consensus`).
3. **PyOD ADEngine** (`automl_ad/pyod_engine.py`) — native, benchmark-gestützte AutoML-AD
   (`data_type='tabular'`), inkl. **PyOD-eigenem** LLM-Routing; wir liefern nur den Provider-
   Transport (`automl_ad/llm.py`).

**Maßgebliche Doku:** [`docs/jd/`](docs/jd/) (00–03 + `slides.md`).

## Setup

```bash
uv sync
```

TEP-Parquet-Dateien nach `data/` legen (`TEP_FaultFree_Training/Testing`, `TEP_Faulty_Training/Testing`;
Spalten `faultNumber, simulationRun, sample, xmeas_1..41, xmv_1..11`; Quelle: Rieth et al., Harvard
Dataverse).

**Optional (nur für das native LLM-Routing):** `.env` aus `.env.example` — `ANTHROPIC_API_KEY` (Claude)
oder lokales Ollama (`ollama pull llama3.1`). Ohne Provider läuft alles außer dem LLM-Routing.

## Ausführen

Am einfachsten über das Makefile (`make help` zeigt alle Befehle):

```bash
make figures      # alle Grafiken als PNG -> reports/figures/ (fertig für die Folien)
make cache        # VOR dem Vortrag: LLM-Routing cachen -> reports/llm_cache.json (Provider nötig)
make present      # Notebook im Präsentationsmodus (read-only App)
make notebook     # Notebook bearbeiten (Code + Zellen)
make experiment   # reproduzierbare Zahlen -> reports/results.csv
make test         # schnelle, datenfreie Tests
```

**Für die Folien:** `make figures` schreibt 13 slide-fertige PNGs (16:9, weiß, selbst-erklärend) nach
`reports/figures/` — 9 nummerierte Pipeline-Grafiken + 4 Modell-Steckbriefe (`modell_*.png`, frei
kombinierbar) — direkt in PowerPoint ziehen. Dieselben Grafiken zeigt `make present` live.

## Modul-Landkarte

```
automl_ad/
  config.py         Pfade, Spalten (52 Features), Onset-Logik, Seeds
  data.py           TEP laden, Run-Level-Split, Onset-Labeling (load_split → Split)
  detectors.py      4 PyOD-Detektoren (knn, pca, hdbscan, iforest) + Registry
  selection.py      Score-Dict-API: consensus_centrality (A), ensemble_consensus (B),
                    agreement, oracle_best (+ Wrapper select_internal/select_oracle)
  internal_metrics.py  label-freie Detektor-Güte EM/MV (Goix 2016) — Achims Vortragsteil
  pyod_engine.py    PyOD ADEngine (run_engine, run_engine_llm_routed, benchmark_ranking)
  llm.py            Provider-Transport (Claude|Ollama) für PyODs plan_detection(llm_client=…)
  metrics.py        ROC-AUC, PR-AUC, F1, Detection-Delay, False-Alarm-Rate
  figures.py        slide-fertige Grafiken (fig_*), 16:9/weiß/selbst-erklärend
scripts/            make_figures.py (-> reports/figures/), run_experiment.py, precompute.py
notebooks/          jd_praesentation.py   (grafik-zentriert)
reports/figures/    13 PNGs für die Folien (9 Pipeline + 4 Modell-Steckbriefe)
docs/jd/            00_narrativ · 01_baseline · 02_konsens · 03_adengine · slides
```

## Literatur

- Bahri et al., *AutoML: state of the art with a focus on anomaly detection, challenges, and
  research directions*, Int. J. Data Science and Analytics.
- Chen et al., *PyOD 2: A Python Library for Outlier Detection with LLM-powered Model Selection*,
  2024 (arXiv:2412.12154).
- Goix, *How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?*, 2016
  (arXiv:1607.01152) — EM/MV.
