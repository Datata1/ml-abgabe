# JD-Vortragsteil — Dokumentation

Outlier Detection **ohne Labels** (TEP): Baseline → Konsens → ADEngine. Diese Docs legen das
**Narrativ** fest (Phase A), bevor Slides & marimo-Notebook entstehen (Phase B).

**Zuerst lesen:** [00_narrativ.md](00_narrativ.md) — Leitfrage, ROC/AUC-Caveat-Regel, Glossar.

| Doc | Thema |
|-----|-------|
| [00_narrativ.md](00_narrativ.md) | Verfassung: no-labels-Framing, label-freie Signale, Begriffe, roter Faden |
| [01_baseline_pyod.md](01_baseline_pyod.md) | PyOD als Library, `BaseDetector`, 4 Detektoren, naiver Startpunkt (fixer Detektor = Blindflug) |
| [02_konsens.md](02_konsens.md) | Konsens in **zwei Modi**: (A) zentralstes Modell wählen, (B) Ensemble-Konsens als Vorhersage |
| [03_adengine.md](03_adengine.md) | PyODs **native** ADEngine (Demo `tabular`; `time_series` passt nicht zu TEP) + **native** LLM-Schicht (`_llm.py`) + label-freies Verdikt |

**Kernbotschaften:** (1) Ohne einen einzigen Label wählt/kombiniert man Detektoren über **Konsens
und Agreement** (+ ADEngine-Verdikt) nahezu so gut wie mit dem „Lösungsblatt". ROC/AUC erscheint nur
als klar markierter „Wenn wir spicken würden"-Block. (2) Die AutoML-Schicht ist die **ADEngine** —
sie industrialisiert genau die Konsens-Idee.
