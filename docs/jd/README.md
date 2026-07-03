# JD-Vortragsteil — Dokumentation

Outlier Detection **ohne Labels** auf einer **Zeitreihe** (TEP): Baseline (zeitbewusst) → Konsens →
ADEngine. Diese Docs legen das **Narrativ** fest (Phase A), bevor Slides & marimo-Notebook entstehen
(Phase B).

**Zuerst lesen:** [00_narrativ.md](00_narrativ.md) — Leitfrage, ROC/AUC-Caveat-Regel, Glossar.

| Doc | Thema |
|-----|-------|
| [00_narrativ.md](00_narrativ.md) | Verfassung: no-labels-Framing, label-freie Signale, „wir haben eine Zeitreihe", Begriffe, roter Faden |
| [01_baseline_pyod.md](01_baseline_pyod.md) | PyOD als Library, `BaseDetector`, 4 Detektoren, naiver i.i.d.-Start **und** die Mechanismus-Erklärung „Wie funktioniert Time-Series-Awareness?" (Fenster-Adapter) |
| [02_konsens.md](02_konsens.md) | Konsens (auf zeitbewussten Detektoren) in **zwei Modi**: (A) zentralstes Modell wählen, (B) Ensemble-Konsens als Vorhersage |
| [03_adengine.md](03_adengine.md) | PyODs **native** ADEngine (Demo `tabular`; `time_series` passt nicht zu TEP) + **native** LLM-Schicht (`_llm.py`) + label-freies Verdikt |

**Kernbotschaften:** (1) Da TEP eine Zeitreihe ist, nutzen wir Detektoren **durchgängig zeitbewusst**
(Fenster-Adapter); der i.i.d.-Blick ist nur der naive Startpunkt. (2) Ohne einen einzigen Label
wählt/kombiniert man Detektoren über **Konsens und Agreement** (+ ADEngine-Verdikt) nahezu so gut
wie mit dem „Lösungsblatt". ROC/AUC erscheint nur als klar markierter „Wenn wir spicken würden"-Block.
(3) `TimeSeriesOD` ist ein **Adapter, kein AutoML** — die AutoML-Schicht ist die **ADEngine**.
