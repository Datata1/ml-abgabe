# Sprechzettel (JD) — Folienplan

Ableitung aus [00](00_narrativ.md)–[03](03_adengine.md). Visuals: PNGs aus
[`reports/figures/`](../../reports/figures/) (erzeugt via `make figures`, Quelle
[`automl_ad/figures.py`](../../automl_ad/figures.py)). **Regel:** jede ROC/AUC-Zahl laut ansagen
als „das, was man real **nicht** hätte".

**Zahlenstand (ein Lauf, ALLE 20 Fehlertypen, [`reports/results.csv`](../../reports/results.csv)):**
naiv i.i.d. (ecod) 0.805 · Konsens A (pca) 0.903 · Konsens B avg 0.903 · ADEngine 0.850 ·
ADEngine+LLM (LOF) 0.883 · Oracle 0.910. Pro Fehlerart:
[`reports/results_per_fault.csv`](../../reports/results_per_fault.csv) — schwere Fehler 3/9/15
nur ~0.59–0.61, agreement bleibt dort trotzdem ~0.92.

**Folien-Hinweis:** Titel + Kernaussage sind in die PNGs eingebrannt — auf der Folie keinen
eigenen Titel doppeln; Bild füllt die Folie.

---

### S1 — Titel & Leitfrage
- Outlier Detection ohne Labels auf einer Zeitreihe: TEP, 52 Sensoren, Prozessindustrie.
- Kernproblem: im Betrieb gibt es keine Zielmetrik — keine ROC/AUC.
- Regel für den ganzen Vortrag: jede gezeigte ROC/AUC-Zahl ist „gespickt", nur Illustration.
- Roter Faden: naive Baseline → zeitbewusst → Konsens → ADEngine (+ LLM).

### S2 — Baseline: PyOD und der naive Blick
- PyOD: einheitliches BaseDetector-Interface (fit / decision_function); 4 Kandidaten: ecod,
  iforest, ocsvm, pca.
- Naiv = fester Detektor, zeilenweise (i.i.d.) — ohne Labels ist die Wahl Glückssache.
- Visual: `01_detektor_streuung.png` (AUC-Spread, caveated).
- Kernsatz: „Schon die Detektorwahl streut stark — und die Zeitstruktur ignoriert dieser Blick
  komplett."

### S3 — Mechanismus: der Fenster-Adapter
- Drei Schritte: pro Lauf gleitende Fenster → Detektor scort jede Fenster-Matrix → Scores per
  max auf Zeitstempel zurückmappen.
- Pro simulationRun gefenstert (keine Fenster über Run-Grenzen); Gut-Fenster aller Läufe für den
  Fit gepoolt.
- Einordnung: Adapter, kein neues Modell, kein AutoML — macht jeden PyOD-Detektor zeitbewusst.
- Setup-Kasten (einmalig, gilt für alles Folgende): TEP, **alle 20 Fehlertypen**, w=30, step=5,
  max-Aggregation, Seed 0 (`config.SPLIT_KW`).
- Visual: `02_fensterung.png`.

### S4 — Effekt: zeitbewusst schlägt i.i.d.
- Gleiche 4 Detektoren, nur gefenstert: Δ ROC/AUC durchweg positiv.
- Visual: `03_iid_vs_zeitbewusst.png`; `04_score_zeitreihe.png` (Onset → Alarm; Schwelle =
  99%-Quantil des Normalbetriebs, wie im echten Monitoring) als Anschauung oder Backup.
- Kernsatz: „Die Zeitstruktur zu nutzen hilft messbar — das klassische Prozess-Monitoring-Bild."

### S5 — Pivot: Was hat man statt Labels? (Mechanik-Folie)
- Signal 1: **agreement** = mittlere paarweise **Spearman-Korrelation** der Detektor-Rankings.
- Spearman in einem Satz: Pearson auf **Rängen** statt Rohwerten — nötig, weil die Score-Skalen
  der Detektoren nicht vergleichbar sind; Ränge machen sie vergleichbar.
- Signal 2: das ADEngine-Qualitätsverdikt (kommt in S9).
- Abgrenzung: EM/MV (Goix 2016) ist ein drittes Signal — Achims Teil.

### S6 — Konsens · Modus A: das zentralste Modell
- Rezept: Scores → Ränge → Konsens = mittlerer Rang-Vektor → Centrality = Korrelation jedes
  Detektors zum Konsens → wähle das Maximum.
- Ergebnis: agreement 0.87 (Schwarm einig), Wahl: pca (Centrality am höchsten).
- Enge Balken sind kein Makel, sondern die Einigkeit selbst — sie macht die Wahl robust.
- Visual: `05_konsens_modusA.png` (Heatmap + Centrality-Balken).

### S7 — Konsens · Modus B: das Ensemble ist die Vorhersage
- Rezept: Scores je Detektor z-normieren, dann kombinieren (average / max / median, wie
  pyod.models.combination).
- Ergebnis: average 0.903 — praktisch Niveau des besten Einzeldetektors, ohne wissen zu müssen,
  welcher das ist.
- Visual: `06_konsens_modusB.png` (Kombinationen vs. Linie „bester Einzeldetektor", caveated).

### S8 — Differenzierung: wie verändert sich das Ergebnis mit der Fehlerart?
- Auswertung pro Fehlertyp (Gutdaten + jeweils ein Fehler): die Erkennung hängt stark an der
  Fehlerart — leichte Fehler nahezu perfekt (F1/F2/F6 ≈ 1.0), Mittelfeld 0.77–0.99.
- Die „quasi unbeobachtbaren" Fehler 3/9/15 (TEP-Literatur) brechen auf ~0.59–0.61 ein — nahe
  Zufall. Nur leichte Fehler zu zeigen wäre zu optimistisch.
- Ehrliche Pointe: das label-freie agreement bleibt dort hoch (~0.92) — der Schwarm ist sich
  einig **und trotzdem blind**. Einigkeit misst Konsistenz, nicht Richtigkeit; genau deshalb
  gilt die Caveat-Regel.
- Visual: `11_pro_fehler.png` — zwei an der x-Achse ausgerichtete Panels (bewusst getrennt, da
  nicht vergleichbare Größen): oben ROC-AUC pro Fehler (schwere rot), unten agreement mit
  eigener Achse. Sprechhilfe: „oben bricht es ein, unten bleibt es flach".

### S9 — ADEngine: Wie funktioniert sie?
- Ein Aufruf `investigate(X)`: profilieren → benchmark-gestützt wählen → Multi-Detektor-Consensus
  → label-freies Verdikt.
- Die „Benchmark" ist eine **vorberechnete Bestenliste** (ADBench: 57 Datensätze × 30 Algorithmen;
  TSB-AD für time_series) — Meta-Learning/Transfer, **kein Training auf TEP**.
- Pointe: Die Liste beginnt bei **#3** — die Plätze #1–#2 der Benchmark brauchen **Labels**
  (semi-/supervised) und stehen deshalb gar nicht in der label-freien Auswahl. Genau unser Thema.
- Visual: `07_benchmark.png` (Rangliste ab #3, mit Fußnote).

### S10 — ADEngine auf TEP: Ergebnis + Ehrlichkeit
- Verdikt = overall aus agreement / stability / separation; steuert die nächste Aktion (melden
  vs. iterieren). Belastbar v. a. agreement; separation ist zirkulär.
- Ergebnis (tabular): Consensus 0.850 — schlägt unseren einfachen Konsens (0.903) **nicht**.
  Diskussionspunkt: mehr Maschinerie ist nicht automatisch besser.
- Ehrlich: der native time_series-Modus wählt Subsequenz-Detektoren — passt nicht zu TEPs
  anhaltenden Fehlern; die richtige Zeitbewusstheit bleibt die Fensterung aus S3.
- Visual: `08_adengine_report.png` (Report-Card; Kacheln ①–④ = Pipeline-Reihenfolge:
  wählen → Konsens → Selbst-Bewertung → Spick-Kontrolle — in dieser Reihenfolge vorlesen).

### S11 — ADEngine · natives LLM-Routing
- Nur der Auswahlschritt (`plan_detection`) geht ans LLM — Prompt-Bau, Parsing, Validierung,
  Ausführung bleiben PyOD.
- Kernaussage 1, **sicher by design**: Structured Output + Validierung gegen die Wissensbasis +
  Regel-Fallback bei ungültiger Antwort — kann nichts Ungültiges ausführen.
- Kernaussage 2, **nicht-deterministisch**: die Wahl schwankt von Lauf zu Lauf → gute Wahl wird
  gecacht (`make cache`; `make cache-replay` rescort sie ohne Provider); aktueller Cache: LOF, 0.883.
- Visual: `09_llm_routing.png` (Flussdiagramm + Cache-Ergebnis).

### S12 — Fazit
- Visual: `10_fazit_vergleich.png` — naiv i.i.d. (ecod) 0.805 · Konsens A (pca) 0.903 ·
  Konsens B 0.903 · ADEngine 0.850 · ADEngine+LLM (LOF) 0.883 · Oracle 0.910.
- Vorsicht Aggregat (Munition gegen Rückfragen): Die 0.903 sind eine Mischung — auf den 17
  erkennbaren Fehlern 0.950, auf F3/9/15 nur 0.599; die drei stellen 15 % der Anomalie-Punkte
  (0.85 × 0.95 + 0.15 × 0.60 ≈ 0.90). Die Gesamtzahl versteckt die blinden Flecken — deshalb
  gehört die Differenzierung (S8) in jede ehrliche Auswertung. Gilt genauso für die
  Oracle-Linie: mehr als 0.91 geht auch mit Lösungsblatt nicht, die Grenze liegt in den Daten.
- Vier Botschaften: (1) Label-freier Konsens erreicht praktisch die Oracle-Obergrenze — die
  Obergrenze selbst liegt aber bei 0.91, weil die schweren Fehler für **alle** Strategien
  unsichtbar bleiben (S8). (2) Die fertige AutoML-Pipeline schlägt den einfachen Konsens nicht —
  die Konsens-Idee zählt, nicht die Maschinerie. (3) Label-freie Signale haben Grenzen:
  Einigkeit ≠ Richtigkeit. (4) Wir optimieren keine Metrik, wir zeigen ein Vorgehen für den
  labelfreien Fall.
