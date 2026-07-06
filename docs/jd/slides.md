# Sprechzettel (JD) — Folienplan

Ableitung aus [00](00_narrativ.md)–[03](03_adengine.md). Visuals: PNGs aus
[`reports/figures/`](../../reports/figures/) (erzeugt via `make figures`, Quelle
[`automl_ad/figures.py`](../../automl_ad/figures.py)). **Regel:** jede ROC/AUC-Zahl laut ansagen
als „das, was man real **nicht** hätte".

**Zahlenstand (ein Lauf, ALLE 20 Fehlertypen, [`reports/results.csv`](../../reports/results.csv)):**
naiv (iforest) 0.826 · Konsens A (pca) 0.883 · Konsens B avg 0.834 · ADEngine 0.850 ·
ADEngine+LLM (LOF) 0.883 · Oracle (knn) 0.890. Einzeldetektoren: knn 0.890 · pca 0.883 ·
iforest 0.826 · hdbscan 0.757. Pro Fehlerart:
[`reports/results_per_fault.csv`](../../reports/results_per_fault.csv) — schwere Fehler 3/9/15
nur ~0.56–0.58; das agreement ist dort unauffällig (~0.73, mitten im Bereich der erkannten Fehler).

**Folien-Hinweis:** Titel + Kernaussage sind in die PNGs eingebrannt — auf der Folie keinen
eigenen Titel doppeln; Bild füllt die Folie.

---

### S1 — Titel & Leitfrage
- Outlier Detection ohne Labels: TEP, 52 Sensoren, Prozessindustrie.
- Kernproblem: im Betrieb gibt es keine Zielmetrik — keine ROC/AUC.
- Regel für den ganzen Vortrag: jede gezeigte ROC/AUC-Zahl ist „gespickt", nur Illustration.
- Roter Faden: naive Baseline → Konsens → ADEngine (+ LLM).

### S2 — Baseline: PyOD und die vier Kandidaten
- PyOD: einheitliches BaseDetector-Interface (fit / decision_function); 4 Kandidaten: knn, pca,
  hdbscan, iforest.
- Setup-Kasten (einmalig, gilt für alles Folgende): TEP, **alle 20 Fehlertypen**, Training nur
  auf Gutdaten-Läufen, Seed 0 (`config.SPLIT_KW`).

### S3 — Die vier Detektoren im Steckbrief (kurz, nicht zu detailliert)
- Je Modell eine Karte: Score-Landschaft auf Beispieldaten + Mechanik / Stärke / Schwäche.
- knn: Abstand zum k-nächsten Nachbarn — lokal, ADBench #4.
- pca: Rekonstruktionsfehler zur „Hauptebene" — nutzt lineare Sensor-Kopplung, sehr schnell.
- hdbscan: Dichte-Clustering (GLOSH) — Cluster beliebiger Form, aber empfindlich konfiguriert.
- iforest: zufällige Schnitte isolieren Anomalien — robuster Allrounder, ADBench #3.
- Visuals: `modell_knn.png` · `modell_pca.png` · `modell_hdbscan.png` · `modell_iforest.png`
  (frei kombinierbar, z. B. 2×2 oder eine Karte pro Folie).
- Sprechhilfe: die vier Score-Landschaften zeigen die **unterschiedlichen Weltbilder** der
  Modelle — lokale Distanz-Blasen (knn), glatte Ellipsen (pca), Dichte-Inseln (hdbscan),
  achsenparallele Blöcke (iforest). Genau deshalb streuen sie gleich auf TEP.

### S4 — Das Problem: ohne Labels ist die Wahl ein Blindflug
- Naiv = fester Detektor (der „Standard-Griff" iforest) — ohne Labels ist die Wahl Glückssache:
  hier nur Mittelfeld (0.826); blind hdbscan zu greifen wäre deutlich schlechter (0.757).
- Visual: `01_detektor_streuung.png` (AUC-Spread Δ=0.13, caveated).
- Kernsatz: „Schon die Detektorwahl streut stark — blind einen zu nehmen ist Glückssache."

### S5 — Anschauung: Score über die Zeit
- Ein Fehlerlauf: nach dem Fehler-Onset steigt der Anomalie-Score über die Schwelle
  (99%-Quantil des Normalbetriebs, wie im echten Monitoring) — das klassische
  Prozess-Monitoring-Bild (Anlage läuft → Störung → Alarm).
- Visual: `02_score_zeitreihe.png`.

### S6 — Pivot: Was hat man statt Labels? (Mechanik-Folie)
- Signal 1: **agreement** = mittlere paarweise **Spearman-Korrelation** der Detektor-Rankings.
- Spearman in einem Satz: Pearson auf **Rängen** statt Rohwerten — nötig, weil die Score-Skalen
  der Detektoren nicht vergleichbar sind; Ränge machen sie vergleichbar.
- Signal 2: das ADEngine-Qualitätsverdikt (kommt in S10).
- Abgrenzung: EM/MV (Goix 2016) ist ein drittes Signal — Achims Teil.

### S7 — Konsens · Modus A: das zentralste Modell
- Rezept: Scores → Ränge → Konsens = mittlerer Rang-Vektor → Centrality = Korrelation jedes
  Detektors zum Konsens → wähle das Maximum.
- Ergebnis: agreement 0.79, Wahl: pca (Centrality 0.97). Pointe: die Centrality **entlarvt
  hdbscan als Außenseiter** (0.81) — und tatsächlich ist er der schwächste Detektor. Das
  label-freie Signal zeigt in die richtige Richtung.
- Visual: `03_konsens_modusA.png` (Heatmap + Centrality-Balken).

### S8 — Konsens · Modus B: das Ensemble ist die Vorhersage
- Rezept: Scores je Detektor z-normieren, dann kombinieren (average / max / median, wie
  pyod.models.combination).
- Ergebnis: average 0.834 — stabil, aber der schwache Außenseiter (hdbscan) verwässert das
  Mittel. Ehrlicher Vergleich: hier ist Modus A (auswählen statt mitteln) die bessere Spielart.
- Visual: `04_konsens_modusB.png` (Kombinationen vs. Linie „bester Einzeldetektor", caveated).

### S9 — Differenzierung: wie verändert sich das Ergebnis mit der Fehlerart?
- Auswertung pro Fehlertyp (Gutdaten + jeweils ein Fehler): die Erkennung hängt stark an der
  Fehlerart — leichte Fehler nahezu perfekt (F1/F2/F6/F7 ≈ 1.0), Mittelfeld 0.63–0.99.
- Die „quasi unbeobachtbaren" Fehler 3/9/15 (TEP-Literatur) brechen auf ~0.56–0.58 ein — nahe
  Zufall. Nur leichte Fehler zu zeigen wäre zu optimistisch.
- Ehrliche Pointe: das label-freie agreement ist dort **unauffällig** (~0.73, mitten im Bereich
  der gut erkannten Fehler) — es unterscheidet die unsichtbaren Fehler nicht von den erkennbaren.
  Einigkeit misst Konsistenz, nicht Richtigkeit; genau deshalb gilt die Caveat-Regel.
- Visual: `05_pro_fehler.png` — zwei an der x-Achse ausgerichtete Panels (bewusst getrennt, da
  nicht vergleichbare Größen): oben ROC-AUC pro Fehler (schwere rot), unten agreement mit
  eigener Achse. Sprechhilfe: „oben bricht es ein, unten sieht man nichts Besonderes".

### S10 — ADEngine: Wie funktioniert sie?
- Ein Aufruf `investigate(X)`: profilieren → benchmark-gestützt wählen → Multi-Detektor-Consensus
  → label-freies Verdikt.
- Die „Benchmark" ist eine **vorberechnete Bestenliste** (ADBench: 57 Datensätze × 30 Algorithmen;
  TSB-AD für time_series) — Meta-Learning/Transfer, **kein Training auf TEP**.
- Pointe: Die Liste beginnt bei **#3** — die Plätze #1–#2 der Benchmark brauchen **Labels**
  (semi-/supervised) und stehen deshalb gar nicht in der label-freien Auswahl. Genau unser Thema.
- Visual: `06_benchmark.png` (Rangliste ab #3, mit Fußnote).

### S11 — ADEngine auf TEP: Ergebnis + Ehrlichkeit
- Verdikt = overall aus agreement / stability / separation; steuert die nächste Aktion (melden
  vs. iterieren). Belastbar v. a. agreement; separation ist zirkulär.
- Ergebnis (tabular): Consensus 0.850 — schlägt unseren einfachen Konsens A (0.883) **nicht**.
  Diskussionspunkt: mehr Maschinerie ist nicht automatisch besser.
- Ehrlich: der native time_series-Modus wählt Subsequenz-Detektoren — passt nicht zu TEPs
  anhaltenden Fehlern (Verdikt low, ~0.46); wir zeigen die Engine im tabular-Modus, wo sie fair
  funktioniert.
- Visual: `07_adengine_report.png` (Report-Card; Kacheln ①–④ = Pipeline-Reihenfolge:
  wählen → Konsens → Selbst-Bewertung → Spick-Kontrolle — in dieser Reihenfolge vorlesen).

### S12 — ADEngine · natives LLM-Routing
- Nur der Auswahlschritt (`plan_detection`) geht ans LLM — Prompt-Bau, Parsing, Validierung,
  Ausführung bleiben PyOD.
- Kernaussage 1, **sicher by design**: Structured Output + Validierung gegen die Wissensbasis +
  Regel-Fallback bei ungültiger Antwort — kann nichts Ungültiges ausführen.
- Kernaussage 2, **nicht-deterministisch**: die Wahl schwankt von Lauf zu Lauf → gute Wahl wird
  gecacht (`make cache`; `make cache-replay` rescort sie ohne Provider); aktueller Cache: LOF, 0.883.
- Visual: `08_llm_routing.png` (Flussdiagramm + Cache-Ergebnis).

### S13 — Fazit
- Visual: `09_fazit_vergleich.png` — naiv (iforest) 0.826 · Konsens A (pca) 0.883 ·
  Konsens B 0.834 · ADEngine 0.850 · ADEngine+LLM (LOF) 0.883 · Oracle (knn) 0.890.
- Vorsicht Aggregat (Munition gegen Rückfragen): Die 0.883 sind eine Mischung — auf den 17
  erkennbaren Fehlern 0.932, auf F3/9/15 nur 0.586; die drei stellen 15 % der Anomalie-Punkte
  (0.85 × 0.93 + 0.15 × 0.59 ≈ 0.88). Die Gesamtzahl versteckt die blinden Flecken — deshalb
  gehört die Differenzierung (S9) in jede ehrliche Auswertung. Gilt genauso für die
  Oracle-Linie: mehr als 0.89 geht auch mit Lösungsblatt nicht, die Grenze liegt in den Daten.
- Vier Botschaften: (1) Label-freier Konsens (Modus A) erreicht praktisch die Oracle-Obergrenze —
  die Obergrenze selbst liegt aber bei 0.89, weil die schweren Fehler für **alle** Strategien
  unsichtbar bleiben (S9). (2) Die fertige AutoML-Pipeline schlägt den einfachen Konsens nicht —
  die Konsens-Idee zählt, nicht die Maschinerie. (3) Label-freie Signale haben Grenzen:
  Einigkeit ≠ Richtigkeit. (4) Wir optimieren keine Metrik, wir zeigen ein Vorgehen für den
  labelfreien Fall.
