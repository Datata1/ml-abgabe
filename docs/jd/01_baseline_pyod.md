# 01 — Baseline: PyOD als Library & das Label-Problem

> Erstes Thema von JD. Ziel: Publikum abholen bei **was ist ein Detektor**, **wie benutzt man PyOD**,
> und **warum „einfach eins nehmen" ohne Labels ein Blindflug ist**. Hält die
> [Narrativ-Regeln](00_narrativ.md) ein (ROC/AUC nur als „was man nicht hat").

---

## Was ist PyOD?

**PyOD** ist die de-facto Standard-Library für **Outlier / Anomaly Detection** in Python: 60+
Detektoren unter **einer** einheitlichen Schnittstelle. Man kann Verfahren austauschen, ohne den
umgebenden Code zu ändern — genau das macht sie ideal als „Detektor-Baukasten".

### Das einheitliche `BaseDetector`-Interface

Jeder PyOD-Detektor spricht dieselbe Sprache (siehe [detectors.py](../../automl_ad/detectors.py#L24-L32)):

```python
det = SomeDetector(contamination=0.1)   # Konstruktor: contamination = erwarteter Anomalie-Anteil
det.fit(X_train)                         # nur auf "Gutdaten" lernen (unüberwacht)
scores = det.decision_function(X_eval)   # kontinuierlicher Score: HÖHER = anomaler
det.decision_scores_                     # Scores der Trainingsdaten
det.threshold_                           # aus contamination abgeleitete Schwelle
det.predict(X_eval)                      # 0/1-Label über threshold_
```

**Wichtig fürs Narrativ:** Das einzige, was der Detektor uns **von sich aus** gibt, ist ein **Score**
und (über `contamination`) ein **Threshold**. **Nichts davon sagt, ob die Wahl gut war** — dafür
bräuchte man Labels.

---

## Unsere vier Baseline-Detektoren

Wir bleiben bewusst schlank: vier klassische, dependency-leichte Detektoren (kein torch/Deep
Learning), registriert in [detectors.py](../../automl_ad/detectors.py):

| Detektor | Grundidee in einem Satz | Stärke | Schwäche |
|----------|--------------------------|--------|----------|
| **knn** | Score = Abstand zum k-nächsten Nachbarn: wer weit weg von allen liegt, ist anomal | einfach, lokal sensitiv, ADBench-Top (#4) | Distanzsuche kostet; hohe Dimensionen verwässern Distanzen |
| **pca** | Findet die normale „Hauptebene"; was weit von ihr rekonstruiert wird, ist anomal | nutzt **lineare Kopplung** (ideal für gekoppelte Sensoren), sehr schnell | nur lineare Struktur |
| **hdbscan** | Dichte-Clustering: Punkte in dünn besiedelten Regionen sind Ausreißer (GLOSH-Score) | findet Cluster beliebiger Form, keine Cluster-Anzahl nötig | empfindlich auf `min_cluster_size`; neue Punkte nur approximativ gescort |
| **iforest** | Isolation Forest: Anomalien lassen sich mit wenigen zufälligen Schnitten isolieren | robuster Allrounder, skaliert linear, ADBench-Top (#3) | achsenparallele Schnitte übersehen korrelierte Features |

**Für die Vorstellungs-Folie:** `make figures` erzeugt je Detektor eine **Steckbrief-Karte**
(`reports/figures/modell_*.png`): links die echte Score-Landschaft des Detektors auf
2D-Beispieldaten (derselbe Code wie in der Pipeline), rechts Mechanik/Stärke/Schwäche in drei
Stichpunkten — frei auf Folien kombinierbar.

**Registry-Nutzung:** `make_detector("pca")` erzeugt den Detektor über eine
Namens→Factory-Tabelle. Jede spätere Auswahlstrategie spricht nur gegen das Protokoll und ist damit
**detektor-unabhängig austauschbar** — das ist die Brücke zu Thema 2 und 3.

---

## Der naive Startpunkt: ein fixer Detektor

Die simpelste Strategie: **immer denselben** Detektor nehmen — z. B. `iforest`, den populären
„Standard-Griff". Das ist naiv, denn die Modellwahl ist **blind** — der Name steht hart im Code;
ohne Labels weiß man nicht, ob er passt.

> ⚠️ *Illustration mit TEP-Labels (real nicht verfügbar):* Die Detektoren streuen deutlich
> (Quelle [`reports/results.csv`](../../reports/results.csv)): knn **0.890**, pca **0.883**,
> iforest **0.826**, hdbscan **0.757**. „Nimm einfach iforest" ist also **Glückssache** — hier
> nur Mittelfeld, und der blinde Griff zu hdbscan wäre deutlich schlechter.

Das fixen wir über die **Konsens-Auswahl** (`02`/`03`).

Anschaulich im Notebook:

- **Score-Zeitreihe eines Fehlerlaufs**: der Score steigt nach dem Fehler-Onset über die Schwelle —
  genau das Prozessindustrie-Bild (Anlage läuft → Störung → Alarm).

---

## Was man OHNE Labels über Detektor-Güte sagen kann

Damit die Folie nicht mit „man weiß nichts" endet, verweisen wir auf das **label-freie** Ersatzsignal,
das dein Teil trägt:

- **Konsens-Agreement** (Vorschau `02`): Übereinstimmung der Detektoren als Vertrauensmaß — ganz
  ohne Labels.

*(Eine weitere, geometrische label-freie Güte — EM/MV — stellt **Achim** vor; hier nicht dein Thema.)*

---

## Übergang zu Thema 2

> „Ein fixer Detektor ist ein Blindflug. Aber ich habe ja **viele** Detektoren in PyOD — kann ich
> die nicht **zusammen** befragen, statt blind einen zu wählen? Genau das macht der *Konsens*."
> → weiter in [02_konsens.md](02_konsens.md)
