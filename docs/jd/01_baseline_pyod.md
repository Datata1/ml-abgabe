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
Learning), registriert in [detectors.py](../../automl_ad/detectors.py#L62-L67):

| Detektor | Grundidee in einem Satz | Stärke | Schwäche |
|----------|--------------------------|--------|----------|
| **ecod** | Rein statistisch: wie unwahrscheinlich ist jeder Messwert laut empirischer Verteilung? (parameterfrei) | schnell, robust, keine Tuning-Knöpfe | ignoriert **Korrelationen** zwischen Features |
| **iforest** | Isolation Forest: Anomalien lassen sich mit wenigen zufälligen Schnitten isolieren | Allrounder, skaliert gut | schwach bei feinen Dichte-Anomalien |
| **pca** | Findet die normale „Hauptebene"; was weit von ihr rekonstruiert wird, ist anomal | nutzt **lineare Kopplung** (ideal für gekoppelte Sensoren) | nur lineare Struktur |
| **ocsvm** | Zieht eine (krumme) Hülle um die Normaldaten; außerhalb = anomal | nichtlineare Grenzen | langsam, empfindlich bei `nu`/`gamma` |

**Registry-Nutzung:** `make_detector("pca")` erzeugt den Detektor über eine
Namens→Factory-Tabelle. Jede spätere Auswahlstrategie spricht nur gegen das Protokoll und ist damit
**detektor-unabhängig austauschbar** — das ist die Brücke zu Thema 2 und 3.

---

## Der naive Startpunkt: ein fixer Detektor, zeilenweise (i.i.d.)

Die simpelste Strategie: **immer denselben** Detektor nehmen (z. B. `ecod`) und ihn **zeilenweise**
auf jeden Messzeitpunkt anwenden. Das ist doppelt naiv:

1. **Blinde Modellwahl** — der Name steht hart im Code; ohne Labels weiß man nicht, ob er passt.
2. **Blind gegenüber der Zeit** — jede Zeile wird **isoliert** (i.i.d.) betrachtet, die zeitliche
   Struktur (Fenster, Verlauf) wird ignoriert.

> ⚠️ *Illustration mit TEP-Labels (real nicht verfügbar):* Schon der zeilenweise Blick streut stark
> je nach Detektor (Quelle [`reports/results.csv`](../../reports/results.csv)): ecod **0.767**,
> iforest ~0.79, pca **0.845**, ocsvm ~0.846. „Nimm einfach ecod" ist also **Glückssache** — hier
> sogar Pech.

Beides fixen wir: die **Modellwahl** über Konsens (`02`/`03`), die **Zeit-Blindheit** ab jetzt über
den Fenster-Adapter.

---

## Wie funktioniert Time-Series-Awareness? (der Fenster-Adapter)

> **Das ist die zentrale Mechanismus-Folie.** Ziel: Publikum versteht, wie aus einem gewöhnlichen
> PyOD-Detektor ein **zeitbewusster** wird — ohne ein neues Modell zu erfinden.

PyOD liefert dafür **`TimeSeriesOD`** ([ts_od.py](../../.venv/lib/python3.13/site-packages/pyod/models/ts_od.py)):
einen **Adapter** (keine neue Lernmethode!), der **einen beliebigen** PyOD-Detektor zeitreihen-fähig
macht. In vier Schritten:

1. **Fenstern:** Aus der Sensor-Zeitreihe werden **überlappende Fenster** der Länge `window_size`
   geschnitten (Schrittweite `step`). Jedes Fenster der Form `(window_size × n_sensoren)` wird zu
   **einer** Zeile „flachgeklopft". → aus *Verlauf* wird ein *Merkmalsvektor*.
2. **Detektor auf Fenstern:** Der gewählte Detektor (ecod/iforest/…) lernt/scored nun auf der
   **Fenster-Matrix** statt auf Einzelpunkten. Er sieht damit **Dynamik**, nicht nur Momentaufnahmen.
3. **Zurückmappen:** Jeder Fenster-Score wird auf die enthaltenen Zeitstempel zurückgeführt und dort
   **aggregiert** (`max` = „ein verdächtiges Fenster reicht", oder `mean`).
4. **Ergebnis:** ein Anomalie-Score **pro Zeitstempel** — gleiche Schnittstelle wie vorher, nur eben
   **zeitbewusst**.

```
Sensor-Zeitreihe (T × 52)
        │  sliding_windows(window_size, step)
        ▼
Fenster-Matrix (n_fenster × window_size·52) ── Detektor.fit/score ──► Fenster-Scores
        │  map_scores_to_timestamps(max/mean)
        ▼
Score pro Zeitstempel (T)
```

**Zwei wichtige Hinweise (kommen im Notebook konkret vor):**
- **Multivariat:** alle 52 Sensoren stecken im Fenster — Kopplungen über Sensoren *und* Zeit werden
  erfasst.
- **Pro Lauf fenstern:** Fenster dürfen **keine Run-Grenzen** überschreiten. TEP besteht aus vielen
  Simulationsläufen; wir fenstern **je `simulationRun`** getrennt (sonst „klebt" das Ende eines Laufs
  an den Anfang des nächsten).

> **Merksatz:** `TimeSeriesOD` ist ein **Adapter**, **kein AutoML**. Er wählt kein Modell aus — das
> tut erst die ADEngine (`03`).

---

## Die zeitbewussten Baselines

Ab hier sind „unsere vier Detektoren" immer `TimeSeriesOD(detector="ecod"|"iforest"|"ocsvm"|"pca")`.
Anschaulich im Notebook:

- **Score-Zeitreihe eines Fehlerlaufs** ([`plots.score_timeseries`](../../automl_ad/plots.py)): der
  gefensterte Score steigt nach dem Fehler-Onset über die Schwelle — genau das Prozessindustrie-Bild
  (Anlage läuft → Störung → Alarm).
- **Score-Histogramm** normal vs. anomal (*nur zur Illustration mit Labels*) + Threshold-Linie.

---

## Was man OHNE Labels über Detektor-Güte sagen kann

Damit die Folie nicht mit „man weiß nichts" endet, verweisen wir auf das **label-freie** Ersatzsignal,
das dein Teil trägt:

- **Konsens-Agreement** (Vorschau `02`): Übereinstimmung der (zeitbewussten) Detektoren als
  Vertrauensmaß — ganz ohne Labels.

*(Eine weitere, geometrische label-freie Güte — EM/MV — stellt **Achim** vor; hier nicht dein Thema.)*

---

## Übergang zu Thema 2

> „Ein fixer Detektor ist ein Blindflug — auch der zeitbewusste. Aber ich habe ja **viele**
> (zeitbewusste) Detektoren in PyOD — kann ich die nicht **zusammen** befragen, statt blind einen zu
> wählen? Genau das macht der *Konsens*." → weiter in [02_konsens.md](02_konsens.md)
