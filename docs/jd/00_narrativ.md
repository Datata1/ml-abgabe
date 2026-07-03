# 00 — Narrativ & Verfassung (JD-Teil)

> Dies ist die **gemeinsame Grundlage** für JDs Vortragsteil. Alle anderen Docs (`01`–`04`) und
> später die Slides + das marimo-Notebook richten sich hiernach. Zweck: **erst das Narrativ
> festzurren**, bevor Code entsteht.

---

## Die eine Leitfrage

> **Was tut man in der Outlier Detection, wenn man keine Labels hat?**

Das ist keine akademische Spielerei, sondern **Industrie-Realität** (Prozessindustrie, Anlagen-
überwachung): Man hat einen Strom von Sensordaten und will „unnormale" Zustände finden — aber
**niemand hat die Daten vorher als normal/anomal markiert**. Es gibt also **keine ROC/AUC-Kurve**,
die einem sagt, wie gut ein Modell die Ausreißer trifft. Genau dieses Fehlen einer Zielmetrik ist das
Kernproblem, um das sich JDs ganzer Teil dreht.

### Und: wir haben eine **Zeitreihe**

TEP ist kein statischer Tabellen-Datensatz, sondern eine **Zeitreihe** von 52 Sensoren. Ein einzelner
Messpunkt kann für sich normal aussehen und erst **im zeitlichen Verlauf** (Drift, Trend, veränderte
Dynamik) anomal sein. Deshalb ist unser Ansatz durchgehend **zeitbewusst**: Wir betrachten nicht
einzelne Zeilen isoliert, sondern **gleitende Zeitfenster**. Der rein zeilenweise („i.i.d.") Blick
kommt nur als **naiver Startpunkt** vor — danach sind alle Detektoren über den **TimeSeriesOD-Fenster-
Adapter** zeitbewusst (siehe [01](01_baseline_pyod.md), Abschnitt „Wie funktioniert Time-Series-
Awareness?"). **Wichtig:** dieser Adapter ist **kein AutoML** — er macht *einen* Detektor
zeitreihen-fähig; die AutoML-Schicht ist die **ADEngine** ([03](03_adengine.md)).

---

## Die verbindliche ROC/AUC-Regel (gilt auf JEDER Folie)

Unser Datensatz — der **Tennessee-Eastman-Prozess (TEP)** — *hat* gelabelte Anomalien. Das ist ein
**Glücksfall für die Evaluation**, aber ein **didaktisches Risiko**: Es verleitet dazu, Modelle nach
ROC/AUC zu ranken, als hätte man diese Zahl auch im echten Betrieb.

**Regel:** Immer wenn wir ROC/AUC (oder irgendeine label-basierte Zahl) zeigen, sagen wir **explizit
dazu**:

> „Diese Zahl gibt es im echten unüberwachten Betrieb **nicht**. Wir zeigen sie nur, um im
> Nachhinein zu **illustrieren**, was die label-freien Verfahren *ohne* diese Information erreicht
> haben."

Praktisch heißt das: ROC/AUC lebt nur in einem **klar markierten „Wenn wir spicken würden"-Block**.
Der Rest der Folie/Zelle steht auf label-freien Signalen.

---

## Was man STATT Labels wirklich hat (die label-freien Signale)

Diese Signale sind der eigentliche Inhalt **deines** Teils — sie brauchen **keine** Wahrheit:

| Signal | Idee in einem Satz | Woher im Repo |
|--------|--------------------|---------------|
| **Konsens-Agreement** | Stimmen viele Detektoren überein, ist das Ergebnis vertrauenswürdiger (Spearman-Korrelation der Ränge). | [`selection.agreement`](../../automl_ad/selection.py); ADEngine `agreement` |
| **ADEngine-Qualitätsverdikt** | PyOD bewertet ein Ergebnis selbst über *separation / agreement / stability* → `high/medium/low`. | ADEngine `analyze`/`quality` |

> **Nicht dein Thema:** interne Metriken **EM / MV** (Goix 2016) sind ein *weiteres* label-freies
> Signal, das **Achim** vorstellt — hier bewusst ausgeklammert (Code liegt in `internal_metrics.py`).

**Kernbotschaft des ganzen Teils:** Ohne einen einzigen Label kann man mit diesen Signalen ein
Modell **nahezu so gut** auswählen wie mit dem „Lösungsblatt".

---

## Begriffs-Glossar

- **Konsens** (JDs Thema 2): **label-frei**. Viele Modelle „stimmen ab"; man nutzt ihre
  Übereinstimmung. Im Code: [`consensus_centrality`](../../automl_ad/selection.py) (zentralstes Modell)
  und [`ensemble_consensus`](../../automl_ad/selection.py) (Ensemble-Score als Vorhersage).
- **Oracle / Obergrenze** ([`oracle_best`](../../automl_ad/selection.py)): **braucht Labels**, wählt
  mit dem Lösungsblatt den besten Detektor. Nur **Referenzlinie** („besser geht's nicht"), im echten
  Betrieb **nicht verfügbar** — gehört in den „Wenn wir spicken würden"-Block.

Weitere Begriffe: **Detektor** = ein OD-Modell (ecod, iforest, …). **Score** = Anomaliewert (höher =
anomaler). **Threshold** = Schwelle aus `contamination`. **Consensus** = aggregierter Score/Rang über
mehrere Detektoren.

---

## Aufbau von JDs Teil (drei Themen)

1. **Baseline / PyOD als Library + Time-Series-Awareness** (`01`) — Was ist ein Detektor, wie sieht
   die PyOD-Schnittstelle aus; der naive i.i.d.-Blick; **wie** man Detektoren via Fenster-Adapter
   zeitbewusst macht (eigene Mechanismus-Erklärung); und warum „einfach eins nehmen" ohne Labels ein
   Blindflug ist.
2. **Konsens** (`02`) — Die elegante label-freie Antwort in **zwei Spielarten**, angewandt auf die
   **zeitbewussten** Detektoren: (A) das zum Konsens **zentralste einzelne** Modell wählen;
   (B) den **Konsens-Score eines Ensembles selbst** als Vorhersage nehmen.
3. **PyODs ADEngine** (`03`) — Die **fertige, native** AutoML-AD-Pipeline: profiliert, wählt
   benchmark-gestützt (ADBench), bildet Consensus, meldet label-freie Qualität — inklusive PyODs
   **eigener** LLM-Routing-Schicht (kein Eigenbau). Demo im `tabular`-Modus (dort funktioniert sie fair).

> **Time-Series-Awareness** zieht sich durch, ist aber **kein eigenes Thema**: In `01` machen wir die
> Baselines über den Fenster-Adapter zeitbewusst. PyODs ADEngine erkennt den `data_type` sogar selbst
> und hätte einen `time_series`-Modus — der wählt aber **Subsequenz**-Detektoren, die zu TEPs
> **anhaltenden** Fehlern nicht passen. Die richtige Zeitbewusstheit für TEP ist die **Fensterung** aus `01`.

**Übergänge (roter Faden):**
`01` macht Detektoren zeitbewusst und stellt das Problem (blinde Einzelwahl) → `02` löst es mit
Konsens über die zeitbewussten Detektoren → `03` zeigt dieselbe Idee **industrialisiert** in einer
Library (nativer TS-Modus) + optional LLM.

---

## Ehrliche Kernaussagen (die wir vertreten)

- Label-freie **Konsens**-Auswahl erreicht auf TEP praktisch die label-Obergrenze — **ohne Labels**.
- Das schwergewichtige **ADEngine** ist die *ehrliche, reale* AutoML-Variante; dass sie hier den
  einfachen Konsens **nicht** schlägt, ist ein guter Diskussionspunkt (mehr Maschinerie ≠ besser).
- **Differenziert nach Fehlerart** (alle 20 Fehler, nicht nur die leichten — sonst zu optimistisch):
  leichte Fehler ≈ 1.0, die „quasi unbeobachtbaren" Fehler **3/9/15** nur ~0.59–0.61. Und: das
  label-freie ``agreement`` bleibt dort hoch (~0.92) — **Einigkeit ≠ Richtigkeit**; der Schwarm
  kann einig und trotzdem blind sein. Das ist die ehrliche Grenze der label-freien Signale.
- Wir **optimieren keine Metrik**, wir zeigen ein **Vorgehen** für den labelfreien Fall.

> Zahlen (nur zur internen Referenz, stets caveaten; ein Lauf über **alle 20 Fehler**):
> naiv i.i.d. (ecod) 0.805 · Konsens A (pca) 0.903 · Konsens B (avg) 0.903 · ADEngine 0.850 ·
> ADEngine+LLM (LOF) 0.883 · label-Oracle (ocsvm) 0.910. Pro Fehlerart:
> [`reports/results_per_fault.csv`](../../reports/results_per_fault.csv). Quelle:
> [`reports/results.csv`](../../reports/results.csv).
