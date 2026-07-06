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

Weitere Begriffe: **Detektor** = ein OD-Modell (knn, iforest, …). **Score** = Anomaliewert (höher =
anomaler). **Threshold** = Schwelle aus `contamination`. **Consensus** = aggregierter Score/Rang über
mehrere Detektoren.

---

## Aufbau von JDs Teil (drei Themen)

1. **Baseline / PyOD als Library** (`01`) — Was ist ein Detektor, wie sieht die PyOD-Schnittstelle
   aus; der naive Startpunkt (fester Detektor); und warum „einfach eins nehmen" ohne Labels ein
   Blindflug ist.
2. **Konsens** (`02`) — Die elegante label-freie Antwort in **zwei Spielarten**: (A) das zum Konsens
   **zentralste einzelne** Modell wählen; (B) den **Konsens-Score eines Ensembles selbst** als
   Vorhersage nehmen.
3. **PyODs ADEngine** (`03`) — Die **fertige, native** AutoML-AD-Pipeline: profiliert, wählt
   benchmark-gestützt (ADBench), bildet Consensus, meldet label-freie Qualität — inklusive PyODs
   **eigener** LLM-Routing-Schicht (kein Eigenbau). Demo im `tabular`-Modus (dort funktioniert sie fair).

**Übergänge (roter Faden):**
`01` stellt das Problem (blinde Einzelwahl ohne Labels) → `02` löst es mit Konsens über mehrere
Detektoren → `03` zeigt dieselbe Idee **industrialisiert** in einer Library + optional LLM.

---

## Ehrliche Kernaussagen (die wir vertreten)

- Label-freie **Konsens**-Auswahl erreicht auf TEP praktisch die label-Obergrenze — **ohne Labels**.
- Das schwergewichtige **ADEngine** ist die *ehrliche, reale* AutoML-Variante; dass sie hier den
  einfachen Konsens **nicht** schlägt, ist ein guter Diskussionspunkt (mehr Maschinerie ≠ besser).
- **Differenziert nach Fehlerart** (alle 20 Fehler, nicht nur die leichten — sonst zu optimistisch):
  leichte Fehler ≈ 1.0, die „quasi unbeobachtbaren" Fehler **3/9/15** nur ~0.56–0.58. Und: das
  label-freie ``agreement`` ist dort **unauffällig** (~0.73, mitten im Bereich der gut erkannten
  Fehler) — **Einigkeit ≠ Richtigkeit**; der Schwarm kann einig und trotzdem blind sein. Das ist
  die ehrliche Grenze der label-freien Signale.
- Wir **optimieren keine Metrik**, wir zeigen ein **Vorgehen** für den labelfreien Fall.

> Zahlen (nur zur internen Referenz, stets caveaten; ein Lauf über **alle 20 Fehler**):
> naiv (iforest) 0.826 · Konsens A (pca) 0.883 · Konsens B (avg) 0.834 · ADEngine 0.850 ·
> ADEngine+LLM (LOF) 0.883 · label-Oracle (knn) 0.890. Pro Fehlerart:
> [`reports/results_per_fault.csv`](../../reports/results_per_fault.csv). Quelle:
> [`reports/results.csv`](../../reports/results.csv).
