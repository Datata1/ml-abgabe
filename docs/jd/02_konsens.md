# 02 — Konsens: Detektoren abstimmen lassen

> Zweites Thema von JD und das **Highlight**. Kernidee: Kein einzelnes Modell ist „die Wahrheit", aber
> die **Übereinstimmung vieler Modelle** ist ein robustes, **label-freies** Signal. Es gibt **zwei
> Spielarten**. Hält die [Narrativ-Regeln](00_narrativ.md) ein.

---

## Die Grundidee: Schwarmintelligenz statt Lösungsblatt

Wir haben keine Labels — also können wir keinen Detektor „gegen die Wahrheit" prüfen. Aber: Wenn
**mehrere unabhängige** Detektoren **dieselben** Punkte verdächtig finden, ist das ein starkes Indiz,
dass an diesen Punkten wirklich etwas ist. Ein **Außenseiter**, der ganz andere Punkte markiert, ist
meist der unzuverlässige. Diese „Weisheit der vielen" ist die Basis für **beide** folgenden Modi.

Technisch arbeiten wir mit **Rängen** statt roher Scores: Jeder Detektor bringt alle Zeitpunkte in
eine Reihenfolge („Punkt 7 am verdächtigsten, dann 12, …"). Ränge sind über Detektoren hinweg
**vergleichbar**, rohe Scores (unterschiedliche Skalen) nicht.

---

## Modus A — **Auswahl**: das zum Konsens zentralste *eine* Modell

Hier nutzen wir den Konsens, um **einen** Detektor auszuwählen, und benutzen dann **nur diesen**.
Code: [`selection.consensus_centrality`](../../automl_ad/selection.py).

**In vier Schritten:**

1. **Scores → Ränge:** Jeder der vier Detektoren scored `X_eval`; wir wandeln in Ränge (`rankdata`).
2. **Konsens-Rang bilden:** elementweiser **Mittelwert** aller Rang-Vektoren → „was die Mehrheit
   verdächtig findet".
3. **Model-Centrality messen:** für jeden Detektor die **Korrelation seiner Ränge mit dem
   Konsens** (`np.corrcoef`). Hoch = „stimmt mit dem Schwarm überein".
4. **Zentralstes Modell wählen:** `argmax` der Centrality.

```python
ranks     = {n: rankdata(scores[n]) for n in names}          # Schritt 1
consensus = np.mean([ranks[n] for n in names], axis=0)       # Schritt 2
centrality = {n: corrcoef(ranks[n], consensus)[0,1] for n}   # Schritt 3
best = max(centrality, key=centrality.get)                   # Schritt 4
```

**Ergebnis (caveated):**
> ⚠️ *ROC/AUC nur zur Illustration — real nicht verfügbar.* Konsens wählt **pca** (**0.883**) —
> praktisch auf Höhe der label-Obergrenze (knn, 0.890), **ohne einen einzigen Label**. Das ist
> die Kernaussage.

**Was man wirklich hat (label-frei):** die **Centrality-Werte selbst** (welcher Detektor ist
Mainstream, welcher Außenseiter) und das **Agreement** (wie einig ist der Schwarm überhaupt).
Hier konkret: die Centrality **entlarvt hdbscan als Außenseiter** (0.81 vs. 0.93–0.97 der
anderen) — und tatsächlich ist hdbscan mit 0.757 der schwächste Detektor. Das label-freie Signal
zeigt also in die richtige Richtung.

---

## Modus B — **Ensemble-Vorhersage**: der Konsens-Score *ist* die Vorhersage

Statt einen Detektor auszuwählen, kann man **alle** laufen lassen und ihren **kombinierten Score
selbst** als finale Anomalie-Vorhersage nehmen. Kein Detektor gewinnt — das **Ensemble** entscheidet.

Die Aggregation ist bewusst simpel — Scores **z-normalisieren**, dann kombinieren
([`selection.ensemble_consensus`](../../automl_ad/selection.py)):

| Methode | Aggregation |
|---------|-------------|
| `average` | Mittelwert (der Standard-Konsens) |
| `maximization` | punktweises Maximum (ein Detektor reicht für Alarm) |
| `median` | Median (robust gegen Ausreißer-Detektoren) |

Das entspricht exakt PyODs `pyod.models.combination` (`average`/`maximization`/`median`); wir
implementieren die drei One-Liner direkt, um die optionale `combo`-Abhängigkeit zu sparen (das Repo
soll offline im Vortrag laufen). Für „fertige", schwerere Ensembles bietet PyOD zusätzlich **SUOD**
und **LSCP** als eigene Detektoren (hier nicht nötig).

**Ergebnis (caveated):**
> ⚠️ *ROC/AUC nur zur Illustration.* Der gemittelte Ensemble-Score liefert eine **stabile**
> Vorhersage, die nicht von der (ungewissen) Wahl eines Einzeldetektors abhängt — der Preis ist, dass
> ein schwacher Außenseiter das Mittel „verwässert". Genau das passiert hier: average **0.834**,
> weil hdbscan (0.757) mit hineingemittelt wird — Modus A (auswählen statt mitteln) ist auf diesem
> Setup die bessere Konsens-Spielart.

---

## Modus A vs. Modus B — wann was?

| | Modus A (Auswahl) | Modus B (Ensemble-Vorhersage) |
|---|---|---|
| **Output** | *ein* gewählter Detektor | *ein* kombinierter Score-Vektor |
| **Laufzeit produktiv** | nur der gewählte Detektor läuft | alle Detektoren laufen dauerhaft |
| **Robustheit** | hängt am gewählten Modell | mittelt Fehler einzelner Modelle heraus |
| **Erklärbarkeit** | „wir nehmen pca, weil zentral" | „das Kollektiv sagt anomal" |
| **Code** | [`consensus_centrality`](../../automl_ad/selection.py) | [`ensemble_consensus`](../../automl_ad/selection.py) |

Beide sind **label-frei**. A ist billiger und liefert einen benennbaren Detektor; B ist robuster und
braucht keine Auswahlentscheidung. **Beide** tauchen später bei der ADEngine wieder auf: die bildet
intern genau so einen **Consensus** (Modus B) und meldet zusätzlich das **Agreement** als Vertrauensmaß.

---

## Das label-freie Vertrauensmaß: Agreement

Egal ob A oder B — wie sehr man dem Ergebnis trauen kann, verrät die **Übereinstimmung** der
Detektoren: die **mittlere paarweise Spearman-Korrelation** ihrer Ränge. Hoch = der Schwarm ist sich
einig (verlässlicher). Niedrig = die Detektoren widersprechen sich (Vorsicht). Das ist exakt das
Signal, das auch PyODs ADEngine als `agreement` meldet → Brücke zu `03`.

> *Eine andere, geometrische label-freie Detektor-Güte (EM/MV, Goix 2016) stellt **Achim** vor — nicht
> Teil deiner Folien.*

---

## Übergang zu Thema 3

> „Diese Konsens-Idee haben wir selbst gebaut. Aber es gibt sie **fertig** in PyOD — die **ADEngine**
> profiliert die Daten, wählt benchmark-gestützt Detektoren, bildet genau so einen Consensus und
> meldet label-freie Qualität. Mit optionaler LLM-Steuerung." → weiter in [03_adengine.md](03_adengine.md)
