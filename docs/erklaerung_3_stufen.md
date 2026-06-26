# Die drei Stufen einfach erklärt

Dieses Dokument erklärt in Alltagssprache, **was** jede Stufe tut und **wie** sie es tut — und
liest dann eure tatsächlichen Ergebnisse damit. Es ist als Begleittext zur Abgabe gedacht.

---

## Worum geht es überhaupt?

Wir haben einen Datensatz (den Tennessee-Eastman-Prozess: 52 Sensoren einer Chemieanlage) und
wollen **Anomalien** finden — also Zeitpunkte, an denen die Anlage nicht normal läuft.

Dafür gibt es viele fertige Verfahren („Detektoren") in der Bibliothek **pyOD**. Wir benutzen vier:

| Detektor | Grundidee in einem Satz |
|----------|--------------------------|
| **ecod** | rein statistisch: wie unwahrscheinlich ist jeder Messwert laut Verteilung? (parameterfrei) |
| **iforest** | „Isolation Forest": Anomalien lassen sich mit wenigen zufälligen Schnitten isolieren |
| **pca** | findet die normale „Hauptebene" der Daten; was weit daneben liegt, ist anomal |
| **ocsvm** | zieht eine (krumme) Hülle um die Normaldaten; außerhalb = anomal |

**Das Problem:** Welcher dieser vier ist für *unsere* Daten der beste? Im echten Betrieb haben wir
**keine Lösungen (Labels)**, an denen wir das prüfen könnten. Genau diese Auswahl-Frage löst AutoML
— und wir zeigen dafür drei Stufen.

### Kurz: Was bedeutet „ROC-AUC"?
Eine Schulnote für den Detektor zwischen 0.5 und 1.0:
- **1.0** = perfekt (trennt Normal und Anomalie fehlerfrei),
- **0.5** = reines Raten (Münzwurf).
Höher = besser. In euren Ergebnissen reicht es von 0.767 (ecod) bis 0.846 (ocsvm).

---

## Die Referenz: „Oracle" (Schummeln mit Lösungsblatt)

Bevor wir die drei *echten* Stufen anschauen, brauchen wir einen Vergleichsmaßstab.

- **Was es tut:** Es probiert alle vier Detektoren aus und schaut **mit den echten Labels** nach,
  welcher am besten war. Es wählt also garantiert den besten.
- **Wie:** Es benutzt das „Lösungsblatt" (`y`), das es im echten Betrieb **nicht gäbe**.
- **Warum trotzdem nützlich:** als **Obergrenze**. Es sagt: „Besser als das geht es nicht." Unsere
  label-freien Stufen wollen möglichst nah an diese Linie herankommen.

> In euren Ergebnissen: Oracle wählt **pca → 0.845**. Das ist die Messlatte.

---

## Stufe 1 — naiv: „Nimm einfach immer den gleichen"

- **Was es tut:** Es wählt **fest einen** Detektor (bei uns `ecod`) und hofft, dass er passt.
- **Wie:** Gar keine Logik — der Name steht hart im Code (`NAIVE_DETECTOR = "ecod"`).
- **Schwäche:** Man hat **keine Ahnung**, ob die Wahl gut war. Die vier Detektoren schwanken hier
  zwischen 0.767 und 0.846 — eine Glückssache.

> In euren Ergebnissen: naiv = **ecod → 0.767**. Das ist zufällig der **schlechteste** der vier.
> Genau das ist die Botschaft dieser Stufe: blind wählen ist riskant.

---

## Stufe 2 — statistisch (Konsens): „Lass die Detektoren abstimmen"

Das ist die elegante, label-freie Idee. Sie braucht **keine Lösungen**.

- **Was es tut:** Sie findet heraus, welcher Detektor der **vertrauenswürdigste** ist — allein
  daraus, wie sehr die Detektoren untereinander übereinstimmen.
- **Wie, in vier Schritten:**
  1. Jeder Detektor vergibt jedem Zeitpunkt einen Anomalie-Score und bringt damit alle Zeitpunkte
     in eine **Rangfolge** („Punkt 7 ist am verdächtigsten, dann Punkt 12, …").
  2. Wir bilden die **Konsens-Rangfolge** = der Durchschnitt aller vier Rangfolgen
     („was die Mehrheit verdächtig findet").
  3. Für jeden Detektor messen wir, **wie stark seine eigene Rangfolge mit dem Konsens überein­stimmt**
     (Korrelation).
  4. Wir wählen den Detektor, der dem Konsens **am nächsten** ist (das „zentralste" Modell).
- **Warum das funktioniert (Schwarmintelligenz):** Ein Detektor, der mit der breiten Mehrheit
  übereinstimmt, liegt meist richtig. Ein Außenseiter, der ganz andere Punkte verdächtigt,
  ist oft der unzuverlässige. Man bevorzugt also den „Mainstream"-Detektor.
- **Mini-Beispiel:** Wenn pca, iforest und ocsvm grob dieselben Punkte verdächtigen und nur ecod
  völlig andere, dann ist ecod der Außenseiter — der Konsens stützt sich auf die drei einigen.

> In euren Ergebnissen: Konsens wählt **pca → 0.845** — **exakt das Oracle**, aber **ohne Labels**.
> Das ist das Kernergebnis: AutoML wählt fast optimal, ohne die Lösung zu kennen.

---

## Stufe 3 — die „echte" moderne AutoML: pyODs ADEngine

Hier nutzen wir nicht mehr nur Eigenbau, sondern die **eingebaute AutoML-AD von pyOD 3**
(`pyod.utils.ad_engine.ADEngine`, die „od-expert"-Skill ist die agentische Schicht darüber).
Auch hier: **keine Labels** nötig.

- **Was es tut:** Ein einziger Aufruf `engine.investigate(X)` macht die ganze Untersuchung selbst.
- **Wie, in vier Schritten:**
  1. **Datensatz profilieren** (welche Art Daten, welche Eigenschaften?).
  2. **Detektoren benchmark-gestützt wählen:** Aus 61 Detektoren nimmt es die, die auf ähnlichen
     Datensätzen in großen **Benchmarks** (ADBench, NeurIPS 2022) am besten waren. Das ist genau das
     **Meta-Learning aus der Vorlesung** (VL06) — „was woanders gut war, probiere ich zuerst".
  3. **Consensus bilden:** Es lässt die gewählten Detektoren gemeinsam abstimmen (wie Stufe 2,
     nur über eine benchmark-gewählte Auswahl) und liefert einen gemeinsamen Anomalie-Score.
  4. **Label-freie Qualität melden:** Es bewertet selbst, wie verlässlich das Ergebnis ist
     (`verdict`/`overall`) — ganz ohne Lösungen.
- **Stärke:** fertige, breite, benchmark-fundierte Pipeline; keine Handarbeit nötig.
- **Schwäche:** „Blackbox-Allrounder" — die benchmark-gewählte Detektor-Mischung ist nicht
  automatisch die beste für *genau diesen* Datensatz.

> In euren Ergebnissen: ADEngine wählte benchmark-gestützt **IForest + ECOD + KNN**,
> Consensus-ROC-AUC **0.794**, label-freie Qualität **„medium"**. Es validiert sich sogar selbst
> (`validate`): sein Consensus (0.794) schlägt seinen besten Einzeldetektor (0.763) →
> **`consensus_helped=True`**. Spannend trotzdem: Das schwergewichtige Library-Tool **schlägt euren
> einfachen Konsens (0.845) nicht** — ein ehrlicher Befund (mehr Maschinerie ≠ automatisch besser).

### Und unser Eigenbau-LLM (Stufe 3 „unter der Haube")?

`automl_ad/llm.py` bleibt als **vereinfachte Erklärung**, wie so eine wissensbasierte Auswahl
*innen* funktioniert: Datensatz-Steckbrief (Kennzahlen → Stichworte) + Detektor-Steckbriefe
(Stärken/Schwächen) → ein Sprachmodell (Claude oder lokal Ollama) wählt **mit Begründung**.

> Lehrreich: Anfangs wählte das kleine **llama3.1** fälschlich ecod (0.767), weil ein Stichwort
> („Schiefe") zu stark zog. Nachdem wir die Steckbriefe **entzerrt** und ins Profil die
> **lineare Kopplung** (PCA-Varianz) aufgenommen haben, wählt dasselbe Modell jetzt **pca → 0.845**
> (= Oracle): *„stark gekoppelte Features, ausgeprägte lineare Struktur → PCA"*. Zeigt schön, wie
> sehr **Prompt-Qualität** über das Ergebnis entscheidet.

---

## Eure Ergebnisse in einem Satz pro Zeile

| Stufe | Wahl | ROC-AUC | Lesart |
|-------|------|---------|--------|
| naiv | ecod | 0.767 | blind gewählt → hier der schlechteste |
| **Konsens (Stufe 2)** | **pca** | **0.845** | **ohne Labels, trifft das Optimum** |
| pyOD ADEngine (Stufe 3) | IForest/ECOD/KNN | 0.794 | echtes Library-Tool, schlägt den Konsens hier nicht |
| **Bonus: ADEngine + LLM-Routing** | LOF (+MCD) | 0.825 | unser LLM wählt aus pyODs 61 Detektoren — findet LOF (nicht in unseren 4!) |
| LLM-Eigenbau (unter der Haube) | pca | 0.845 | nach Prompt-Verbesserung optimal |
| Oracle (Referenz) | pca | 0.845 | Obergrenze (nutzt Labels, gibt's real nicht) |

### Bonus: unser LLM steuert pyODs Auswahl

pyODs `plan_detection(..., llm_client=…)` erlaubt es, die Detektorwahl von **unserem** LLM treffen zu
lassen — pyOD baut den Prompt aus seiner Wissensbasis (61 Detektoren), unser Provider (Claude/Ollama)
antwortet, pyOD führt den Plan aus. Verbindet „unter der Haube" (unser LLM) mit „echt" (pyOD-Engine).
Schön im Lauf: Das LLM wählte **LOF** — einen Detektor, der in unseren vier gar nicht vorkommt — und
landete bei **0.825**, also über pyODs eigener regelbasierter Auswahl (0.794). *Code:
`pyod_engine.run_engine_llm_routed` + `llm.llm_router`.*

**Fazit für die Abgabe:** Die **statistische Konsens-Auswahl** ist das Highlight — sie erreicht ohne
jegliche Labels die Oracle-Obergrenze. **Stufe 3** zeigt die moderne, library-native AutoML-AD
(ADEngine) als realen Vergleich; dass sie hier nicht gewinnt, ist ein ehrlicher und guter
Diskussionspunkt. Der Eigenbau-LLM bleibt als didaktischer Blick „unter die Haube".
