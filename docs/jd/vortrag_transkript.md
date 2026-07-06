# Vortrags-Transkript: PyOD bis Fazit PyOD

> Umfang: 16 Folien (PDF S. 12–27) · Richtzeit: ca. 1–1,5 min pro Folie.
> **[ZEIGEN: …]** = an dieser Stelle auf das Element in der Grafik deuten.
> Kursiv = Übergang zur nächsten Folie.

---

## Folie 1 — Baseline Modelle (4 Karten)

Das sind unsere vier Baseline-Modelle. Sie sind bewusst so gewählt, dass sie
Anomalien auf **vier grundverschiedene Arten** definieren — jede Karte zeigt die
Score-Landschaft auf Beispieldaten: blau = unauffällig, rot = anomal.

- **HDBSCAN** *(links oben)*: Dichte-Clustering. Wer in dünn besiedelten Regionen
  liegt, ist Ausreißer. Output: **Dichte** — wenig Dichte = anomal.
- **Isolation Forest** *(rechts oben)*: zufällige Schnitte durch den Datenraum.
  Anomalien sind schnell isoliert — kurzer Pfad = anomal. Output: **Pfadlänge**.
- **KNN** *(links unten)*: abstandsbasiert. Wer weit von seinen k nächsten
  Nachbarn entfernt ist, ist anomal. Output: **K-Distanz**.
- **PCA** *(rechts unten)*: lernt die „Hauptebene" der Normaldaten. Der Score ist
  der **Rekonstruktionsfehler** — der Abstand zwischen einem Punkt und seiner
  Projektion auf diese Ebene. Punkte, die das normale Korrelationsmuster der
  Sensoren brechen, lassen sich schlecht rekonstruieren.

Merkt euch: vier Modelle, vier Definitionen von „anomal", vier Skalen —
Fehler, Distanz, Pfadtiefe, Dichte.

*Wie sieht so ein Detektor nun im laufenden Betrieb aus?*

---

## Folie 2 — PyOD (Was ist PyOD? + Code-Beispiel)

Nachdem wir gesehen haben, dass Standard-Metriken ohne Labels versagen, stelle ich
euch jetzt unser Werkzeug vor: **PyOD** — die Python-Library, die sich als
Industriestandard für Outlier Detection etabliert hat.

Die Kernidee von PyOD ist Einfachheit: Es gibt **ein einheitliches
BaseDetector-Interface** für über 60 verschiedene Algorithmen. Egal welches
Modell — die API ist immer dieselbe: `fit()` zum Trainieren,
`decision_function()` für die Anomalie-Scores. Entwickler müssen die Algorithmen
nicht selbst implementieren.

**[ZEIGEN: Code links]** Hier sieht man das komplett: Vier verschiedene Detektoren —
PCA, KNN, IForest, HDBSCAN — laufen in einer einzigen Schleife, weil alle dasselbe
Interface haben. Ein *Anomalie-Score* bedeutet dabei immer: höher = auffälliger.

**[ZEIGEN: Terminal rechts]** Und hier der wichtigste Befund für alles, was danach
kommt: Die maximalen Scores liegen auf **völlig verschiedenen Skalen** — PCA etwa
300, IForest 0,05, HDBSCAN 1,0. Jedes Modell misst in seiner eigenen „Währung".
Die Scores sind also **nicht direkt vergleichbar** — das wird uns gleich beim
Konsens wieder begegnen.

*Schauen wir uns die vier Modelle erst einmal einzeln an.*

---

## Folie 3 — Anomalie-Score im Zeitverlauf

Hier sehen wir **ein** Modell — PCA — im Prozess-Monitoring auf einem echten
TEP-Fehlerlauf. Das ist das Bild, das man im Betrieb sehen will.

**[ZEIGEN: X-Achse]** Die X-Achse ist die Zeit in Samples.
**[ZEIGEN: Y-Achse]** Die Y-Achse ist der Anomalie-Score, normiert auf 0 bis 1
über den gesamten Testdatensatz — die absolute Höhe ist deshalb nicht
entscheidend, sondern der Abstand zur Schwelle.

**[ZEIGEN: graue gestrichelte Linie]** Diese Schwelle ist das **99 %-Quantil des
Normalbetriebs**: Wir nehmen alle Scores, die die Anlage im fehlerfreien Betrieb
produziert, und legen die Linie so, dass 99 % darunter liegen. Wichtig: Dafür
brauchen wir **keine Fehler-Labels** — nur die Annahme „das war Normalbetrieb".
Bewusst nehmen wir 1 % Fehlalarme in Kauf.

**[ZEIGEN: rote gepunktete Linie + roter Bereich]** Ab Sample 160 ist der Fehler
aktiv — das ist die Wahrheit aus der Simulation. Und man sieht: Vor dem Onset
dümpelt der Score unter der Schwelle, **beim Onset springt er hoch und bleibt
dauerhaft darüber**. Anlage läuft → Störung → Alarm. Genau so soll es aussehen.

*Das funktioniert gut — aber mit welchem der vier Modelle? Das ist die
eigentliche Frage.*

---

## Folie 4 — Model Selection: Streuung der Detektoren

Jetzt kommt unser Kernproblem. Angenommen, wir müssen uns **blind** — also ohne
Labels — für einen der vier Detektoren entscheiden.

Kurz zur Y-Achse, die taucht ab jetzt öfter auf: **ROC-AUC** ist die
Standard-Metrik für Anomalieerkennung. Sie misst, wie gut ein Modell anomale
Punkte über normale einsortiert: **1,0 = perfekte Trennung, 0,5 = Münzwurf**.
Wir haben sie gewählt, weil sie **schwellen-unabhängig** ist — sie bewertet die
gesamte Score-Reihenfolge, nicht eine einzelne Alarm-Grenze — und weil sie der
Standard in der AD-Literatur ist, also vergleichbar mit anderen Arbeiten.

**Aber — und das steht bewusst rot unter jeder Grafik:** ROC-AUC braucht Labels.
Im echten unüberwachten Betrieb gibt es sie **nicht**. Wir benutzen sie hier nur
zur Illustration, quasi als „Spick-Kontrolle" im Nachhinein — nie zur Auswahl.

**[ZEIGEN: Balken]** Und jetzt das Problem: Die vier Detektoren streuen von
0,757 bis 0,890 — eine Spanne von 0,133. **[ZEIGEN: Δ-Pfeil]** Wer blind
HDBSCAN greift, verschenkt massiv Leistung gegenüber KNN. Blinde Einzelwahl ist
Glückssache.

*Wie wählt man also ohne Labels? Unsere Antwort: Man fragt alle vier.*

---

## Folie 5 — Model Selection: Konsens Modus A

Idee: Wenn wir keine Wahrheit haben, nehmen wir die **Mehrheitsmeinung der
Modelle** als Ersatz. Dafür müssen die vier erst vergleichbar werden — und da
hilft der Trick von vorhin: Wir werfen die Roh-Scores weg und behalten nur die
**Rangfolge**. Jeder Detektor sortiert alle Punkte vom unauffälligsten zum
anomalsten; Ränge sind über Modelle hinweg vergleichbar, Skalen egal.

Der **Rang-Konsens** ist dann einfach das Mittel der vier Ranglisten — die
Schwarm-Meinung.

**[ZEIGEN: Heatmap links]** Hier die Rang-Korrelation zwischen allen Detektoren —
also wie ähnlich ihre Ranglisten sind (1,0 = identisch). KNN und PCA sind sich
mit 0,99 fast einig, HDBSCAN schert mit 0,64–0,68 deutlich aus.

Das **agreement = 0,78** oben ist der Durchschnitt all dieser Paare — unsere
„Schwarm-Einigkeit". Ein label-freies Vertrauenssignal: Wenn vier verschieden
gebaute Algorithmen ähnlich ranken, ist die Konsens-Aussage vermutlich belastbar.

**[ZEIGEN: Balken rechts]** Modus A wählt nun den Detektor, der dem Konsens am
nächsten ist — die „Centrality". Das ist **PCA mit 0,97**. Sprich: Wenn wir uns
für genau ein Modell entscheiden müssen, nehmen wir das zentralste — komplett
ohne Labels.

*Modus A wählt ein Modell. Man kann aber auch alle vier behalten.*

---

## Folie 6 — Model Selection: Konsens Modus B (Ensemble)

Modus B sagt: Kein Detektor gewinnt — **der kombinierte Score ist die
Vorhersage**. Dafür werden die Scores je Detektor erst z-normiert, also auf
dieselbe Skala gebracht, und dann pro Punkt kombiniert. Die drei Balken sind
drei Kombinations-Regeln:

- **average**: der Durchschnitt der vier Scores — ausgewogen, jeder zählt gleich.
- **maximization**: der höchste der vier — „einer schlägt Alarm reicht".
  Empfindlich, fängt seltene Anomalien, kostet aber Fehlalarme.
- **median**: der mittlere Wert — robust, ein einzelner durchdrehender Detektor
  kann das Ergebnis nicht kippen. Hier mit 0,858 auch am besten.

**[ZEIGEN: gestrichelte Linie]** Zum Vergleich der beste Einzeldetektor mit 0,890
— den kennt man aber nur mit Labels. Die Botschaft ist nicht „Ensemble gewinnt",
sondern **Robustheit**: Alle drei Regeln liegen dicht beieinander und nah am
Optimum, ohne dass wir die riskante Einzelwahl treffen mussten.

*Klingt alles sehr gut. Jetzt kommt die wichtigste Folie meines Teils — die
Ehrlichkeits-Folie.*

---

## Folie 7 — Model Selection: Einigkeit ≠ Korrektheit

Bisher könnte man denken: agreement hoch, Konsens gut, Problem gelöst. Diese
Folie zeigt, warum das gefährlich ist.

**[ZEIGEN: obere Grafik]** Hier die Erkennungsleistung des Konsens **pro
Fehlertyp**, sortiert von leicht nach schwer. Links die gutmütigen Fehler —
ROC-AUC nahe 1,0. Und rechts, rot markiert: **F9, F15 und F3 mit etwa 0,57 bis
0,56 — praktisch Zufallsniveau.** Diese drei Fehlertypen sind für alle unsere
Detektoren quasi unsichtbar.

**[ZEIGEN: untere Grafik]** Und jetzt der entscheidende Punkt: Das ist unser
label-freies Vertrauenssignal, das agreement, für dieselben Fehlertypen. Es ist
**flach**. Bei den unsichtbaren Fehlern rechts ist die Einigkeit genauso hoch
wie bei den leichten links.

Heißt: Die vier Detektoren sind sich einig — und **gemeinsam blind**. Hohe
Einigkeit schützt nicht vor Blindheit. Das label-freie Signal warnt uns bei
genau den Fehlern nicht, bei denen wir die Warnung bräuchten. **Einigkeit ist
nicht Richtigkeit** — das ist die zentrale Grenze aller label-freien Evaluation,
und die nehmen wir mit in die Diskussion.

*So viel zu unserem selbstgebauten Konsens. PyOD bringt aber seit Version 3
eine eigene AutoML-Lösung mit — die ADEngine.*

---

## Folie 8 — PyOD ADEngine (Übersicht)

Die **ADEngine** ist PyODs eingebaute „Anomaly Detection Lifecycle Engine". Sie
übernimmt genau das, was wir eben von Hand gemacht haben — und mehr: Data
Profiling, Planung der Detection, Erstellung der Detektoren und benchmark-gestützte
Auswahl aus einer **Wissensbasis mit über 60 Detektoren**. Nutzbar als reine
Python-API oder als Backend für LLM-Agents und MCP-Server.

**[ZEIGEN: Code links]** Der Aufwand für den Nutzer: **ein Aufruf.**
`engine.investigate(X)` — man übergibt nur die Daten, ohne je einen Algorithmus
zu nennen.

**[ZEIGEN: Terminal rechts]** Zurück kommt ein komplettes Ergebnis: Die Engine
hat selbstständig KNN, IForest und LOF gewählt, einen Konsens-Score gebildet und
sich selbst benotet — hier im Demo-Beispiel auf Zufallsdaten mit Einigkeit 0,94
und Qualität „medium".

**[ZEIGEN: Pipeline rechts oben]** Intern durchläuft dieser eine Aufruf vier
Phasen: **profile → plan → run → analyze**. Die schauen wir uns jetzt einzeln
an — mit den echten Zahlen aus unserem TEP-Lauf.

---

## Folie 9 — ADEngine Phase ①: profile

Phase 1 ist unspektakulärer, als der Name klingt — und das ist der Punkt:
**Profiling ist ein Steckbrief der Daten, kein Modell.**

**[ZEIGEN: Karte links]** Unser echter Lauf: Die Engine erkennt den Datentyp
tabular, zählt 3 000 Samples und 52 Features, prüft auf fehlende Werte — keine —
und den Datentyp float64.

**[ZEIGEN: Skala unten rechts]** Interessantester Punkt: die
**Dimensionalitätsklasse**. Bis 10 Features gilt „low", bis 100 „medium",
darüber „high". Unsere 52 Features landen in „medium".

Warum das zählt: Genau diese Merkmale — und nur diese — sind die **Eingabe für
die Planungsregeln** der nächsten Phase. Die Engine schaut sich also nicht die
Werte im Detail an, sondern nur die Form der Daten.

---

## Folie 10 — ADEngine Phase ②: plan

Phase 2 wählt die Detektoren — und zwar durch **Nachschlagen statt Trainieren**.

**[ZEIGEN: Regel-Zitat oben]** Für unser Profil greift die Regel: „general
tabular → robuste Allrounder aus den ADBench-Top-5". ADBench ist ein großer
Benchmark über 57 fremde Datensätze — es findet also **kein Fit auf unseren
TEP-Daten** statt, die Engine nutzt eingebautes Benchmark-Wissen.

**[ZEIGEN: drei Karten]** Der Plan: **IForest als Primär-Detektor** (ADBench-Rang
3, Konfidenz 0,85), dazu ECOD und KNN als Alternativen. IForest und KNN kennt
ihr schon; **ECOD** ist neu für uns: Es prüft pro Sensor, wie extrem ein Wert am
Rand seiner Verteilung liegt, und multipliziert diese Randwahrscheinlichkeiten
über alle Dimensionen auf — parameterfrei und sehr schnell, aber es betrachtet
jede Dimension einzeln und ist damit blind für Anomalien, die nur im
Zusammenspiel der Sensoren stecken. Die **Konfidenz** ist
dabei wichtig richtig einzuordnen: Das ist **keine berechnete Größe**, sondern
ein fest in der Wissensbasis hinterlegter Wert pro Regel — quasi ein Prior der
PyOD-Autoren, wie stark diese Regel dem Detektor vertraut. Der Detektor mit der
höchsten Konfidenz wird primär.

Der Plan enthält außerdem die Hyperparameter — im Wesentlichen nur
**contamination = 0,1**, der Default. Contamination ist der *angenommene*
Anomalie-Anteil: „Erkläre die obersten 10 % der Scores zu Anomalien." Merkt euch
diese Annahme — die fällt uns in Phase 4 auf die Füße.

Und: Über `priority` kann man die Regeln Richtung speed oder accuracy schieben;
optional kann statt der Regeln ein **LLM** die Wahl treffen — dazu gleich mehr.

---

## Folie 11 — ADEngine Phase ②: plan (Code)

Kurz der Beleg, dass das wirklich so einfach ist, wie es klingt.

**[ZEIGEN: Code oben]** `profile_data(X)`, dann `plan_detection(profile)` — zwei
Zeilen, und wir haben den kompletten Plan mit Wahl, Begründung, Konfidenz und
Parametern.

**[ZEIGEN: explain_detector]** Und das finde ich das schönste Detail: Mit
`explain_detector()` kann man die Wissensbasis direkt befragen. Zu jedem
Detektor gibt es einen Steckbrief — Stärken, Schwächen und sogar ein
„avoid_when": IForest zum Beispiel meiden, wenn Anomalien lokale
Dichte-Abweichungen sind oder Features stark korreliert sind. Die
Detektor-Auswahl ist damit **erklärbar**, nicht Blackbox.

---

## Folie 12 — ADEngine Phase ③: run

Phase 3 führt den Plan aus, in zwei Schritten.

**[ZEIGEN: Liste rechts oben]** Erstens: Jeder geplante Detektor fittet und
scort auf den Daten — alle drei erfolgreich, jeweils unter 0,3 Sekunden. Über
die contamination-Annahme flaggt jeder seine Top-10 %, also je 300 Punkte.

Zweitens: der **Rang-Konsens** — dasselbe Prinzip wie unser Modus A vorhin:
Scores je Detektor in Ränge umwandeln, Ränge mitteln.

**[ZEIGEN: Histogramm]** So liest man das Bild: Auf der X-Achse der
Konsens-Score von 0 bis 1, auf der Y-Achse, **wie viele der 3 000 Punkte** in
das jeweilige Score-Intervall fallen. **[ZEIGEN: Cutoff-Linie]** An der
gestrichelten Linie schneidet die Engine ab: Die orangen 316 Punkte rechts —
die obersten 10,5 % — gelten als anomal.

Zwei Kennzahlen daneben: **agreement = 0,79** — die Schwarm-Einigkeit, wie bei
uns die mittlere Rang-Korrelation der Detektoren. Und **432 Streitfälle** — das
sind Punkte, bei denen das Ja/Nein-Votum der drei Detektoren gespalten war, also
2:1 statt einstimmig.

Und schaut euch die Verteilung schon mal an: Sie läuft am Cutoff **glatt durch** —
kein Tal, kein Sprung. Das wird gleich wichtig.

---

## Folie 13 — ADEngine Phase ④: analyze

Phase 4 ist das Alleinstellungsmerkmal: Die Engine **benotet sich selbst — ohne
Labels.** Das Verdikt ist der Mittelwert aus drei Diagnose-Metriken, jede prüft
einen anderen Fehlermodus:

**[ZEIGEN: Balken separation]** **separation 0,89** — heben sich die geflaggten
Punkte klar vom Rest ab? Hier ja. Aber Achtung, das ist an den *eigenen* Labels
gemessen, also beschreibend, kein unabhängiger Beweis.

**[ZEIGEN: Balken agreement]** **agreement 0,79** — die Einigkeit von eben,
direkt übernommen.

**[ZEIGEN: Balken stability]** Und jetzt der spannende: **stability 0,00.**
Stability misst den Score-Sprung genau an der Cutoff-Grenze. Erinnert euch an
das glatte Histogramm: Es gibt **keinen** Sprung. Das heißt, die
10 %-Grenze ist wackelig — eine minimal andere contamination ergäbe eine
deutlich andere Anomalie-Menge. Die Engine deckt damit selbst auf, dass ihre
eigene Default-Annahme nicht zu TEP passt, wo real deutlich mehr als 10 % der
Testdaten anomal sind.

**[ZEIGEN: schwarze Linie + Zonen]** Mittelwert: (0,89 + 0,79 + 0,00) / 3 =
**0,56 → verdict „medium"** — ab 0,4 medium, ab 0,7 high.

**[ZEIGEN: best_detector rechts unten]** Und noch eine ehrliche Pointe: Die
Engine kürt label-frei **ECOD** zum besten Detektor. Mit Labels gespickt war es
aber KNN. Auch die Selbst-Bewertung kann sich irren — Einigkeit ist nicht
Richtigkeit, dasselbe Muster wie bei unserem Konsens.

---

## Folie 14 — ADEngine: LLM-Adapter

Bonus-Feature: Die Detektor-Wahl aus Phase 2 kann statt der Regeln ein **LLM**
übernehmen — und zwar PyOD-nativ, wir haben da nichts drumherum gebaut.

**[ZEIGEN: Flussdiagramm, im Uhrzeigersinn]** Der Ablauf: PyOD baut selbst den
Prompt aus seiner Wissensbasis mit den 60+ Detektoren. Wir liefern nur den
Transport zum Modell — Claude oder lokal Ollama. Die Antwort kommt als JSON
zurück, PyOD **parst und validiert sie gegen die Wissensbasis**, und erst der
geprüfte Plan geht in `run_detection`.

Ergebnis bei uns: Das LLM wählte **LOF**, den Local Outlier Factor — der
lokale Bruder unseres KNN: Er vergleicht die lokale Dichte eines Punktes mit
der seiner Nachbarn — wer deutlich einsamer wohnt als seine Umgebung, ist
anomal. Im Gegensatz zu KNN misst LOF also relativ zur Nachbarschaft statt
absolut. Mit ROC-AUC 0,883 eine richtig gute Wahl, auf Augenhöhe mit unserem
Konsens A — und kein Zufall, dass ein KNN-Verwandter hier gut landet.

Zwei Eigenschaften muss man festhalten: **① Sicher** — das LLM kann nicht
ausbrechen und nichts Ungültiges wählen, weil Form-Prüfung, KB-Validierung und
Regel-Fallback dahinterstehen. Wenn die Antwort Unsinn ist, fällt PyOD einfach
auf das Regel-Routing zurück. **② Nicht-deterministisch** — die Wahl kann je
Lauf schwanken. Für reproduzierbare Ergebnisse cachen wir deshalb einen guten
Lauf.

*Damit haben wir fünf Strategien auf dem Tisch. Zeit für den Vergleich.*

---

## Folie 15 — PyOD Fazit (Strategie-Vergleich)

Hier alle label-freien Strategien nebeneinander, gemessen in ROC-AUC — wie
immer nur illustrativ, im Betrieb hätten wir diese Zahlen nicht.

**[ZEIGEN: von links nach rechts]**

- **Naiv, fix IForest, 0,826**: der typische Standard-Griff ohne nachzudenken.
  Unsere Untergrenze.
- **Konsens A, 0,883**: unser zentralstes Modell — die Wahl fiel label-frei auf
  PCA, und die ist fast optimal.
- **Konsens B, 0,834**: das Ensemble als Vorhersage — solide, robust, aber hier
  etwas schwächer als die beste Einzelwahl.
- **ADEngine, 0,850**: bemerkenswert, denn die Engine kannte TEP nicht — reines
  Benchmark-Wissen aus der Wissensbasis, ohne jedes Tuning.
- **ADEngine + LLM, 0,883**: die LLM-Wahl LOF, gleichauf mit unserem Konsens A.

Drei Take-aways: **Erstens**, jede informierte Strategie schlägt den naiven
Griff. **Zweitens**, unser einfacher Rang-Konsens hält mit der fertigen
AutoML-Engine mit — und der Abstand hat einen Grund: Die ADEngine schlägt nur in
Benchmark-Ranglisten nach und kann PCA **strukturell gar nicht wählen** — PCA
steht in keiner ihrer Routing-Regeln, weil es über 57 fremde Datensätze
gemittelt nur Mittelfeld ist. Dass PCA auf TEP glänzt, liegt an den stark
korrelierten Sensoren der gekoppelten Anlage — genau diese Domänenstruktur
misst das Profil aber nicht. Unser Konsens findet PCA, weil er die Kandidaten
tatsächlich **auf unseren Daten** laufen lässt: Evidenz schlägt Benchmark-Prior.
**Drittens**, und das bleibt die ehrliche Klammer meines Teils: Alle diese
Auswahlverfahren funktionieren label-frei, aber ihre *Bewertung* hier nicht —
und bei den drei unsichtbaren Fehlertypen von vorhin versagen sie alle
gemeinsam, ohne dass ein label-freies Signal uns warnt.

*Und genau da setzen die nächsten Ansätze an — [Übergabe: EM/MV bzw. MetaOD].*

---

## Spickzettel: Begriffe in einem Satz (für Rückfragen)

| Begriff | Ein-Satz-Antwort |
|---|---|
| **Anomalie-Score** | Kontinuierlicher Wert pro Datenpunkt, höher = auffälliger; jedes Modell hat seine eigene Skala. |
| **ROC-AUC** | Wahrscheinlichkeit, dass ein zufälliger anomaler Punkt einen höheren Score bekommt als ein zufälliger normaler; 1,0 perfekt, 0,5 Zufall; schwellen-unabhängig, braucht aber Labels. |
| **99 %-Quantil-Schwelle** | Alarm-Grenze, unter der 99 % der Normalbetriebs-Scores liegen; label-frei bestimmbar, 1 % Fehlalarm einkalkuliert. |
| **agreement** | Mittlere paarweise Spearman-Rang-Korrelation der Detektoren; misst Einigkeit, nicht Korrektheit. |
| **Rang-Konsens** | Scores je Detektor in Ränge umwandeln und mitteln — macht verschiedene Skalen vergleichbar. |
| **Centrality (Modus A)** | Korrelation eines Detektors zum Rang-Konsens; der zentralste wird gewählt. |
| **Konfidenz (Plan)** | Fest in der Wissensbasis hinterlegter Vertrauenswert der Routing-Regel — nicht aus unseren Daten berechnet. |
| **contamination** | Angenommener Anomalie-Anteil (Default 10 %); macht aus Scores Ja/Nein-Labels; passt bei TEP nicht → stability 0. |
| **separation** | Wie stark sich die geflaggten Punkte vom Rest abheben — gemessen an den eigenen Labels, daher nur beschreibend. |
| **stability** | Score-Sprung an der Cutoff-Grenze; 0 heißt: die Anomalie-Menge hängt stark an der contamination-Annahme. |
| **Streitfälle** | Punkte mit gespaltenem Ja/Nein-Votum der Detektoren (2:1 statt einstimmig). |
| **ECOD** | Prüft pro Sensor, wie extrem ein Wert am Rand der empirischen Verteilung liegt, und kombiniert die Randwahrscheinlichkeiten aller Dimensionen; parameterfrei und schnell, aber blind für reine Kombinations-Anomalien (jede Dimension wird einzeln betrachtet). |
| **LOF** | Local Outlier Factor: Verhältnis der lokalen Dichte eines Punktes zur Dichte seiner k Nachbarn — „wohnt einsamer als seine Umgebung" = anomal; misst relativ zur Nachbarschaft, KNN dagegen absolut. |
| **Warum wählt die ADEngine kein PCA?** | PCA steht in keiner Routing-Regel (ADBench-übergreifend nur Mittelfeld, kein Top-5-Rang); die Engine rechnet zur Auswahl nichts auf den Daten und kann TEPs lineare Korrelationsstruktur — PCAs Sweetspot — nicht sehen. Bonus: Der KB-Steckbrief des gewählten IForest rät bei stark korrelierten Features sogar ab (avoid_when), aber das Routing prüft das nie gegen die echten Daten. |
