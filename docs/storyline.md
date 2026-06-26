# Vortrags-Storyline: Modellauswahl ohne Labels (naiv → Consensus → pyOD ADEngine)

**Format:** 20–30 min, 3 Vortragende · **Leitfrage:** *Wie wählt man ohne Labels ein passendes
Anomalie-Detektionsmodell für einen Datensatz?*

Roter Faden in **drei Stufen** + Oracle-Obergrenze, bewertet auf dem Tennessee-Eastman-Prozess (TEP).

---

## Block 1 — Motivation & Stufe 1 (naiv) · **Person A** (~8 min)

1. **Titel** — Thema, Namen, Kurs.
2. **Aufgabe & Motivation** — AutoML automatisiert die ML-Pipeline; Forschung meist *überwacht*.
   Unser Fokus: **unüberwachte Anomalieerkennung**. Leitfrage: *gegeben ein Datensatz → finde eine
   passende Anomalieerkennung*. (Lit.: Bahri et al.; PyOD-2.)
3. **Anomalieerkennung kurz** — Anomalietypen; AD vs. Fehlerklassifikation; un-/semi-/überwacht.
   Der „Detektor-Zoo" aus pyOD (ecod, iforest, ocsvm, pca, …).
4. **Das Kernproblem** — Man muss **einen** Detektor (+ Hyperparameter) wählen, hat aber **keine
   Labels**, um „gut" zu messen. Genau hier setzt AutoML an.
5. **Stufe 1 — naiv** — ein fixer pyOD-Detektor mit Defaults (z. B. `ecod`). Läuft, liefert Scores
   + Threshold — aber die Detektoren **streuen** stark (ROC-AUC ~0.78–0.855 auf TEP), und ohne
   Labels weiß man nicht, ob die Wahl gut war. *Code: `make_detector`.*

---

## Block 2 — AutoML & Stufe 2 (statistisch) · **Person B** (~8 min)

6. **Was ist AutoML** — automatisiert Preprocessing, Modellwahl, **Hyperparameter**.
7. **Klassisches Werkzeug: HPO & CASH** — Suchraum, Random/Grid → Bayesian (TPE); CASH =
   Algorithmenwahl + HPO gemeinsam. Aber: braucht eine **Zielfunktion** — im unüberwachten Fall
   fehlt sie. (1 Folie, `hpo.py` als Beleg; HPO-Gewinn auf TEP klein, Defaults sind stark.)
8. **Der Knackpunkt: Auswahl ohne Labels** — wie bewertet man Detektoren ohne Wahrheit?
9. **Stufe 2 — statistisch (Konsens)** — Idee: kein Modell ist die Wahrheit, aber der **Rang-Konsens**
   vieler Modelle ist robust; wähle das **zentralste** Modell (höchste Korrelation zum Konsens).
   Ergebnis: **≈ Oracle** ohne Labels. *Code: `select_internal`.*

---

## Block 3 — Stufe 3 (pyOD ADEngine), Ergebnisse & Fazit · **Person C** (~8 min)

10. **Stufe 3 — die echte AutoML-AD: pyOD ADEngine** — `pyod.utils.ad_engine.ADEngine`
    (`investigate(X)`): profiliert, wählt **benchmark-gestützt** (ADBench → Meta-Learning aus VL06!)
    Detektoren, bildet **Consensus** und meldet **label-freie Qualität**. Ein Aufruf, fertige
    Library-Pipeline. *Code: `pyod_engine.py`.*
    - **Unter der Haube** (1 Folie): so funktioniert wissensbasierte Auswahl intern — PyOD-2-Idee
      (Chen et al. 2024): Datensatz-Profil + Detektor-Steckbriefe → LLM wählt **mit Begründung**.
      *Eigenbau-Erklärer: `llm.py` / `select_llm` (Claude, Fallback Ollama).*
11. **Setup TEP** — realer Benchmark, 52 Features, 20 Fehler; run-level Splits + onset-korrektes
    Labeling **nur zur Evaluation**.
12. **Ergebnis** — Vergleich (Plot `reports/auswahl_vergleich.png`): naiv (0.767) vs. **Konsens
    (0.845 = Oracle)** vs. **pyOD ADEngine (0.794)** vs. Oracle (0.845). Kernaussage: label-freie
    Auswahl ≈ Oracle. Ehrlich: das schwergewichtige ADEngine schlägt den **einfachen** Konsens
    hier **nicht** (validiert sich selbst: `consensus_helped=True`, aber „medium" Qualität).
    - **Bonus-Folie:** unser LLM steuert pyODs Routing (`plan_detection(llm_client=…)`) und wählt
      aus **61** Detektoren — fand **LOF** (nicht in unseren 4!) → 0.825, über ADEngines
      Regel-Auswahl. Brücke „Eigenbau-LLM" ↔ „echte Engine". *Code: `run_engine_llm_routed`.*
13. **Ehrliche Einordnung** — absolute ROC-AUC „mittel" wegen heterogener Fehler (leichte ~1.0,
    harte ~0 Recall); wir optimieren **keine** Metrik, sondern zeigen das **Vorgehen**.
    Bonus: Prompt-Qualität entscheidet — llama3.1 wählte erst ecod (0.767), nach besseren
    Steckbriefen + Profil dann pca (0.845).
14. **Grenzen & Ausblick** — benchmark-/LLM-Auswahl hängt von Profil & Modell ab; nächste Schritte:
    Zeitmodelle, wenige Labels (semi-supervised), fehlerbewusste Ensembles.
15. **Fazit** — AutoML kann ein AD-Modell **ohne Labels** nahezu optimal wählen — statistisch
    (Konsens) **und** wissensbasiert (pyOD ADEngine). Direkte Antwort auf die Aufgabenstellung.

---

## Rollen & Timing
| Person | Folien | Inhalt | Zeit |
|---|---|---|---|
| **A** | 1–5 | Motivation, AD-Grundlagen, Stufe 1 (naiv) | ~8 min |
| **B** | 6–9 | AutoML/HPO/CASH, Stufe 2 (Konsens) | ~8 min |
| **C** | 10–15 | Stufe 3 (pyOD ADEngine), TEP, Ergebnisse, Fazit | ~8 min |

## Mapping auf Code/Artefakte
- Stufe 1/2/3 + Oracle → `scripts/run_experiment.py` → `reports/results.csv` + `auswahl_vergleich.png`
- Stufe 3 (echt) → `automl_ad/pyod_engine.py` (`ADEngine.investigate` + `validate`); „unter der Haube" → `automl_ad/llm.py`
- Live-Vorführung → `notebooks/demo.py` (ADEngine-Verdict + LLM-Begründung)
- HPO-Folie → `automl_ad/hpo.py`
