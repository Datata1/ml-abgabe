"""AutoML für Anomalieerkennung — schlanke Abgabe-Fassung.

Roter Faden: *Wie wählt man ohne Labels ein passendes Anomalie-Detektionsmodell?*
Drei Stufen — naiv (fixer Detektor) → statistisch (Konsens, label-frei) → wissensbasiert
(LLM-gestützte Auswahl, PyOD-2-Idee). Bewertet auf dem Tennessee-Eastman-Prozess (TEP).
"""
