"""AutoML für Anomalieerkennung ohne Labels — auf dem Tennessee-Eastman-Prozess (TEP).

Roter Faden: *Wie wählt/kombiniert man ohne Labels einen Anomalie-Detektor?* TEP ist eine
**Zeitreihe**, daher durchgängig **zeitbewusst** (Fenster-Adapter, :mod:`automl_ad.ts`); der
i.i.d.-Blick ist nur der naive Startpunkt.

- **Konsens** (label-frei, :mod:`automl_ad.selection`): zentralstes Modell **oder** Ensemble-Konsens.
- **PyOD ADEngine** (:mod:`automl_ad.pyod_engine`): native, benchmark-gestützte AutoML-AD inkl.
  PyOD-eigenem LLM-Routing (wir liefern nur den Provider-Transport, :mod:`automl_ad.llm`).

Maßgebliche Doku: ``docs/jd/``.
"""
