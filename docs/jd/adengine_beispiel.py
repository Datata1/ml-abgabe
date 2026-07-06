import numpy as np
from pyod.utils.ad_engine import ADEngine

X = np.random.default_rng(42).normal(size=(500, 8))

engine = ADEngine()
state = engine.investigate(X, data_type="tabular")  

print("Gewählte Detektoren:", [r["detector_name"] for r in state.results])
print("Einigkeit  :", round(state.consensus["agreement"], 2))
print("Qualität   :", state.quality["verdict"])
print("Scores     :", np.round(state.consensus["scores"][:3], 2), "...")
