"""Phase ① + ②: profilieren, planen — und die Wissensbasis befragen."""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pyod.utils.ad_engine import ADEngine

X = np.random.default_rng(42).normal(size=(3000, 52))
engine = ADEngine()


# ── ① PROFILE + ② PLAN ──────────────────────────────────────────

profile = engine.profile_data(X)
plan = engine.plan_detection(profile, priority="balanced")
print("Profil :", profile)
print("Wahl   :", plan["detector_name"], "—", plan["reason"])
print("Konfid.:", plan["confidence"], "· Params:", plan["params"])
print("Altern.:", [a["detector_name"] for a in plan["alternatives"]])

info = engine.explain_detector(plan["detector_name"])

print("Stärke  :", info["strengths"][0])
print("Schwäche:", info["weaknesses"][0])
print("Meiden  :", info["avoid_when"])
