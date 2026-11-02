"""Smallest checks of the calibration and scoring logic. Run: python test_fuel_forecast.py"""
import numpy as np
import pandas as pd

import fuel_forecast as ff

rng = np.random.default_rng(0)
n = 400
h = rng.integers(1, 7, n)
p50 = np.full(n, 100.0)
actual = p50 * np.exp(0.1 * np.sqrt(h) * rng.standard_normal(n))     # true spread: 10% per sqrt(month)
ev = pd.DataFrame(dict(origin=pd.Timestamp("2026-01-01"), month=pd.Timestamp("2026-01-01"), h=h, actual=actual,
                       p10=p50 * np.exp(-0.03 * np.sqrt(h)), p50=p50, p90=p50 * np.exp(0.03 * np.sqrt(h)),   # fan far too narrow
                       unchanged=p50, fx_pass=p50, usual_level=p50))

raw = ff.score(ev, {"m": ev}).loc["m"]
assert raw["above_p90"] > 0.2, raw

c = ff.calibration(ev)
assert c["hi"] > 0 and c["lo"] > 0, c
fixed = ev.copy()
fixed["p10"], fixed["p90"] = ff._widen(ev["p10"], ev["p50"], ev["p90"], c, ev["h"])
s = ff.score(fixed, {"m": fixed}).loc["m"]
assert abs(s["above_p90"] - 0.1) < 0.04 and abs(s["below_p10"] - 0.1) < 0.04, s

# sticky price: p10 = p90 = actual up to exp(log()) rounding counts as covered
sticky = ev.assign(actual=132.0, p10=np.exp(np.log(132.0)), p50=132.0, p90=np.exp(np.log(132.0)))
assert ff.score(sticky, {"m": sticky}).loc["m", "cover_80"] == 1.0

# no look-ahead: the online fan of the first origin cannot be corrected (nothing realized before it)
early = ev.assign(origin=pd.Timestamp("2026-01-01"), month=pd.Timestamp("2026-02-01"))
assert (ff.calibrate_online(early)["p90"] == early["p90"]).all()
print("ok")
