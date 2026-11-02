"""Syria fuel price: data loading, features, backtest and forecast.

Pipeline: load_decisions() -> make_pairs() -> backtest() / fit_process() -> forecast()
Run `python fuel_forecast.py` for a text summary; the notebook syria_fuel_forecast.ipynb calls the same functions.

Why decisions are modelled as "pairs": fuel prices in Syria are administrative, so the price only changes at
decisions. Each pair is two consecutive decisions of the same product and tier; the question is how the price
moves from one decision to the next (log change of the new-SYP price), given FX and Brent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import re

DATA = Path(__file__).parent / "data"
UNIT_TO_NEW_SYP = {"SYP_old_per_litre": 0.01, "SYP_new_per_litre": 1.0}   # new lira = old lira / 100 (factor converts a unit's price to new SYP)
MARKET_SINCE = "2024-05-01"    # first date with a dense, comparable log of free-market (unsubsidized) prices
PRODUCTS = ["gasoline_90", "gasoline_95", "diesel"]


# ---------------------------------------------------------------- loading
# Stage 1: load and clean the data. Everything below depends on this.
"""
'2007-2008' -> 2008-01-01, '2025-10' -> 2025-10-15
(mid month, so it sorts after early-month decisions),
'2000' -> 2000-01-01.
"""
def parse_date(text):
    # Dates in the files are inconsistent (year only, month, full day, year range); turn them all into real timestamps.
    text = str(text).strip()

    # Case 1: full day/month/year (e.g. 5/3/2025) -> direct date
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", text)
    if m:
        day, month, year = map(int, m.groups())
        return pd.Timestamp(year, month, day)

    # Case 2: year range '2007-2008' -> keep the last year only
    text = re.sub(r"^\d{4}-(\d{4})$", r"\1", text)
    parts = [int(p) for p in text.split("-")]
    # Pad the missing parts with None: (year) or (year, month) or (year, month, day)
    year, month, day = (parts + [None, None])[:3]
    # No month -> 1 January; month without day -> the 15th (mid-month) so it sorts after early-month decisions
    return pd.Timestamp(year, month or 1, day or (15 if month else 1))


def load_fx():
    # Estimated market USD rate (new lira per dollar), sorted by time
    fx = pd.read_csv(DATA / "fx_chart_estimates.csv")
    fx["date"] = pd.to_datetime(fx["date"], format="ISO8601")
    return fx.sort_values("date")[["date", "buy_est"]].reset_index(drop=True)          # buy_est = new SYP per USD (chart estimate)


def load_brent():
    # Monthly Brent price as a Series indexed by month start
    return pd.read_csv(DATA / "brent_monthly.csv", parse_dates=["date"]).set_index("date")["brent_usd"]


def load_events():
    # External events (wars, sanctions, fuel availability); an open-ended event is treated as lasting until today
    ev = pd.read_csv(DATA / "external_events.csv", parse_dates=["start", "end"])
    ev["end"] = ev["end"].fillna(pd.Timestamp.today())
    return ev


def availability_on(date, events):
    """
    available / limited / crisis from external_events;
    2014-2024 defaults to limited,
    unknown elsewhere.
    """
    # Was fuel available on this day? Find an 'availability' event covering the date and translate its label
    av = events[events["category"] == "availability"]
    hit = av[(av["start"] <= date) & (date <= av["end"])]
    if len(hit):
        return {"plentiful": "available", "rationed": "limited", "scarce": "crisis"}[hit["effect"].iloc[0]]
    # No event found: during the war years (2014-2024) assume 'limited', otherwise 'unknown' (do not invent a value)
    return "limited" if 2014 <= date.year <= 2024 else "unknown"


def load_decisions():
    """One row per (decision date, product): original file + extra decisions, in new SYP and in USD, with FX and Brent."""
    # Main function: merge all price decisions into one table, convert every price to new lira and to USD
    cols = ["effective_date", "product", "tier", "price_before", "price_after", "price_unit", "fuel_fx_per_usd"]
    # Merge the original file (rows without a price dropped) with the extra decisions file
    dec = pd.concat([pd.read_csv(DATA / "fuel_decisions.csv").dropna(subset=["price_after"]),
                     pd.read_csv(DATA / "fuel_decisions_extra.csv")], ignore_index=True).reindex(columns=cols)
    dec["date"] = dec["effective_date"].map(parse_date)
    # backward: only an FX estimate published before the decision (nearest could look up to two weeks ahead)
    # merge_asof backward = for each decision take the latest FX rate strictly before it (no look-ahead, so no leakage)
    dec = pd.merge_asof(dec.sort_values(["date", "product"]), load_fx(), on="date", direction="backward")
    scale = dec["price_unit"].map(UNIT_TO_NEW_SYP)
    usd_rows = dec["price_unit"] == "USD_per_litre"      # pump price = USD price x the ministry's fuel FX (1.10 x 133 = 146.30)
    # Convert the price to new lira with three fallbacks, in order (fillna = if the first fails try the next):
    #   1) old/new lira unit -> multiply by the conversion factor
    #   2) price quoted in USD -> multiply by the ministry's fuel FX rate
    #   3) otherwise -> multiply by the estimated market rate
    dec["syp_new"] = (dec["price_after"] * scale).fillna(dec["price_after"] * dec["fuel_fx_per_usd"].where(usd_rows)).fillna(dec["price_after"] * dec["buy_est"])
    # USD price = lira price / FX rate; the 'real' price, since the lira keeps losing value
    dec["usd"] = dec["syp_new"] / dec["buy_est"]
    dec["fuel"] = np.where(dec["product"].str.startswith("gasoline"), "gasoline", "diesel")
    # Brent in the decision month (lag 0) and the month before (lag 1), because local prices lag world prices
    brent, month = load_brent(), dec["date"].dt.to_period("M").dt.to_timestamp()
    for lag in (0, 1):
        dec[f"brent_l{lag}"] = month.map(lambda t: brent.get(t - pd.DateOffset(months=lag)))
    events = load_events()
    dec["availability"] = dec["date"].map(lambda d: availability_on(d, events))
    return dec.reset_index(drop=True)


# ---------------------------------------------------------------- pairs and features
# Stage 2: the core idea. Prices are administrative (change only at decisions), so I do not predict the price every day;
# I predict how much it changes from one decision to the next. Two consecutive decisions of the same product = one 'pair'.
def make_pairs(dec, since=MARKET_SINCE, tier="unsubsidized"):
    """Consecutive decisions of the same product: y = log change of the new-SYP price (the thing to predict)."""
    rows = []
    # Free-market (unsubsidized) tier only, and only after 2024-05 because earlier data is sparse
    sel = dec[(dec["tier"] == tier) & (dec["date"] >= since) & dec["product"].isin(PRODUCTS)]
    for product, g in sel.groupby("product"):
        g = g.sort_values("date")
        # zip(a[:-1], a[1:]) = each row with the row right after it
        for a, b in zip(g.index[:-1], g.index[1:]):
            r0, r1 = g.loc[a], g.loc[b]
            usd_before = r0["syp_new"] / r1["buy_est"]               # old SYP price expressed at the new FX
            # y = log of the price ratio (~ percentage change); this is the target I predict.
            # usd_before = old price valued at the new FX: measures whether the price became cheap in USD.
            # dfx / dbrent = how much the dollar and Brent moved between the two decisions
            rows.append(dict(product=product, d0=r0["date"], d1=r1["date"], y=np.log(r1["syp_new"] / r0["syp_new"]),
                             usd_before=usd_before, usd_after=r1["usd"], dfx=np.log(r1["buy_est"] / r0["buy_est"]),
                             dbrent=np.log(r1["brent_l1"] / r0["brent_l1"]), availability=r1["availability"]))
    return pd.DataFrame(rows).sort_values("d1").reset_index(drop=True)


def features(pairs, center):
    """center: dict product -> typical USD price. gap = how far the price (at the new FX) sits below its usual USD level."""
    # gap is my most important variable: if the USD price fell below its usual level (lira collapsed), the government will probably raise.
    # gap = log(usual / current): positive = cheap, expected to rise. The rest: FX change, Brent change, and dummies for product type
    X = pd.DataFrame({"gap": np.log(pairs["product"].map(center).values / pairs["usd_before"].values),
                      "dfx": pairs["dfx"], "dbrent": pairs["dbrent"],
                      "is95": (pairs["product"] == "gasoline_95").astype(float), "isdiesel": (pairs["product"] == "diesel").astype(float)})
    return X


# ---------------------------------------------------------------- backtest
# Stage 3: honest testing. Before trusting any model it must beat simple baselines (e.g. 'the price does not change').
def _models():
    # Three ML models for comparison: linear, trees, small neural net. Imports are inside the function so sklearn is not required by the rest of the module
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {"Ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
            "NeuralNet (8,4)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8, 4), alpha=1.0, max_iter=3000, random_state=0))}



def backtest(pairs, min_train=6):
    """Expanding-window backtest for different error-correction strengths."""
    # Expanding window: train on the past only, predict one pair, then widen the past and repeat. Mimics reality without cheating

    errors = {}
    kappa_values = [0.2, 0.3, 0.4, 0.5, 0.5843]

    for i in range(min_train, len(pairs)):
        te = pairs.iloc[[i]]                       # test pair (a single one)
        tr = pairs[pairs["d1"] <= te["d0"].iloc[0]]  # train = pairs that ended before the test pair starts (no look-ahead)

        if len(tr) < 4:
            continue

        # 'Usual USD level' per product = median of the training data only
        center = tr.groupby("product")["usd_after"].median().to_dict()
        center = {
            p: center.get(p, tr["usd_after"].median())
            for p in PRODUCTS
        }

        Xtr = features(tr, center)
        Xte = features(te, center)

        ytr = tr["y"].values
        y = te["y"].iloc[0]

        # Baseline predictions
        # Baselines: if a model does not beat these it has no value
        preds = {
            "price unchanged (SYP)": 0.0,                        # SYP price unchanged
            "price unchanged (USD)": Xte["dfx"].iloc[0],         # USD price unchanged (moves with FX)
            "back to usual USD level": Xte["gap"].iloc[0],       # returns immediately to the usual level
            "Brent pass-through": (                              # Brent pass-through: simple regression coefficient from the past x current Brent change
                (Xtr["dbrent"].values @ ytr)
                / (Xtr["dbrent"].values @ Xtr["dbrent"].values)
                * Xte["dbrent"].iloc[0]
                if (Xtr["dbrent"].values @ Xtr["dbrent"].values) > 0
                else 0.0
            ),
        }

        # Test fixed kappa values using past data only
        # beta = slope of the regression line through the origin between gap and y, from the past only
        x = Xtr["gap"].values
        denominator = x @ x
        beta = (x @ ytr) / denominator if denominator > 0 else 0.0

        # Error correction: the government closes a fraction kappa of the gap at each decision. Try several kappa values
        for kappa in kappa_values:
            preds[f"error correction kappa={kappa}"] = (
                kappa * Xte["gap"].iloc[0]
            )

        # Original data-estimated error-correction baseline
        preds["partial return (estimated beta)"] = (
            beta * Xte["gap"].iloc[0]
        )

        # Machine learning models
        for name, model in _models().items():
            preds[name] = model.fit(Xtr, ytr).predict(Xte)[0]

        # error = truth - prediction, per model
        for name, prediction in preds.items():
            errors.setdefault(name, []).append(y - prediction)

    # Results table: MAE (mean absolute error) and RMSE (penalizes big errors more)
    table = pd.DataFrame({
        name: {
            "n_tests": len(e),
            "mae_log": np.mean(np.abs(e)),
            "rmse_log": np.sqrt(np.mean(np.square(e))),
        }
        for name, e in errors.items()
    }).T

    # exp(rmse) = 'typical' error as a ratio (1.2 = about 20% off). Best model first
    table["typical_error"] = np.exp(table["rmse_log"])
    return table.sort_values("rmse_log")


def factor_check(pairs):
    """Which drivers of a decision does the data support? OLS of y on each candidate with t-stats (in-sample, diagnostic only).

    gap already prices FX in (it uses the FX at the new decision), so dfx measures pass-through *beyond* the gap.
    A driver earns a place in the process only if |t| > 2; jumps show up as fat tails of the residual."""
    # Question: which factor (gap, FX, Brent) really explains decisions? OLS regression with a t-statistic.
    # Rule: a factor enters the model only if |t| > 2. (This is where Brent turns out weak.)
    center = pairs.groupby("product")["usd_after"].median().to_dict()
    X = features(pairs, center)[["gap", "dfx", "dbrent"]].values
    y = pairs["y"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)          # coefficients by least squares
    resid = y - X @ b                                    # residuals = what the model could not explain
    se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))   # standard error
    z = (resid - resid.mean()) / resid.std(ddof=1)
    coefs = pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=["gap", "dfx", "dbrent"])
    # high kurtosis = 'fat tails' = rare large jumps in decisions (matters later for the bootstrap option)
    return coefs, dict(n=len(y), excess_kurtosis=float(np.mean(z ** 4) - 3), share_beyond_2sd=float(np.mean(np.abs(z) > 2)))




# ---------------------------------------------------------------- forecast
# Stage 4: forecasting by Monte Carlo simulation. Instead of one number, produce a 'fan' of probabilities (P10 / P50 / P90).
def _cutoff(as_of):
    """Data strictly before this date is known. Default: everything up to and including today."""
    # Cutoff date: everything before it is known, nothing after it may be used. Lets the same code serve the real forecast and the historical test
    return pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def fit_process(pairs, dec, hazard_window_days=365, as_of=None):
    """Parameters of the price process from data before as_of: usual USD level, partial return kappa, decision noise
    (sigma, plus the raw residuals for bootstrap) and how often decisions happen (rate, plus monthly counts for bootstrap)."""
    # Estimate 4 things from history: usual USD level, kappa (how much the government corrects), sigma (randomness), decisions per month
    t = _cutoff(as_of)
    pairs, dec = pairs[pairs["d1"] < t], dec[dec["date"] < t]
    center = pairs.groupby("product")["usd_after"].median().to_dict()
    center = {p: center.get(p, pairs["usd_after"].median()) for p in PRODUCTS}
    X = features(pairs, center)["gap"].values
    # kappa = regression slope, clipped to [0, 0.5] because the backtest says the relationship is weak (do not overstate it)
    kappa = float(np.clip((X * pairs["y"].values).sum() / (X * X).sum(), 0.0, 0.5))   # shrunk: the fit is weak (see backtest)
    resid = pairs["y"].values - kappa * X   # what is left is random after the correction
    # 'Hazard rate': how many decisions happened in the last year. Split the year into 12 monthly bins and count decisions in each
    recent = dec[(dec["tier"] == "unsubsidized") & (dec["date"] >= t - pd.Timedelta(days=hazard_window_days))]
    bins = np.arange(0, hazard_window_days + 1, 30.4)                                   # 12 month-long bins ending at t
    counts = {p: np.histogram((t - recent[recent["product"] == p]["date"].drop_duplicates()).dt.days, bins)[0] for p in PRODUCTS}
    per_month = {p: float(c.sum() / (hazard_window_days / 30.4)) for p, c in counts.items()}
    return dict(center=center, kappa=kappa, sigma=float(resid.std(ddof=1)), per_month=per_month, resid=resid, counts=counts)


def fx_volatility(since="2025-03-01", as_of=None):
    # Monthly FX volatility: divide the log change by the square root of elapsed time (random-walk law), then take the standard deviation
    fx = load_fx()
    fx = fx[(fx["date"] >= since) & (fx["date"] < _cutoff(as_of))]
    months = fx["date"].diff().dt.days / 30.4
    return float((np.log(fx["buy_est"]).diff() / np.sqrt(months)).std())      # monthly log volatility


def _widen(p10, p50, p90, calib, h):
    """Apply calibration() log corrections, scaled by sqrt(horizon); the band never crosses the median."""
    # Widen (or narrow) the fan according to calibration. The further the horizon h, the larger the correction, by sqrt(h). Bands never cross the median
    if not calib:
        return p10, p90
    w = np.sqrt(h)
    return np.minimum(p10 * np.exp(-calib["lo"] * w), p50), np.maximum(p90 * np.exp(calib["hi"] * w), p50)


def forecast(dec, params, months=6, n=4000, fx_drift=0.0, brent_path=None, brent_elasticity=0.0, seed=0,
             as_of=None, fx_shock=0.0, noise="normal", counts="poisson", calib=None):
    """Monte Carlo fan for each product. Returns a long DataFrame: month, product, p10/p50/p90 in new SYP, old SYP and USD.

    The price is sticky in SYP: in a month it changes only if a decision happens; a decision moves it by
    kappa x (distance to the usual USD level) + noise. FX is a log random walk. Brent only matters if
    brent_elasticity > 0 (not supported by the data, kept as a what-if).
      as_of    forecast made with data before this date (default: today); months count from its month start.
      fx_shock one-off log jump of FX at the start (np.log(1.1) = dollar up 10%).
      noise    "normal" (sigma x N(0,1)) or "bootstrap" (resample past decision residuals: keeps fat tails / jumps).
      counts   "poisson" (recent rate) or "empirical" (resample the last 12 monthly counts: keeps bursts and quiet spells).
      calib    {"lo", "hi"} from calibration(): widens (or narrows) p10 and p90."""
    # Idea: run n=4000 'possible futures' at once (numpy vectors). Each month:
    #   1) FX moves randomly   2) each future draws a number of decisions (Poisson)   3) each decision moves the price
    #   by kappa x gap + noise. At the end take the 10/50/90 percentiles over the 4000.
    rng = np.random.default_rng(seed)
    t = _cutoff(as_of)
    # Last known price of each product = starting point
    known = dec[(dec["date"] < t) & (dec["tier"] == "unsubsidized") & dec["product"].isin(PRODUCTS)]
    last = known.sort_values("date").groupby("product").tail(1).set_index("product")
    fx_hist = load_fx()
    fx0 = fx_hist[fx_hist["date"] < t]["buy_est"].iloc[-1]
    sig_fx = fx_volatility(as_of=t)
    start = t.to_period("M").to_timestamp()
    out = []
    # Everything in logs: add instead of multiply, then exp at the end. fx_shock = optional what-if shock
    fx = np.full(n, np.log(fx0) + fx_shock)
    price = {p: np.full(n, np.log(last.loc[p, "syp_new"])) for p in PRODUCTS}
    for k in range(1, months + 1):
        month = start + pd.DateOffset(months=k)
        fx = fx + fx_drift + sig_fx * rng.standard_normal(n)     # FX: random walk
        brent_factor = 1.0 if brent_path is None else (brent_path(month) / brent_path(start)) ** brent_elasticity
        for p in PRODUCTS:
            # Number of decisions this month in each simulation: Poisson (from the rate) or resampled from past months
            n_decisions = rng.poisson(params["per_month"][p], n) if counts == "poisson" else rng.choice(params["counts"][p], n)
            for j in range(n_decisions.max()):
                gap = np.log(params["center"][p] * brent_factor) - (price[p] - fx)    # log(usual USD) - log(current USD)
                shock = params["sigma"] * rng.standard_normal(n) if noise == "normal" else rng.choice(params["resid"], n)
                # (n_decisions > j) is 1 only for simulations that have a j-th decision; otherwise the price stays put (a 'sticky' price)
                price[p] = price[p] + (n_decisions > j) * (params["kappa"] * gap + shock)
            syp, usd = np.exp(price[p]), np.exp(price[p] - fx)
            row = dict(month=month, product=p, fx_p50=float(np.exp(np.percentile(fx, 50))))
            for unit, x in (("syp_new", syp), ("usd", usd)):
                p10, p50, p90 = np.percentile(x, [10, 50, 90])
                p10, p90 = _widen(p10, p50, p90, calib, k)
                row[f"{unit}_p10"], row[f"{unit}_p50"], row[f"{unit}_p90"] = p10, p50, p90
            out.append(row)
    res = pd.DataFrame(out)
    # Same numbers in old lira (x100) because people still count in it
    for l in ("p10", "p50", "p90"):
        res[f"syp_old_{l}"] = res[f"syp_new_{l}"] * 100
    return res


# ---------------------------------------------------------------- rolling-origin evaluation and calibration
# Stage 5: the most honest test. Go back in time to the start of each month, forecast without seeing the future, compare with the actual price.
# Then calibrate the fan: if reality falls outside it more than 20% of the time, widen it.
EVAL_SINCE = "2025-11-01"   # first origin: before it the free-market log has an 11-month hole (2024-11 to 2025-10), not a quiet price


def price_in_force(dec, product, date):
    # Price in force on a date = the last decision before it
    d = dec[(dec["tier"] == "unsubsidized") & (dec["product"] == product) & (dec["date"] < date)]
    return d.sort_values("date").iloc[-1]


def evaluate(dec, since=EVAL_SINCE, end=None, months=6, n=2000, **kw):
    """Rolling-origin test of the whole forecast. On the 1st of each month: fit on data before it, forecast 1-6 months
    and compare with the price actually in force. One row per (origin, product, horizon) with the fan and three baselines:
      unchanged    the last price stays
      fx_pass      the last price moves with the FX change since that decision
      usual_level  the price jumps straight back to its usual USD level at today's FX"""
    end = pd.Timestamp(end) if end is not None else load_fx()["date"].max()     # the log is checked up to the last FX estimate
    pairs_all, fx = make_pairs(dec), load_fx()
    rows = []
    # freq='MS' = start of each month. At each origin t: refit on the past only and forecast 1-6 months
    for t in pd.date_range(since, end, freq="MS"):
        params = fit_process(pairs_all, dec, as_of=t)
        fc = forecast(dec, params, months=months, n=n, as_of=t, **kw)
        fx_t = fx[fx["date"] < t]["buy_est"].iloc[-1]
        # Only months that already happened (<= end) can be compared with the truth
        for r in fc[fc["month"] <= end].itertuples():
            last = price_in_force(dec, r.product, t)
            rows.append(dict(origin=t, product=r.product, month=r.month, h=(r.month.year - t.year) * 12 + r.month.month - t.month,
                             actual=price_in_force(dec, r.product, r.month)["syp_new"],
                             p10=r.syp_new_p10, p50=r.syp_new_p50, p90=r.syp_new_p90,
                             unchanged=last["syp_new"], fx_pass=last["syp_new"] * fx_t / last["buy_est"],
                             usual_level=params["center"][r.product] * fx_t))
    return pd.DataFrame(rows)


def calibration(ev, level=0.9, min_n=12):
    """Split-conformal correction of p10 / p90 on the log scale, pooled over products and horizons (scaled by sqrt(h)).
    hi > 0: past prices landed above p90 too often, widen; hi < 0: the fan was too wide. Same for lo below p10.
    ponytail: rows overlap (3 products x 6 horizons per origin), so the effective n is far below len(ev); per-horizon
    corrections once there are a few years of origins."""
    # Conformal calibration: measure how often reality fell outside the fan in the past and derive a log correction that widens it.
    # With little data (< min_n) no correction is made. Dividing by sqrt(h) because uncertainty grows with the square root of time
    if len(ev) < min_n:
        return dict(lo=0.0, hi=0.0, n=len(ev))
    s, q = np.sqrt(ev["h"]), min(1.0, level * (1 + 1 / len(ev)))
    return dict(lo=float(np.quantile(np.log(ev["p10"] / ev["actual"]) / s, q)),
                hi=float(np.quantile(np.log(ev["actual"] / ev["p90"]) / s, q)), n=len(ev))


def calibrate_online(ev, **kw):
    """The fan each origin would have shown with calibration, using only past forecasts whose month had already come."""
    # Same calibration without cheating: for each origin use only forecasts whose month had ended before it (ev['month'] <= t)
    out = ev.copy()
    for t in ev["origin"].unique():
        m = ev["origin"] == t
        out.loc[m, "p10"], out.loc[m, "p90"] = _widen(ev.loc[m, "p10"], ev.loc[m, "p50"], ev.loc[m, "p90"],
                                                      calibration(ev[ev["month"] <= t], **kw), ev.loc[m, "h"])
    return out


def score(ev, fans):
    """Log errors x100 (~%). Baselines: MAE / RMSE. Fans (name -> evaluate-like frame): also P10-P90 coverage
    (target 80%), share above P90 and below P10 (target 10% each) and mean pinball loss over P10 / P50 / P90."""
    # Evaluation. For baselines: MAE and RMSE. For the fan: coverage (80% of truths should fall inside P10-P90)
    # and pinball loss = a metric that scores the tails and the median together
    y, rows = np.log(ev["actual"]), {}
    for name in ("unchanged", "fx_pass", "usual_level"):
        e = y - np.log(ev[name])
        rows[name] = dict(n=len(e), mae=100 * e.abs().mean(), rmse=100 * np.sqrt((e ** 2).mean()))
    for name, f in fans.items():
        e = y - np.log(f["p50"])
        pin = np.mean([np.maximum(q * d, (q - 1) * d).mean() for q, d in ((0.1, y - np.log(f["p10"])), (0.5, e), (0.9, y - np.log(f["p90"])))])
        above, below = f["actual"] > f["p90"] * (1 + 1e-9), f["actual"] < f["p10"] * (1 - 1e-9)   # sticky price: p10 = p90 = last price up to exp(log()) rounding
        rows[name] = dict(n=len(e), mae=100 * e.abs().mean(), rmse=100 * np.sqrt((e ** 2).mean()), pinball=100 * pin,
                          cover_80=1 - (above | below).mean(), above_p90=above.mean(), below_p10=below.mean())
    return pd.DataFrame(rows).T


def eia_brent_path():
    """EIA STEO of 2026-10-06: Brent about 105 USD/b in 4Q2026 and 84 on average in 2027 (linear fall 100 -> 68 during 2027)."""
    # Expected Brent path from EIA (key points + linear interpolation between months). Returns a function month -> price. Scenarios only
    knots = {"2026-09-01": 114.16, "2026-10-01": 105.0, "2026-12-01": 105.0, "2027-01-01": 100.0, "2027-12-01": 68.0}
    s = pd.Series(knots.values(), index=pd.to_datetime(list(knots))).resample("MS").interpolate()
    return lambda month: float(s.get(pd.Timestamp(month).to_period("M").to_timestamp(), s.iloc[-1]))



def export_html(dec, ev, params, calib, fc, bt, path=Path(__file__).parent / "fuel_forecast.html"):
    """Write the page's data block (between /* DATA START */ and /* DATA END */): parameters, calibration, past forecasts
    vs actual, the test scores and the series behind the notebook's charts. The page re-runs the normal/poisson Monte Carlo
    in the browser from these."""
    # Stage 6: export. Gather all results into a JSON dict and write it into fuel_forecast.html between the DATA START/END markers;
    # the page re-runs the simulation in the browser from these numbers.
    import json
    fx = load_fx()
    series = lambda s: [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.dropna().items()]   # Series -> list of [date, value]
    lines = dec.groupby(["date", "fuel"])["syp_new"].mean().unstack()
    g90 = dec[(dec["product"] == "gasoline_90") & (dec["tier"] == "unsubsidized") & (dec["date"] >= MARKET_SINCE)]
    recent = dec[(dec["tier"] == "unsubsidized") & (dec["date"] >= "2026-01-01")]
    charts = dict(
        fx=series(fx.set_index("date")["buy_est"].loc["2000":]),
        fuel={f: series(lines[f].loc["2000":]) for f in lines},
        brent=series(load_brent().loc["2024-01":]),
        g90_usd=series(g90.set_index("date")["usd"]),
        brent_corr=round(float(np.log(g90["usd"]).corr(np.log(g90["brent_l1"]))), 2),
        backtest=[[name, round(float(r["typical_error"]), 4), round(float(r["mae_log"]), 4), int(r["n_tests"])] for name, r in bt.iterrows()],
        fan={p: dict(observed=series(recent[recent["product"] == p].set_index("date")["syp_new"]),
                     forecast=[[m.strftime("%Y-%m-%d"), round(a, 2), round(b, 2), round(c, 2)]
                               for m, a, b, c in fc[fc["product"] == p][["month", "syp_new_p10", "syp_new_p50", "syp_new_p90"]].values])
             for p in PRODUCTS},
    )
    live = dec[(dec["tier"] == "unsubsidized") & dec["product"].isin(PRODUCTS)].sort_values("date")
    last = live.groupby("product").tail(1).set_index("product")
    end = fx["date"].max()
    sc = score(ev, {"model": ev, "model_calib_online": calibrate_online(ev)})
    data = dict(
        built=pd.Timestamp.today().strftime("%Y-%m-%d"), data_end=end.strftime("%Y-%m-%d"),
        base_month=pd.Timestamp.today().strftime("%Y-%m"), fx0=float(fx["buy_est"].iloc[-1]),
        vol=round(100 * fx_volatility(), 2), kappa=round(params["kappa"], 3), sigma=round(100 * params["sigma"], 2),
        calib={k: round(calib[k], 4) for k in ("lo", "hi")}, calib_n=calib["n"],
        products={p: dict(price=float(last.loc[p, "syp_new"]), date=last.loc[p, "date"].strftime("%Y-%m-%d"),
                          center=round(params["center"][p], 4), rate=round(params["per_month"][p], 4)) for p in PRODUCTS},
        actual={p: [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in g[g["date"] >= "2025-10-01"][["date", "syp_new"]].values]
                for p, g in live.groupby("product")},
        history=[dict(origin=r.origin.strftime("%Y-%m"), month=r.month.strftime("%Y-%m-%d"), product=r.product, h=int(r.h),
                      actual=round(r.actual, 2), p10=round(r.p10, 2), p50=round(r.p50, 2), p90=round(r.p90, 2)) for r in ev.itertuples()],
        score={name: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()} for name, row in sc.iterrows()},
        by_h={int(h): {k: round(float(v), 3) for k, v in score(g, {"m": g}).loc["m", ["n", "cover_80", "above_p90", "below_p10"]].items()}
              for h, g in ev.groupby("h")},
        charts=charts,
    )
    # Read the page and replace only the part between the two markers (the rest of the page is untouched). '</' is escaped so it cannot break the <script> tag
    html = path.read_text(encoding="utf8")
    a, b = html.index("/* DATA START */"), html.index("/* DATA END */")
    block = "/* DATA START */\nconst MODEL = " + json.dumps(data, ensure_ascii=False).replace("</", "<\\/") + ";\n"
    path.write_text(html[:a] + block + html[b:], encoding="utf8", newline="\n")


# Entry point: running the file directly executes the full pipeline in order: load -> pairs -> backtest -> evaluate -> calibrate -> forecast -> export
if __name__ == "__main__":
    pd.set_option("display.width", 200)

    # Load historical decisions
    dec = load_decisions()
    pairs = make_pairs(dec)

    print(
        f"{len(dec)} decisions, "
        f"{len(pairs)} consecutive-decision pairs since {MARKET_SINCE}"
    )

    # Compare models using expanding-window backtesting
    print("\nBacktest (expanding window, lower is better):")
    bt = backtest(pairs)
    print(bt.round(3))

    # Compare tested kappa values
    kappa_rows = bt[
        bt.index.to_series().str.startswith("error correction kappa=")
    ]

    if not kappa_rows.empty:
        best_kappa_name = kappa_rows["rmse_log"].idxmin()
        best_kappa = float(best_kappa_name.split("=")[1])

        print(f"\nBest tested kappa: {best_kappa:.4f}")
        print(
            "Backtest RMSE:",
            round(kappa_rows.loc[best_kappa_name, "rmse_log"], 4)
        )

    # Which drivers does the data support? (a driver joins the process only if |t| > 2)
    coefs, tails = factor_check(pairs)
    print("\nDrivers of a decision (OLS, in-sample):")
    print(coefs.round(3))
    print({k: round(v, 3) for k, v in tails.items()})

    # Rolling-origin test of the whole Monte Carlo forecast, against baselines, for each simulation variant
    # 4 combinations: noise (normal/bootstrap) x decision count (poisson/empirical)
    variants = {f"{a}/{b}": dict(noise=a, counts=b) for a in ("normal", "bootstrap") for b in ("poisson", "empirical")}
    evs = {name: evaluate(dec, **kw) for name, kw in variants.items()}
    ev = evs["normal/poisson"]
    fans = {**evs, "normal/poisson +online calib": calibrate_online(ev)}
    print(f"\nRolling-origin test ({ev['origin'].nunique()} monthly origins from {EVAL_SINCE}, horizons 1-6, log error x100):")
    print(score(ev, fans).round(3).to_string())
    print("\nnormal/poisson by horizon (targets: cover_80 = 0.8, above_p90 = 0.1):")
    print(pd.concat({h: score(g, {"m": g}).loc["m", ["n", "cover_80", "above_p90", "below_p10"]] for h, g in ev.groupby("h")}, axis=1).round(2))

    # Fit process parameters on everything, calibrate the fan on every realized past forecast
    params = fit_process(pairs, dec)
    calib = calibration(ev)
    print("\nFitted process:", {k: round(params[k], 3) for k in ("kappa", "sigma")},
          {k: {a: round(b, 2) for a, b in params[k].items()} for k in ("center", "per_month")})
    print("Calibration (log correction per sqrt(month)):", {k: round(v, 3) for k, v in calib.items()})

    # Forecast the next six months
    fc = forecast(dec, params, months=6, calib=calib)

    print("\nGasoline 90 forecast (calibrated):")
    print(
        fc[fc["product"] == "gasoline_90"]
        .round(2)
        .to_string(index=False)
    )

    export_html(dec, ev, params, calib, fc, bt)
    print("\nfuel_forecast.html data block updated")
