# =============================================================================
# Presentation-only edit — logic untouched.
# =============================================================================
"""
EnerVision AI — afternoon daydream dashboard for Home C dataset.
No emojis. No money. Unified states: Surplus / Balanced / Shortage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import warnings, subprocess, sys, json, joblib, base64
import torch
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction

st.set_page_config(page_title="EnerVision AI", page_icon="", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "HomecCleaned.csv"
XGB_FILE = BASE_DIR / "xgboost_Ver2.joblib"
XGB_PKL = BASE_DIR / "xgboost_Ver2.pkl"
SCALER_X = BASE_DIR / "scaler_X.joblib"
SCALER_Y = BASE_DIR / "scaler_y.joblib"
LOGO_FILE = BASE_DIR / "logo.png"           # <-- put your logo file here

# PatchTSMixer artifacts — MODEL SWITCH: after a fair, apples-to-apples 24-hour
# evaluation (same anchors, same ground truth), XGBoost's recursive forecast
# scored R² = -0.46 (worse than predicting the mean) on the genuine 24-hour
# task, because its recursive lag approximations don't match what it saw
# during training. PatchTSMixer natively forecasts all 24 hours in a single
# pass and scored R² = 0.16 on the same task — the only workable choice for
# this view. See the project report for the full comparison.
PATCH_MODEL_FILE = BASE_DIR / "patchtsmixer_model.pt"
PATCH_SCALER_FILE = BASE_DIR / "patchtsmixer_scaler.joblib"

# ------------------------------------------------------------------
# TARGET UNIFICATION: the deployed XGBoost model now predicts household
# Demand/Consumption ('use [kW]') instead of Net_Energy, matching the
# retrained notebooks. Appliance sub-meters (incl. 'House overall [kW]',
# an exact duplicate of 'use [kW]') are removed to avoid leakage, and
# Net_Energy_lag* is replaced with Demand_lag* (lags of the new target).
# ------------------------------------------------------------------
FEATURES = [
    "temperature","humidity","visibility",
    "apparentTemperature","pressure","windSpeed","cloudCover","windBearing",
    "precipIntensity","dewPoint","precipProbability","is_weekend","hour_sin",
    "hour_cos","month_sin","month_cos","Demand_lag1","Demand_lag60",
    "Demand_lag1440",
]

# ---- Load the logo from the folder as base64 (used inside the HTML brand) ----
if LOGO_FILE.exists():
    LOGO_B64 = base64.b64encode(LOGO_FILE.read_bytes()).decode()
else:
    LOGO_B64 = ""

# =========================================================
# MODEL
# =========================================================
# DEPRECATED — no longer called. Kept for reference only. A fair 24-hour
# evaluation showed this recursive scheme performs worse than predicting
# the historical mean (R² = -0.46), because the "last known value" used
# for Demand_lag1/Demand_lag60 at each recursive step does not match what
# the model actually saw during training (a true 1-minute-old value).
# Replaced below by predict_patchtsmixer_demand, which forecasts all 24
# hours natively in one pass — no recursion, no lag approximation needed.
def predict_xgb_recursive_isolated(df, anchor_row, hours=24):
    if not hasattr(anchor_row, "get"):
        raise TypeError(f"anchor_row must be Series, got {type(anchor_row)}")
    model_file = XGB_FILE if XGB_FILE.exists() else XGB_PKL
    if not model_file.exists():
        return None, None, f"Missing {model_file.name}"
    df_hist = df.dropna(subset=["Demand_kW"]).sort_values("time").reset_index(drop=True)
    anchor_time = anchor_row["time"]
    anchor_demand = float(anchor_row["Demand_kW"])
    future_times = pd.date_range(start=anchor_time + pd.Timedelta(hours=1),
                                 periods=hours, freq="h")
    refs = []
    for ft in future_times:
        ref_time = ft - pd.Timedelta(hours=24)
        ref_pos = int((df_hist["time"] - ref_time).abs().argmin())
        ref_row = df_hist.iloc[ref_pos]
        refs.append({f: ref_row.get(f, np.nan) for f in FEATURES})
    worker = r"""
import json, sys, warnings
import pandas as pd, numpy as np, joblib
model_path = sys.argv[1]
payload = json.loads(sys.stdin.read())
FEATURES = payload["features"]; refs = payload["refs"]
anchor_demand = float(payload["anchor_demand"])
future_times = pd.to_datetime(payload["future_times"])
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list): pass
    _RemainderColsList.__module__ = ct.__name__
    ct._RemainderColsList = _RemainderColsList
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load(model_path)
preds = []; last = anchor_demand
for i, ft in enumerate(future_times):
    r = dict(refs[i])
    h = ft.hour + ft.minute / 60; m = ft.month
    r["is_weekend"] = float(ft.dayofweek >= 5)
    r["hour_sin"] = float(np.sin(2*np.pi*h/24))
    r["hour_cos"] = float(np.cos(2*np.pi*h/24))
    r["month_sin"] = float(np.sin(2*np.pi*m/12))
    r["month_cos"] = float(np.cos(2*np.pi*m/12))
    r["Demand_lag1"] = last; r["Demand_lag60"] = last
    r["Demand_lag1440"] = float(refs[i].get("Demand_kW", anchor_demand))
    X = pd.DataFrame([r])[FEATURES].apply(pd.to_numeric, errors="coerce")
    p = float(model.predict(X)[0]); preds.append(p); last = p
print(json.dumps({"predictions": preds}))
"""
    payload = {"features": FEATURES, "refs": refs, "anchor_demand": anchor_demand,
               "future_times": [str(t) for t in future_times]}
    try:
        proc = subprocess.run([sys.executable, "-c", worker, str(model_file)],
                              input=json.dumps(payload), text=True,
                              capture_output=True, timeout=300)
    except Exception as e:
        return None, None, str(e)
    if proc.returncode != 0:
        return None, None, (proc.stderr or proc.stdout or "worker error")[-1500:]
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        return result["predictions"], future_times, None
    except Exception as e:
        return None, None, f"invalid worker output: {e}"


# ---- PatchTSMixer: the model actually used for the "Predict 24h" view ----
PATCH_TARGET_COL = "use [kW]"
PATCH_WEATHER_COLS = ["temperature", "humidity", "visibility", "apparentTemperature",
                      "pressure", "windSpeed", "cloudCover", "windBearing",
                      "precipIntensity", "dewPoint", "precipProbability"]
PATCH_CALENDAR_COLS = ["is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos"]
PATCH_FEATURE_COLS = [PATCH_TARGET_COL] + PATCH_WEATHER_COLS + PATCH_CALENDAR_COLS
PATCH_TARGET_IDX = 0
PATCH_CONTEXT_LENGTH = 96   # last 96 HOURS (4 days) of history
PATCH_PREDICTION_LENGTH = 24  # next 24 HOURS, forecast natively in one pass


@st.cache_resource
def load_patchtsmixer():
    """Load the PatchTSMixer weights + scaler once per session."""
    config = PatchTSMixerConfig(
        context_length=PATCH_CONTEXT_LENGTH,
        prediction_length=PATCH_PREDICTION_LENGTH,
        num_input_channels=len(PATCH_FEATURE_COLS),
        prediction_channel_indices=[PATCH_TARGET_IDX],
        patch_length=16, patch_stride=8, d_model=64, num_layers=3, dropout=0.2,
    )
    model = PatchTSMixerForPrediction(config)
    model.load_state_dict(torch.load(PATCH_MODEL_FILE, map_location="cpu"))
    model.eval()
    scaler = joblib.load(PATCH_SCALER_FILE)
    return model, scaler


@st.cache_data
def build_hourly_frame(_df_in):
    """Hourly-resampled frame PatchTSMixer expects: target + weather as
    hourly means, calendar features recomputed directly at hourly
    resolution — the same construction used when the model was trained.
    Parameter is prefixed with `_` so Streamlit's cache does not try to
    hash the (large) DataFrame itself."""
    d = _df_in.set_index("time")
    agg = {PATCH_TARGET_COL: "mean"}
    agg.update({c: "mean" for c in PATCH_WEATHER_COLS if c in d.columns})
    hourly = d.resample("h").agg(agg).dropna().reset_index()
    hourly["is_weekend"] = (hourly["time"].dt.dayofweek >= 5).astype(float)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["time"].dt.hour / 24.0)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["time"].dt.hour / 24.0)
    hourly["month_sin"] = np.sin(2 * np.pi * hourly["time"].dt.month / 12.0)
    hourly["month_cos"] = np.cos(2 * np.pi * hourly["time"].dt.month / 12.0)
    return hourly.set_index("time")


@st.cache_data
def hourly_solar_climatology(_df_in):
    """Historical-average Solar generation by hour-of-day. This is a
    simple statistical ESTIMATE, not a model prediction — no model in
    this project forecasts future solar generation. It is used only to
    reconstruct an approximate future Net Energy for the 24h view, and
    is labeled as an estimate everywhere it appears in the UI.
    Parameter is prefixed with `_` so Streamlit's cache does not try to
    hash the (large) DataFrame itself."""
    d = _df_in.dropna(subset=["Solar_kW"]).copy()
    d["hour"] = d["time"].dt.hour
    return d.groupby("hour")["Solar_kW"].mean()


def predict_patchtsmixer_demand(hourly_df, anchor_time, hours=24):
    """Forecast the next `hours` hours of Demand in a single native pass
    (no recursion). Returns (demand_kW_array, future_times, error_or_None)."""
    context = hourly_df.loc[:anchor_time].tail(PATCH_CONTEXT_LENGTH)
    if len(context) < PATCH_CONTEXT_LENGTH:
        return None, None, (
            f"Not enough hourly history before {anchor_time} "
            f"(need {PATCH_CONTEXT_LENGTH}h, have {len(context)}h)."
        )
    if not PATCH_MODEL_FILE.exists() or not PATCH_SCALER_FILE.exists():
        return None, None, f"Missing {PATCH_MODEL_FILE.name} or {PATCH_SCALER_FILE.name}"

    model, scaler = load_patchtsmixer()
    x = scaler.transform(context[PATCH_FEATURE_COLS].values)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(past_values=x)

    # Channel width returned by prediction_outputs varies by transformers
    # version even with prediction_channel_indices set — handle both.
    raw = out.prediction_outputs.squeeze(0).numpy()
    if raw.shape[-1] == 1:
        mean_, scale_ = scaler.mean_[PATCH_TARGET_IDX], scaler.scale_[PATCH_TARGET_IDX]
        demand = raw.flatten() * scale_ + mean_
    else:
        inv = scaler.inverse_transform(raw)
        demand = inv[:, PATCH_TARGET_IDX]

    future_times = pd.date_range(start=anchor_time + pd.Timedelta(hours=1), periods=hours, freq="h")
    return demand, future_times, None


# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Net_Energy" not in df.columns:
        df["Net_Energy"] = (
            pd.to_numeric(df.get("gen [kW]", 0), errors="coerce").fillna(0)
            - pd.to_numeric(df.get("use [kW]", 0), errors="coerce").fillna(0))
    if "Solar_kW" not in df.columns:
        solar_col = "gen [kW]" if "gen [kW]" in df.columns else "Solar [kW]"
        df["Solar_kW"] = pd.to_numeric(df.get(solar_col, np.nan), errors="coerce")
    hour = df["time"].dt.hour + df["time"].dt.minute/60
    month = df["time"].dt.month
    df["is_weekend"] = df["time"].dt.dayofweek.ge(5).astype(float)
    df["hour_sin"] = np.sin(2*np.pi*hour/24); df["hour_cos"] = np.cos(2*np.pi*hour/24)
    df["month_sin"] = np.sin(2*np.pi*month/12); df["month_cos"] = np.cos(2*np.pi*month/12)
    # Net_Energy lags: kept for the REAL-TIME status classifier ("Now" view),
    # which uses the actual current Solar & Demand values, not a prediction.
    df["Net_Energy_lag1"] = df["Net_Energy"].shift(1)
    df["Net_Energy_lag60"] = df["Net_Energy"].shift(60)
    df["Net_Energy_lag1440"] = df["Net_Energy"].shift(1440)
    # Demand lags: NEW - used as inputs by the retrained forecasting model,
    # which now predicts Demand ('use [kW]') instead of Net_Energy.
    if "Demand_kW" not in df.columns:
        df["Demand_kW"] = pd.to_numeric(df.get("use [kW]", np.nan), errors="coerce")
    df["Demand_lag1"] = df["Demand_kW"].shift(1)
    df["Demand_lag60"] = df["Demand_kW"].shift(60)
    df["Demand_lag1440"] = df["Demand_kW"].shift(1440)
    return df


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}"); st.stop()
try:
    df = load_data(str(DATA_FILE))
except Exception as e:
    st.error(f"Load error: {e}"); st.stop()
df_clean = df.dropna(subset=["Net_Energy"]).sort_values("time").reset_index(drop=True)

# =========================================================
# THEME  — Anything (afternoon daydream over a wildflower field)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&family=Instrument+Sans:wght@400;600&display=swap');

:root {
    --bg: #ffffff;
    --surface: rgba(255,255,255,0.35);
    --surface-2: #ffffff;
    --border: rgba(37,99,235,0.16);
    --border-hover: rgba(37,99,235,0.34);
    --border-soft: rgba(37,99,235,0.12);
    --shadow: rgba(0,0,0,0.10) 0px 10px 15px -3px,
              rgba(0,0,0,0.10) 0px 4px 6px -4px;
    --shadow-soft: rgba(0,0,0,0.05) 0px 6px 14px -4px;

    --text: #18191b;
    --text-2: #6a6a6c;
    --text-3: #acadae;

    --accent: #18191b;                 /* label color for pickers (black) */
    --accent-soft: rgba(24,25,27,.06);
    --accent-grad: linear-gradient(116deg, #2563eb, #60a5fa);
    --blue: #2563eb;

    --mint: #dbe7fb;

    --pos: #10B981;
    --pos-soft: rgba(16,185,129,.12);
    --neg: #EF4444;
    --neg-soft: rgba(239,68,68,.10);
    --neu: #F59E0B;
    --neu-soft: rgba(245,158,11,.12);
    --info: #2563eb;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    letter-spacing: 0.02em;
}
.stApp {
    background:
        radial-gradient(900px 480px at 90% -8%, rgba(96,165,250,.20), transparent 62%),
        radial-gradient(700px 420px at -5% 2%, rgba(147,197,253,.16), transparent 60%),
        #ffffff;
    font-variant-numeric: tabular-nums;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.main .block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1280px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 32px 40px 80px !important;
}
@media (max-width: 640px) {
    .main .block-container,
    [data-testid="stMainBlockContainer"] { padding: 24px 16px 64px !important; }
}
div[data-testid="stVerticalBlock"] { gap: 16px; }

/* =============================================================
   FIX 1 — Center the Now / Predict 24h bar, options side by side
   ============================================================= */
.st-key-view_mode,
.st-key-view_mode div[data-testid="stRadio"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}
.st-key-view_mode div[data-testid="stRadio"] > div {
    width: auto !important;
    display: flex !important;
    justify-content: center !important;
}

/* the pill itself: one row, shrink-wrapped, centered */
.st-key-view_mode [role="radiogroup"],
.stRadio [role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;      /* Now | Predict 24h side by side */
    flex-wrap: nowrap !important;
    justify-content: center !important;
    align-items: center !important;
    width: fit-content !important;
    margin: 0 auto !important;
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(18px) saturate(150%);
    -webkit-backdrop-filter: blur(18px) saturate(150%);
    border: 1px solid var(--border-soft) !important;
    border-radius: 9999px !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: var(--shadow-soft) !important;
}

.stRadio label,
div[data-testid="stRadio"] label {
    flex: 0 0 auto !important;
    width: auto !important;
    white-space: nowrap !important;      /* stops "Predict 24h" from wrapping */
    background: transparent !important;
    border: 1px solid transparent !important;
    padding: 8px 22px !important;
    border-radius: 9999px !important;
    color: var(--text-3) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    margin: 0 !important;
    letter-spacing: 0.02em;
    transition: color .15s ease, background .15s ease;
}
.stRadio label:hover { color: var(--text) !important; }
.stRadio label:has(input:checked) {
    background: #18191b !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
.stRadio label p { color: inherit !important; font-size: 13px !important; font-weight: inherit !important; }
.stRadio input { display: none !important; }
.stRadio label > div:first-child { display: none !important; }

/* remove any extra highlight / outline / yellow button inside the bar */
.st-key-view_mode label,
.st-key-view_mode label * {
    outline: none !important;
    box-shadow: none !important;
    background-image: none !important;
}
.st-key-view_mode label:not(:has(input:checked)),
.st-key-view_mode label:not(:has(input:checked)) * {
    background: transparent !important;
    background-color: transparent !important;
}
.st-key-view_mode label:has(input:checked) {
    background: #18191b !important;
    background-color: #18191b !important;
}
.st-key-view_mode label:has(input:checked) * {
    background: transparent !important;
    color: #ffffff !important;
}

/* =============================================================
   FIX 2 — FORCE every KPI card to the SAME fixed height
   ============================================================= */
.card.card-kpi {
    min-height: 190px !important;
    height: 190px !important;
    max-height: 190px !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
}
.card.card-kpi .micro-label { flex: 0 0 auto !important; }
.card.card-kpi .card-value  { flex: 0 0 auto !important; }
.card.card-kpi .totals-row  { flex: 0 0 auto !important; }
.card.card-kpi .card-sub {
    margin-top: auto !important;
    flex: 0 0 auto !important;
    padding-top: 12px !important;
}

/* ---------- Brand ---------- */
.brand { display:flex; align-items:center; gap:10px; }
.brand-logo {
    width:40px; height:40px; object-fit:contain; display:block;
}
.brand-name {
    font-family: 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-size: 26px; font-weight: 400; color: var(--text);
    letter-spacing: normal;
    line-height: 1;
}
.clock {
    font-family: 'Inter', sans-serif;
    font-size: 12px; color: var(--text-3);
    font-weight: 400; letter-spacing: 0.02em;
}

/* =====================================================
   DATE / TIME PICKERS
   ===================================================== */
div[data-testid="stDateInput"],
div[data-testid="stSelectbox"] { width: 100%; }

/* ---- ONE shared shell for BOTH boxes: same white, border, radius, height ---- */
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 2px 16px !important;
    min-height: 46px !important;
    box-shadow: var(--shadow-soft) !important;
    display: flex !important;
    align-items: center !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stDateInput"] [data-baseweb="input"]:hover,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:hover {
    border-color: var(--border-hover) !important;
}
div[data-testid="stDateInput"] [data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.14) !important;
}

/* ---- Inner layers: fully transparent so no grey shows through ---- */
div[data-testid="stDateInput"] [data-baseweb="base-input"],
div[data-testid="stDateInput"] [data-baseweb="input"] > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ---- Same text style in both ---- */
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div,
div[data-testid="stSelectbox"] input {
    background: transparent !important;
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    text-align: center !important;
}
div[data-testid="stDateInput"] input { border: none !important; padding: 8px 0 !important; }
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child > div {
    justify-content: center !important;
    padding: 8px 0 !important;
}

div[data-testid="stDateInput"] svg,
div[data-testid="stSelectbox"] svg { color: var(--accent) !important; }

/* ---- Force light look + visible text in BOTH boxes (beats dark theme) ---- */
div[data-testid="stDateInput"],
div[data-testid="stSelectbox"] { color-scheme: light; }

div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="input"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
    background-color: #ffffff !important;
    color: #18191b !important;
    -webkit-text-fill-color: #18191b !important;
    opacity: 1 !important;
}
div[data-testid="stDateInput"] svg,
div[data-testid="stSelectbox"] svg {
    color: #2563eb !important;
    fill: #2563eb !important;
}

/* dropdown list that opens under the Time box */
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: #ffffff !important;
    color: #18191b !important;
}
div[data-baseweb="popover"] li:hover { background-color: #f1f1f1 !important; }

/* ---- Time box: brute force, every layer white + dark visible text ---- */
div[data-testid="stSelectbox"] div,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] [role="combobox"],
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"] * {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #18191b !important;
    -webkit-text-fill-color: #18191b !important;
    caret-color: transparent !important;
    opacity: 1 !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    min-height: 46px !important;
    box-shadow: var(--shadow-soft) !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
    border: none !important;
    box-shadow: none !important;
    justify-content: center !important;
}
div[data-testid="stSelectbox"] svg {
    color: #2563eb !important;
    fill: #2563eb !important;
    background: transparent !important;
}

/* ---- Date box: EXACT same design as the Time box ---- */
div[data-testid="stDateInput"] div,
div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] [data-baseweb="input"],
div[data-testid="stDateInput"] [data-baseweb="base-input"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #18191b !important;
    -webkit-text-fill-color: #18191b !important;
    opacity: 1 !important;
}
div[data-testid="stDateInput"] [data-baseweb="input"] {
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    min-height: 46px !important;
    padding: 0 16px !important;
    box-shadow: var(--shadow-soft) !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stDateInput"] [data-baseweb="base-input"] {
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stDateInput"] input {
    border: none !important;
    box-shadow: none !important;
    text-align: center !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    padding: 8px 0 !important;
}
div[data-testid="stDateInput"] svg {
    color: #2563eb !important;
    fill: #2563eb !important;
    background: transparent !important;
}

/* ---- Date + Time: dark visible text, blue icons, blue focus, NO yellow ---- */
div[data-testid="stDateInput"][data-testid="stDateInput"] input,
div[data-testid="stDateInput"][data-testid="stDateInput"] [data-baseweb="input"] *,
div[data-testid="stSelectbox"][data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #18191b !important;
    -webkit-text-fill-color: #18191b !important;
}
div[data-testid="stDateInput"][data-testid="stDateInput"] svg,
div[data-testid="stSelectbox"][data-testid="stSelectbox"] svg {
    color: #2563eb !important;
    fill: #2563eb !important;
    -webkit-text-fill-color: #2563eb !important;
}
div[data-testid="stDateInput"][data-testid="stDateInput"] [data-baseweb="input"]:hover,
div[data-testid="stDateInput"][data-testid="stDateInput"] [data-baseweb="input"]:focus-within,
div[data-testid="stSelectbox"][data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
div[data-testid="stSelectbox"][data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,.14) !important;
}

/* calendar popup: selected / hovered day in blue */
div[data-baseweb="calendar"] [aria-selected="true"],
div[data-baseweb="calendar"] [aria-selected="true"] * {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
div[data-baseweb="calendar"] [role="gridcell"]:hover,
div[data-baseweb="calendar"] [role="gridcell"]:hover * {
    background-color: #dbe7fb !important;
    color: #18191b !important;
}
div[data-baseweb="calendar"] [role="gridcell"] > div { border-color: #2563eb !important; }

/* time dropdown: selected / hovered option in blue */
div[data-baseweb="popover"] li[aria-selected="true"],
div[data-baseweb="popover"] li:hover {
    background-color: #dbe7fb !important;
    color: #18191b !important;
}

label[data-testid="stWidgetLabel"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    color: var(--accent) !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
}

/* ---------- Cards ---------- */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    height: 100%; box-sizing: border-box;
    box-shadow: var(--shadow-soft);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    backdrop-filter: blur(14px) saturate(140%);
    transition: border-color .15s ease, box-shadow .15s ease, transform .15s ease;
}
.card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow);
    transform: translateY(-1px);
}
.card-hero {
    border-radius: 20px; padding: 40px 32px; text-align: center;
    background: linear-gradient(135deg, rgba(219,231,251,.60) 0%, rgba(255,255,255,.30) 55%, rgba(191,219,254,.40) 100%);
}

[class*="st-key-chart"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    box-sizing: border-box;
    box-shadow: var(--shadow-soft);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    backdrop-filter: blur(14px) saturate(140%);
    gap: 14px;
    transition: border-color .15s ease, box-shadow .15s ease;
}
[class*="st-key-chart"]:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow);
}

/* ---------- Type ---------- */
.micro-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 12px;
}
.card-value {
    font-family: 'Instrument Sans', 'Inter', sans-serif;
    font-size: 34px; font-weight: 600;
    letter-spacing: -0.01em; line-height: 1.2;
    color: var(--text);
}
.card-unit {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 400;
    color: var(--text-3); margin-left: 6px;
    letter-spacing: 0.02em;
}
.card-sub {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 400;
    color: var(--text-2);
    margin-top: 10px; line-height: 1.5;
    letter-spacing: 0.02em;
}
.pos { color: var(--pos); }
.neg { color: var(--neg); }
.neu { color: var(--neu); }
.info { color: var(--info); }

.hero-sentence {
    font-family: 'Instrument Sans', 'Inter', sans-serif;
    font-size: 30px; font-weight: 600;
    color: var(--text); line-height: 1.2;
    letter-spacing: normal;
}
.hero-facts {
    font-family: 'Inter', sans-serif;
    font-size: 12px; color: var(--text-3);
    margin-top: 14px; letter-spacing: 0.02em;
    font-weight: 400;
}
.hero-badge-wrap { margin-top: 20px; display: flex; justify-content: center; }

.sec-head {
    font-family: 'Instrument Sans', 'Inter', sans-serif;
    font-size: 20px; font-weight: 600;
    color: var(--text); margin: 0 0 6px;
    letter-spacing: normal;
    line-height: 1.4;
}
.sec-sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px; font-weight: 400;
    color: var(--text-3); margin: 0;
    letter-spacing: 0.02em;
}

.status-badge {
    display: inline-block;
    padding: 7px 18px;
    border-radius: 9999px;
    white-space: nowrap;
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-pos { background: var(--pos-soft); color: var(--pos); }
.badge-neg { background: var(--neg-soft); color: var(--neg); }
.badge-neu { background: var(--neu-soft); color: var(--neu); }

.actions { list-style: disc; padding: 0 0 0 18px; margin: 2px 0 0; }
.actions li {
    padding: 6px 0;
    font-family: 'Inter', sans-serif;
    font-size: 14px; color: var(--text);
    font-weight: 400;
    letter-spacing: 0.02em;
}
.actions li::marker { color: var(--blue); }

.totals-row { display: flex; gap: 10px; }
.totals-cell { flex: 1; min-width: 0; }
.totals-num {
    font-family: 'Instrument Sans', 'Inter', sans-serif;
    font-size: 20px; font-weight: 600;
    letter-spacing: -0.01em; line-height: 1.2;
}
.totals-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 11px; color: var(--text-3);
    margin-top: 6px; font-weight: 400;
    letter-spacing: 0.02em;
}

h1,h2,h3,h4 {
    color: var(--text);
    font-family: 'Instrument Sans', 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: normal;
}
.js-plotly-plot .plotly .modebar { display: none !important; }

.footer-line {
    border-top: 1px solid transparent;
    border-image: linear-gradient(90deg,
        rgb(172,173,174) 0%,
        rgb(172,173,174) 30%,
        rgb(96,165,250) 35%,
        rgb(153,183,250) 40%,
        rgb(172,173,174) 45%,
        rgb(172,173,174) 100%) 1;
    padding-top: 14px; margin-top: 12px;
    font-size: 12px; color: var(--text-3); text-align: center;
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

# Chart colors — the LINE is BLUE (matches the bars)
C_POS, C_NEG, C_NEU = "#10B981", "#EF4444", "#F59E0B"
C_INFO = "#2563eb"                 # <-- graph line + bar color (was #18191b)
C_TEXT, C_TEXT_2, C_TEXT_3, C_BORDER = "#18191b", "#6a6a6c", "#acadae", "#c4c4c4"
C_SURFACE = "#f9f9f9"
TONE = {"Surplus": "pos", "Balanced": "neu", "Shortage": "neg"}

def sign_tone(v):
    if v > 0: return "pos"
    if v < 0: return "neg"
    return "neu"

# =========================================================
# TOP BAR — brand left (logo loaded from folder), clock right
# =========================================================
tb_l, tb_spacer, tb_r = st.columns([2, 3, 2], gap="medium")
with tb_l:
    if LOGO_B64:
        st.markdown(
            f'<div class="brand"><img class="brand-logo" '
            f'src="data:image/png;base64,{LOGO_B64}" alt="EnerVision AI logo">'
            '<div class="brand-name">EnerVision AI</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand">'
                    '<div class="brand-name">EnerVision AI</div></div>',
                    unsafe_allow_html=True)
with tb_spacer:
    st.empty()
with tb_r:
    st.markdown(f"<div style='text-align:right; padding-top:8px;'>"
                f"<span class='clock'>{pd.Timestamp.now():%H:%M}</span></div>",
                unsafe_allow_html=True)

# =========================================================
# CENTERED VIEW SELECTOR — own row, same proportions as pickers
# =========================================================
v_l, v_mid, v_r = st.columns([1, 2.2, 1], gap="medium")
with v_mid:
    view = st.radio("View", ["Now", "Predict 24h"],
                    horizontal=True, label_visibility="collapsed", key="view_mode")

# =========================================================
# HELPERS
# =========================================================
def classify(v):
    if v > 1.5: return "Surplus"
    if v >= 0:  return "Balanced"
    return "Shortage"

def badge_class(s):
    return {"Surplus":"badge-pos","Balanced":"badge-neu","Shortage":"badge-neg"}[s]

def style_plot(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif",
                  color="#6a6a6c", size=11),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#c4c4c4",
                        font=dict(color="#18191b",
                                  family="Inter, system-ui, sans-serif",
                                  size=12)),
        legend=dict(orientation="h", y=1.15, x=0,
                    font=dict(size=11, color="#6a6a6c"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(196,196,196,.45)", linecolor="rgba(0,0,0,0)",
                   zeroline=False, ticks="",
                   tickfont=dict(size=11, color="#acadae")),
        yaxis=dict(gridcolor="rgba(196,196,196,.45)", linecolor="rgba(0,0,0,0)",
                   zeroline=False, ticks="",
                   tickfont=dict(size=11, color="#acadae")),
    )
    return fig

# =========================================================
# VIEW 1 — NOW
# =========================================================
if view == "Now":
    sp_left, mid_block, sp_right = st.columns([1, 2.2, 1], gap="medium")
    with mid_block:
        sub_date, sub_time = st.columns(2, gap="medium")
        with sub_date:
            sel_date = st.date_input("Date",
                value=df_clean["time"].max().date(),
                min_value=df_clean["time"].min().date(),
                max_value=df_clean["time"].max().date())
        day_rows = df_clean[df_clean["time"].dt.date == sel_date]
        with sub_time:
            if not day_rows.empty:
                sel_time = st.selectbox("Time",
                    options=day_rows["time"].dt.strftime("%H:%M").tolist(),
                    index=len(day_rows) - 1)
            else:
                st.info("No data"); st.stop()

    sel_dt = pd.Timestamp(f"{sel_date} {sel_time}")
    cur = df_clean.iloc[int((df_clean["time"] - sel_dt).abs().argmin())]

    gen_now = float(cur.get("gen [kW]", 0) or 0)
    use_now = float(cur.get("use [kW]", 0) or 0)
    net_now = gen_now - use_now
    status = classify(net_now)

    today = df_clean[df_clean["time"].dt.date == cur["time"].date()]
    today_so_far = today[today["time"] <= cur["time"]]
    today_solar = today_so_far["gen [kW]"].sum()
    today_used = today_so_far["use [kW]"].sum()
    today_from_grid = max(today_used - today_solar, 0)

    if status == "Surplus":
        sentence = f"Making more power than the house uses. Sending {net_now:.2f} kW to the grid."
    elif status == "Balanced":
        sentence = "Making exactly as much as the house uses. Nothing flows to or from the grid."
    else:
        sentence = f"Making less than the house uses. Pulling {abs(net_now):.2f} kW from the grid."

    st.markdown(f"""
    <div class="card card-hero">
        <div class="micro-label">Right now</div>
        <div class="hero-sentence">{sentence}</div>
        <div class="hero-facts">{cur['time']:%A, %d %B %Y · %H:%M}</div>
        <div class="hero-badge-wrap">
            <span class="status-badge {badge_class(status)}">{status.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Solar made</div>
            <div class="card-value">{gen_now:.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">Power from the panels right now.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Home used</div>
            <div class="card-value">{use_now:.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">Power the house draws right now.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        tone = sign_tone(net_now)
        if net_now > 1.5:
            direction = f"Sending {net_now:.2f} kW to the grid."
        elif net_now >= 0:
            direction = "Balanced. Nothing flows either way."
        else:
            direction = f"Pulling {abs(net_now):.2f} kW from the grid."
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Net power</div>
            <div class="card-value" style="color:var(--{tone});">{net_now:+.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">{direction}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Today&rsquo;s totals so far (kWh)</div>
            <div class="totals-row">
                <div class="totals-cell">
                    <div class="totals-num pos">{today_solar:.1f}</div>
                    <div class="totals-lbl">Solar made</div>
                </div>
                <div class="totals-cell">
                    <div class="totals-num info">{today_used:.1f}</div>
                    <div class="totals-lbl">Home used</div>
                </div>
                <div class="totals-cell">
                    <div class="totals-num neg">{today_from_grid:.1f}</div>
                    <div class="totals-lbl">From grid</div>
                </div>
            </div>
            <div class="card-sub" style="margin-top:10px;">Up to {cur['time']:%H:%M}</div>
        </div>""", unsafe_allow_html=True)

    ch_l, ch_r = st.columns([1, 1], gap="medium")

    with ch_l:
        with st.container(key="chart-now-detail"):
            st.markdown('<div class="sec-head">Today in detail</div>'
                        '<div class="sec-sub">Line above the middle is surplus. Below is shortage.</div>',
                        unsafe_allow_html=True)

            net_series = today["gen [kW]"] - today["use [kW]"]
            ymax = max(float(net_series.max()) * 1.5, 0.5)
            ymin = min(float(net_series.min()) * 1.5, -0.5)

            fig = go.Figure()
            fig.add_hrect(y0=0, y1=ymax, fillcolor="rgba(16,185,129,.08)",
                          line_width=0, layer="below")
            fig.add_hrect(y0=ymin, y1=0, fillcolor="rgba(239,68,68,.07)",
                          line_width=0, layer="below")

            fig.add_trace(go.Scatter(
                x=today["time"], y=net_series,
                mode="lines", name="Net power",
                line=dict(color=C_INFO, width=2.5, shape="spline"),   # BLUE line
                hovertemplate="%{x|%H:%M}<br>Net %{y:.2f} kW<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color=C_TEXT_3, line_width=1)

            fig.add_annotation(
                x=today["time"].iloc[2], y=ymax * 0.82,
                text="<b>SURPLUS — sending to grid</b>",
                showarrow=False, xanchor="left",
                font=dict(color=C_POS, size=12, family="Inter, sans-serif"),
            )
            fig.add_annotation(
                x=today["time"].iloc[2], y=ymin * 0.82,
                text="<b>SHORTAGE — pulling from grid</b>",
                showarrow=False, xanchor="left",
                font=dict(color=C_NEG, size=12, family="Inter, sans-serif"),
            )

            fig.update_layout(
                xaxis_title="Hour of day",
                yaxis_title="Net power (kW)",
                showlegend=False,
            )
            st.plotly_chart(style_plot(fig, 340), use_container_width=True,
                            config={"displayModeBar": False})

    with ch_r:
        with st.container(key="chart-now-rooms"):
            st.markdown('<div class="sec-head">Where the power went</div>'
                        '<div class="sec-sub">Total energy used today by each room (kWh).</div>',
                        unsafe_allow_html=True)
            area = {
                "Kitchen":     today["Dishwasher [kW]"].sum()+today["Fridge [kW]"].sum()+today["Microwave [kW]"].sum()+today["Kitchen 12 [kW]"].sum()+today["Kitchen 14 [kW]"].sum()+today["Kitchen 38 [kW]"].sum(),
                "Heating":     today["Furnace 1 [kW]"].sum()+today["Furnace 2 [kW]"].sum(),
                "Home office": today["Home office [kW]"].sum(),
                "Living room": today["Living room [kW]"].sum(),
                "Barn":        today["Barn [kW]"].sum(),
                "Garage":      today["Garage door [kW]"].sum(),
                "Wine cellar": today["Wine cellar [kW]"].sum(),
                "Utilities":   today["Well [kW]"].sum(),
            }
            adf = pd.DataFrame(list(area.items()), columns=["Area","kWh"]).sort_values("kWh")
            fig2 = go.Figure(go.Bar(
                x=adf["kWh"], y=adf["Area"], orientation="h",
                marker=dict(color=adf["kWh"],
                            colorscale=[[0, "#bfdbfe"], [1, "#2563eb"]],
                            line=dict(width=0)),
                text=[f"{v:.1f}" for v in adf["kWh"]],
                textposition="outside",
                textfont=dict(color=C_TEXT_2, size=11,
                              family="Inter, sans-serif"),
                hovertemplate="%{y}: %{x:.2f} kWh<extra></extra>",
            ))
            fig2.update_layout(xaxis=dict(title="Energy (kWh)", showgrid=False),
                               yaxis=dict(title="Room", showgrid=False),
                               showlegend=False)
            st.plotly_chart(style_plot(fig2, 340), use_container_width=True,
                            config={"displayModeBar": False})

    if status == "Surplus":
        actions = ["Run heavy appliances now",
                   "Charge the electric car before 16:00"]
    elif status == "Balanced":
        actions = ["Nothing to change"]
    else:
        actions = ["Delay laundry and dishwasher to the next sunny hour",
                   "Reduce heating or air conditioning"]
    items = "".join(f"<li>{a}</li>" for a in actions)
    st.markdown(f"""
    <div class="card">
        <div class="micro-label">What to do</div>
        <ul class="actions">{items}</ul>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# VIEW 2 — PREDICT 24h
# =========================================================
else:
    sp_left, mid_block, sp_right = st.columns([1, 2.2, 1], gap="medium")
    with mid_block:
        sub_date, sub_time = st.columns(2, gap="medium")
        with sub_date:
            pred_date = st.date_input("Start date",
                value=df_clean["time"].max().date(),
                min_value=df_clean["time"].min().date(),
                max_value=df_clean["time"].max().date())
        day_rows = df_clean[df_clean["time"].dt.date == pred_date]
        with sub_time:
            if not day_rows.empty:
                pred_time = st.selectbox("Start hour",
                    options=day_rows["time"].dt.strftime("%H:%M").tolist(),
                    index=len(day_rows) - 1)
            else:
                st.info("No data"); st.stop()

    anchor_dt = pd.Timestamp(f"{pred_date} {pred_time}")
    anchor_row = df_clean.iloc[int((df_clean["time"] - anchor_dt).abs().argmin())]

    hourly_df = build_hourly_frame(df)
    anchor_hour = anchor_dt.floor("h")

    with st.spinner("Forecasting..."):
        demand_preds, times, perr = predict_patchtsmixer_demand(hourly_df, anchor_hour, 24)

    if perr:
        st.error(f"Forecast failed: {perr}"); st.stop()

    demand_preds = np.array(demand_preds)

    # ------------------------------------------------------------------
    # Net Energy = Solar - Demand. The model above forecasts Demand only;
    # no model in this project forecasts future Solar generation, so the
    # Solar side is approximated using the historical average generation
    # for that hour of day (climatology) -- a simple statistical estimate,
    # not a model prediction. This is disclosed to the user below the
    # chart. See the project report ("Predict 24h" design discussion).
    # ------------------------------------------------------------------
    solar_clim = hourly_solar_climatology(df_clean)
    solar_estimates = np.array([solar_clim.get(t.hour, solar_clim.mean()) for t in times])
    preds = solar_estimates - demand_preds

    avg = float(preds.mean())
    peak = float(preds.max()); peak_h = times[int(preds.argmax())]
    trough = float(preds.min()); trough_h = times[int(preds.argmin())]
    total_kwh = float(preds.sum())
    surplus_hours = int((preds > 1.5).sum())
    balanced_hours = int(((preds >= 0) & (preds <= 1.5)).sum())
    shortage_hours = int((preds < 0).sum())

    if surplus_hours > shortage_hours + balanced_hours:
        fut_status = "Surplus"
    elif shortage_hours > surplus_hours + balanced_hours:
        fut_status = "Shortage"
    else:
        fut_status = "Balanced"

    range_str = f"{times[0]:%d %b %H:%M} → {times[-1]:%d %b %H:%M}"

    if fut_status == "Surplus":
        sentence = "The house will make more than it uses for most of the window."
    elif fut_status == "Balanced":
        sentence = "Production and usage will roughly match across the window."
    else:
        sentence = "The house will pull from the grid for most of the window."

    st.markdown(f"""
    <div class="card card-hero">
        <div class="micro-label">Forecast window</div>
        <div class="hero-sentence">{sentence}</div>
        <div class="hero-facts">{range_str}</div>
        <div class="hero-badge-wrap">
            <span class="status-badge {badge_class(fut_status)}">{fut_status.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        tone = sign_tone(avg)
        if avg > 1.5:
            avg_word = "Most hours in surplus."
        elif avg >= 0:
            avg_word = "Most hours balanced."
        else:
            avg_word = "Most hours in shortage."
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Average net power</div>
            <div class="card-value" style="color:var(--{tone});">{avg:+.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">{avg_word}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        tone = sign_tone(peak)
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Peak surplus hour</div>
            <div class="card-value" style="color:var(--{tone});">{peak:+.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">Most power sent to grid at {peak_h:%H:%M}.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        tone = sign_tone(trough)
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Deepest shortage hour</div>
            <div class="card-value" style="color:var(--{tone});">{trough:+.2f}<span class="card-unit">kW</span></div>
            <div class="card-sub">Most power pulled from grid at {trough_h:%H:%M}.</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        if shortage_hours >= surplus_hours and shortage_hours >= balanced_hours:
            dominant = shortage_hours
            label_word = "shortage"
        elif surplus_hours >= balanced_hours:
            dominant = surplus_hours
            label_word = "surplus"
        else:
            dominant = balanced_hours
            label_word = "balanced"
        st.markdown(f"""
        <div class="card card-kpi">
            <div class="micro-label">Hours in {label_word.capitalize()}</div>
            <div class="card-value">{dominant} of 24<span class="card-unit">hours</span></div>
            <div class="card-sub">Surplus {surplus_hours}h · Balanced {balanced_hours}h · Shortage {shortage_hours}h</div>
        </div>""", unsafe_allow_html=True)

    with st.container(key="chart-predict-net"):
        st.markdown('<div class="sec-head">Net power over the next 24 hours</div>'
                    '<div class="sec-sub">Line above the middle is surplus. Below is shortage.</div>',
                    unsafe_allow_html=True)

        ymax = max(float(preds.max()) * 1.5, 0.5)
        ymin = min(float(preds.min()) * 1.5, -0.5)

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=ymax,
                      fillcolor="rgba(16,185,129,.08)", line_width=0, layer="below")
        fig.add_hrect(y0=ymin, y1=0,
                      fillcolor="rgba(239,68,68,.07)", line_width=0, layer="below")

        fig.add_trace(go.Scatter(
            x=times, y=preds, mode="lines+markers", name="Net power",
            line=dict(color=C_INFO, width=2.5, shape="spline"),   # BLUE line
            marker=dict(size=6, color=C_INFO,
                        line=dict(color="#ffffff", width=1.5)),
            hovertemplate="%{x|%H:%M}<br>Net %{y:.2f} kW<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color=C_TEXT_3, line_width=1)

        fig.add_annotation(
            x=times[2], y=ymax * 0.82,
            text="<b>SURPLUS — sending to grid</b>",
            showarrow=False, xanchor="left",
            font=dict(color=C_POS, size=12, family="Inter, sans-serif"),
        )
        fig.add_annotation(
            x=times[2], y=ymin * 0.82,
            text="<b>SHORTAGE — pulling from grid</b>",
            showarrow=False, xanchor="left",
            font=dict(color=C_NEG, size=12, family="Inter, sans-serif"),
        )
        fig.add_annotation(x=peak_h, y=peak, text=f"Peak {peak:.2f} kW",
                           showarrow=True, arrowhead=0, arrowcolor=C_TEXT_3,
                           font=dict(color=C_TEXT, size=11,
                                     family="Inter, sans-serif"),
                           bgcolor="#ffffff", bordercolor=C_BORDER, borderpad=4)
        fig.add_annotation(x=trough_h, y=trough, text=f"Low {trough:.2f} kW",
                           showarrow=True, arrowhead=0, arrowcolor=C_TEXT_3,
                           font=dict(color=C_TEXT, size=11,
                                     family="Inter, sans-serif"),
                           bgcolor="#ffffff", bordercolor=C_BORDER, borderpad=4)

        fig.update_layout(
            xaxis_title="Hour of day",
            yaxis_title="Net power (kW)",
            showlegend=False,
        )
        st.plotly_chart(style_plot(fig, 380), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "Demand is forecast by the PatchTSMixer model. Solar generation for these "
            "future hours is not modeled — it is approximated from the historical "
            "average generation at that hour of day. Net power above combines the two."
        )

    if fut_status == "Surplus":
        plan = ["Run heavy appliances around midday",
                "Charge the electric car during surplus hours"]
    elif fut_status == "Balanced":
        plan = ["Normal appliance schedule"]
    else:
        plan = ["Delay laundry and dishwasher to the sunniest hour",
                "Reduce heating or air conditioning in the evening"]
    items = "".join(f"<li>{a}</li>" for a in plan)
    st.markdown(f"""
    <div class="card">
        <div class="micro-label">What to do</div>
        <ul class="actions">{items}</ul>
    </div>
    """, unsafe_allow_html=True)