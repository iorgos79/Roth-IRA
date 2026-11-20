import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    import pytz
    ET_TZ = pytz.timezone("America/New_York")

# ==============================================================================
# STRATEGY DOCUMENTATION — ROTH IRA (Strategy Ss + 1.2x SCHD)
# ==============================================================================
STRATEGY_DOCS = """
**Strategy Ss + 1.2x SCHD (Roth IRA — Single-Asset 100% States)**
Check daily at 3:45 PM ET using DIVIDEND-ADJUSTED (`Adj Close` in yfinance) prices.

- **GREEN STATE:** 100% `TQQQ` (or `UPRO` if running S&P 500).
- **ORANGE STATE:** 100% `SCHD` (High-Volatility Regime).
- **EXTENDED / OVERBOUGHT STATE:** 100% `SCHD` (Temporary step-down when `QQQ > 1.20 * SMA200`; stays in `SCHD` until `QQQ < 1.17 * SMA200`).
- **RED STATE:** 100% `SGOV` (or `BIL` / Treasury Money Market).

**Signal Parameters (Evaluated at 3:45 PM ET Daily):**
1. **MACRO BREAKER**
   Raw trigger = (`^VIX > ^VIX3M`) AND (20-day return of `HYG < IEI`).
   Exit to **RED (`100% SGOV`)** immediately. Latch for **2 consecutive clear days** before releasing.

2. **TREND GATE (`QQQ` vs. 200-day dividend-adjusted SMA)**
   Exit to **RED (`100% SGOV`)** if `QQQ < 0.98 * SMA200`.
   Re-enter only after **2 consecutive closes above `SMA200`**.

3. **VOLATILITY LATCH**
   `vol20` = 20-day annualized realized volatility of `QQQ` (sample std × √252), tracked daily.
   Shift to **ORANGE (`100% SCHD`)** if `vol20 > 26.0%`. Stay in `SCHD` until `vol20 < 21.0%`.

4. **OVEREXTENSION STEP-DOWN (`1.20x / 1.17x`)**
   Evaluated on `QQQ` close relative to its 200-day dividend-adjusted SMA.
   Step to **EXTENDED (`100% SCHD`)** if `QQQ > 1.20 * SMA200`. Stay in `SCHD` until `QQQ < 1.17 * SMA200` before returning to GREEN.

**Decision Hierarchy:**
1. If Macro Breaker is ON **OR** Trend Gate is OFF → Hold **100% `SGOV` (RED)**.
2. Otherwise, if Volatility Latch is ON (`>26% / <21%`) → Hold **100% `SCHD` (ORANGE)**.
3. Otherwise, if Overextension Step-Down is ON (`>1.20x / <1.17x`) → Hold **100% `SCHD` (EXTENDED)**.
4. Otherwise → Hold **100% `TQQQ` (GREEN)**.
"""

TICKERS = ['QQQ', 'HYG', 'IEI', '^VIX', '^VIX3M']

st.set_page_config(
    page_title="Roth IRA",
    layout="centered"
)

# Mobile-friendly compact CSS
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; max-width: 820px; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True
)


def get_et_time():
    return datetime.now(ET_TZ)


def extract_adj_series(df, ticker=None):
    """Safely extracts dividend-adjusted Close (Adj Close) with fallback to Close."""
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        col = 'Adj Close' if 'Adj Close' in lvl0 else ('Close' if 'Close' in lvl0 else None)
        if col is None:
            return None
        sub = df[col]
        if ticker is not None and isinstance(sub, pd.DataFrame) and ticker in sub.columns:
            return pd.to_numeric(sub[ticker], errors='coerce')
        return pd.to_numeric(sub.squeeze(), errors='coerce')
    col = 'Adj Close' if 'Adj Close' in df.columns else ('Close' if 'Close' in df.columns else None)
    return pd.to_numeric(df[col], errors='coerce') if col else None


@st.cache_data(ttl=60)
def fetch_data_with_retry(tickers):
    """Fetches 5y daily dividend-adjusted prices (Adj Close) for signal calculation."""
    try:
        raw = yf.download(tickers, period="5y", progress=False, auto_adjust=False)
        if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            raise ValueError("Empty or unexpected bulk download response.")

        lvl0 = raw.columns.get_level_values(0)
        if 'Adj Close' in lvl0:
            data = raw['Adj Close'].copy()
            if 'Close' in lvl0:
                data = data.combine_first(raw['Close'])
        elif 'Close' in lvl0:
            data = raw['Close'].copy()
        else:
            raise ValueError("Source data missing 'Adj Close' and 'Close'.")

        missing = [t for t in tickers if t not in data.columns]
        if data.empty or missing:
            raise ValueError(f"Missing ticker columns: {missing}")

        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data[tickers].apply(pd.to_numeric, errors='coerce')

    except Exception:
        combined = {}
        for t in tickers:
            try:
                df = yf.download(t, period="5y", progress=False, auto_adjust=False)
                s = extract_adj_series(df, ticker=t)
                if isinstance(s, pd.Series) and not s.empty:
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    combined[t] = s
            except Exception:
                pass

        if len(combined) < len(tickers):
            return None
        return pd.DataFrame(combined)[tickers]


def compute_strategy_state(data):
    """Replays 5-year history sequentially so every hysteresis latch matches exact live state."""
    sma_200_series = data['QQQ'].rolling(200).mean()
    qqq_ret_series = data['QQQ'].pct_change()
    vol_20_series = qqq_ret_series.rolling(20).std(ddof=1) * np.sqrt(252.0)
    ext_ratio_series = data['QQQ'] / sma_200_series

    hyg_ret_series = data['HYG'].pct_change(20)
    iei_ret_series = data['IEI'].pct_change(20)
    credit_stress_series = hyg_ret_series < iei_ret_series
    vix_panic_series = data['^VIX'] > data['^VIX3M']
    macro_raw_series = vix_panic_series & credit_stress_series

    start_idx = 200
    trend_gate_on = bool(data['QQQ'].iloc[start_idx - 1] >= sma_200_series.iloc[start_idx - 1])
    above_count = 2 if trend_gate_on else 0

    macro_on = False
    macro_clear_streak = 2

    vol_latch_on = bool(vol_20_series.iloc[start_idx - 1] > 0.26)
    ext_latch_on = bool(ext_ratio_series.iloc[start_idx - 1] > 1.20)

    hist_state = []
    hist_asset = []

    for i in range(start_idx, len(data)):
        p_q = float(data['QQQ'].iloc[i])
        sma_q = float(sma_200_series.iloc[i])
        v_20 = float(vol_20_series.iloc[i])
        e_rat = float(ext_ratio_series.iloc[i])
        m_raw = bool(macro_raw_series.iloc[i])

        # 1. Macro Breaker: ON immediately; releases after 2 consecutive clear days.
        if m_raw:
            macro_on = True
            macro_clear_streak = 0
        else:
            macro_clear_streak += 1
            if macro_on and macro_clear_streak >= 2:
                macro_on = False

        # 2. Trend Gate: Exit if QQQ < 0.98 * SMA200; re-enter after 2 consecutive closes > SMA200.
        if p_q > sma_q:
            above_count += 1
        else:
            above_count = 0

        if not trend_gate_on:
            if above_count >= 2:
                trend_gate_on = True
        else:
            if p_q < (sma_q * 0.98):
                trend_gate_on = False

        # 3. Volatility Latch: ON when vol20 > 26.0%; OFF only when vol20 < 21.0%.
        if not vol_latch_on and v_20 > 0.26:
            vol_latch_on = True
        elif vol_latch_on and v_20 < 0.21:
            vol_latch_on = False

        # 4. Overextension Step-Down: ON when QQQ > 1.20 * SMA200; OFF only when < 1.17 * SMA200.
        if not ext_latch_on and e_rat > 1.20:
            ext_latch_on = True
        elif ext_latch_on and e_rat < 1.17:
            ext_latch_on = False

        # Decision Hierarchy (100% Single-Asset Target)
        if macro_on or (not trend_gate_on):
            state, asset = "RED", "SGOV"
        elif vol_latch_on:
            state, asset = "ORANGE", "SCHD"
        elif ext_latch_on:
            state, asset = "EXTENDED", "SCHD"
        else:
            state, asset = "GREEN", "TQQQ"

        hist_state.append(state)
        hist_asset.append(asset)

    return {
        'sma_200': float(sma_200_series.iloc[-1]),
        'vol_20': float(vol_20_series.iloc[-1]),
        'ext_ratio': float(ext_ratio_series.iloc[-1]),
        'hyg_ret': float(hyg_ret_series.iloc[-1]),
        'iei_ret': float(iei_ret_series.iloc[-1]),
        'vix_panic': bool(vix_panic_series.iloc[-1]),
        'credit_stress': bool(credit_stress_series.iloc[-1]),
        'macro_raw_today': bool(macro_raw_series.iloc[-1]),
        'macro_on': macro_on,
        'macro_clear_streak': macro_clear_streak,
        'trend_gate_on': trend_gate_on,
        'above_count': above_count,
        'vol_latch_on': vol_latch_on,
        'ext_latch_on': ext_latch_on,
        'active_state': hist_state[-1],
        'prev_day_state': hist_state[-2] if len(hist_state) > 1 else hist_state[-1],
        'active_asset': hist_asset[-1],
        'prev_day_asset': hist_asset[-2] if len(hist_asset) > 1 else hist_asset[-1],
    }


# ==============================================================================
# UI HEADER & DOCUMENTATION EXPANDER
# ==============================================================================
st.title("Roth IRA")
st.caption(f"Server Time (ET): {get_et_time().strftime('%Y-%m-%d %I:%M:%S %p')} | Daily @ 3:45 PM ET (Adj Close)")

with st.expander("📄 View Official Locked Strategy Documentation"):
    st.markdown(STRATEGY_DOCS)

# ==============================================================================
# MAIN BACKEND & SIMULATION ENGINE
# ==============================================================================
with st.spinner("Fetching Market Data and Running Strategy Engine..."):
    data = fetch_data_with_retry(TICKERS)

if data is None or data.empty:
    st.error("Connection Failed: No data returned from Yahoo Finance.")
    st.stop()

data = data.sort_index().ffill().dropna(subset=TICKERS)

if data.empty or len(data) < 205:
    st.error("Insufficient history: Need 205+ aligned trading days.")
    st.stop()

last_row = data.iloc[-1]
nan_tickers = last_row[last_row.isna()].index.tolist()
if nan_tickers:
    st.error(f"CRITICAL DATA MISSING (NaN): {', '.join(nan_tickers)}\nTry again in 15 minutes.")
    st.stop()

s = compute_strategy_state(data)

cur = data.iloc[-1]
last_bar_date = data.index[-1].strftime('%Y-%m-%d')
q_price = float(cur['QQQ'])
sma_200 = s['sma_200']
exit_band_98 = sma_200 * 0.98
ext_band_120 = sma_200 * 1.20
ext_band_117 = sma_200 * 1.17

active_state = s['active_state']
prev_day_state = s['prev_day_state']
active_asset = s['active_asset']
prev_day_asset = s['prev_day_asset']

asset_changed = (active_asset != prev_day_asset)
state_changed = (active_state != prev_day_state)

if asset_changed:
    action_note = f"TRADE TODAY: SWITCH 100% {prev_day_asset} -> 100% {active_asset}"
elif state_changed:
    action_note = f"STATE SHIFT ({prev_day_state} -> {active_state}) — STAY 100% {active_asset} (NO TRADE)"
else:
    action_note = f"NO TRADE REQUIRED — HOLD 100% {active_asset}"

# Format Top Signal Banner
if active_state == "RED":
    title = "🔴 RED SIGNAL: DEFENSIVE CASH (100% SGOV)"
    alloc = "BUY / HOLD: 100% SGOV (or BIL / Treasury Money Market)"
    drivers = []
    if s['macro_on']:
        drivers.append("Macro Breaker ON")
    if not s['trend_gate_on']:
        drivers.append("Trend Gate OFF (< 0.98 × SMA200)")
    reason = f"{' & '.join(drivers)} | {action_note}"
    box_color = "#ffcccc"
elif active_state == "ORANGE":
    title = "🟠 ORANGE SIGNAL: HIGH-VOLATILITY REGIME (100% SCHD)"
    alloc = "BUY / HOLD: 100% SCHD (Volatility Defense)"
    reason = f"Vol Latch ON (vol20 = {s['vol_20']*100:.2f}%, exit < 21.0%) | {action_note}"
    box_color = "#ffe5cc"
elif active_state == "EXTENDED":
    title = "🟡 EXTENDED SIGNAL: OVERBOUGHT STEP-DOWN (100% SCHD)"
    alloc = "BUY / HOLD: 100% SCHD (Overbought Step-Down)"
    reason = f"QQQ / SMA200 = {s['ext_ratio']:.3f}x (exit < 1.170x) | {action_note}"
    box_color = "#fff2cc"
else:
    title = "🟢 GREEN SIGNAL: RISK-ON GROWTH (100% TQQQ)"
    alloc = "BUY / HOLD: 100% TQQQ"
    reason = f"All Defenses Clear (vol20 = {s['vol_20']*100:.2f}%, ratio = {s['ext_ratio']:.3f}x) | {action_note}"
    box_color = "#ccffcc"

# ==============================================================================
# STREAMLIT UI LAYOUT & RENDERING (MOBILE-OPTIMIZED)
# ==============================================================================

# 1. TOP SIGNAL BANNER
st.markdown(
    f"""
    <div style="background-color: {box_color}; padding: 16px; border-radius: 8px; border: 2px solid #333333; margin-bottom: 18px;">
        <h2 style="color: #111111; margin-top: 0; margin-bottom: 6px; font-size: 20px; line-height: 1.25;">{title}</h2>
        <p style="color: #111111; font-weight: bold; font-size: 15px; margin-bottom: 6px;">{alloc}</p>
        <p style="color: #222222; font-size: 13px; margin-bottom: 0;">{reason}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. THREE-COLUMN RULES GRID (STACKS CLEANLY ON MOBILE)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Macro Breaker")
    c1a, c1b = st.columns(2)
    c1a.metric("VIX (Spot)", f"{cur['^VIX']:.2f}")
    c1b.metric("VIX3M (3M)", f"{cur['^VIX3M']:.2f}")
    st.markdown(f"**Term Structure:** `{'INVERTED (PANIC)' if s['vix_panic'] else 'NORMAL'}`")

    st.divider()
    c1c, c1d = st.columns(2)
    c1c.metric("HYG (20d)", f"{s['hyg_ret']*100:+.2f}%")
    c1d.metric("IEI (20d)", f"{s['iei_ret']*100:+.2f}%")
    st.markdown(f"**Credit Spread:** `{'STRESS (HYG < IEI)' if s['credit_stress'] else 'HEALTHY'}`")

    st.divider()
    if s['macro_on']:
        if s['macro_raw_today']:
            macro_txt = "ON (FIRED — 0/2d Clear)"
        else:
            macro_txt = f"ON (Cooling: {s['macro_clear_streak']}/2d Clear)"
        st.markdown(f":red[**MACRO BREAKER: {macro_txt}**]")
    else:
        st.markdown(":green[**MACRO BREAKER: OFF (PASS)**]")
    st.divider()

with col2:
    st.subheader("2. Trend Gate")
    st.metric("QQQ (Adj Close)", f"${q_price:.2f}")
    c2a, c2b = st.columns(2)
    c2a.metric("200 SMA", f"${sma_200:.2f}")
    c2b.metric("0.98× Exit", f"${exit_band_98:.2f}")

    st.divider()
    if s['trend_gate_on']:
        pct_above_exit = ((q_price / exit_band_98) - 1.0) * 100.0
        st.markdown(f":green[**GATE ON (+{pct_above_exit:.2f}% vs Exit)**]")
    else:
        st.markdown(f":red[**GATE OFF (Closes > SMA200: {s['above_count']}/2)**]")
    st.divider()

with col3:
    st.subheader("3. Vol & Extended")
    st.metric("QQQ 20d Vol", f"{s['vol_20']*100:.2f}%")
    st.markdown("**Thresholds:** `ON > 26%` | `OFF < 21%`")
    if s['vol_latch_on']:
        st.markdown(":orange[**VOL LATCH ON (→ 100% SCHD)**]")
    else:
        st.markdown(":green[**VOL LATCH OFF (Normal Vol)**]")

    st.divider()
    st.metric("QQQ / SMA200", f"{s['ext_ratio']:.3f}x")
    st.caption(f"1.20x: ${ext_band_120:.2f} | 1.17x: ${ext_band_117:.2f}")
    if s['ext_latch_on']:
        st.markdown(":orange[**EXTENDED ON (→ 100% SCHD)**]")
    else:
        st.markdown(":green[**EXTENDED OFF (Not Overbought)**]")

    st.divider()
    if asset_changed:
        st.markdown(f":red[**ACTION: SWITCH {prev_day_asset} → {active_asset}**]")
    elif state_changed:
        st.markdown(f":blue[**SHIFT ({prev_day_state} → {active_state}) | HOLD {active_asset}**]")
    else:
        st.markdown(f":green[**ACTION: HOLD 100% {active_asset}**]")

# ==============================================================================
# RULES LEGEND FOOTER
# ==============================================================================
st.caption(f"Latest Market Bar Evaluated: {last_bar_date} (Dividend-Adjusted Close)")
st.markdown(
    """
    <div style="background-color: #f8f8f8; padding: 12px; border-radius: 6px; border: 1px solid #dcdcdc; margin-top: 8px;">
        <p style="font-weight: bold; color: #111111; margin-bottom: 4px; font-size: 13px;">LOCKED EXECUTION RULES — STRATEGY Ss + 1.2x SCHD (DAILY @ 3:45 PM ET)</p>
        <ul style="font-size: 12px; color: #333333; margin-bottom: 0; padding-left: 18px;">
            <li><b>1. MACRO BREAKER:</b> (^VIX &gt; ^VIX3M) AND (HYG 20d ret &lt; IEI 20d ret) → RED (100% SGOV) immediately. Releases after 2 consecutive clear days.</li>
            <li><b>2. TREND GATE:</b> Exit to RED (100% SGOV) if QQQ &lt; 0.98 × SMA200. Re-enter only after 2 consecutive closes above SMA200.</li>
            <li><b>3. VOLATILITY LATCH:</b> QQQ 20d annualized realized vol &gt; 26.0% → ORANGE (100% SCHD). Stays in SCHD until vol20 &lt; 21.0%.</li>
            <li><b>4. OVEREXTENSION STEP-DOWN:</b> QQQ &gt; 1.20 × SMA200 → EXTENDED (100% SCHD). Stays in SCHD until QQQ &lt; 1.17 × SMA200.</li>
            <li><b>HIERARCHY:</b> RED (100% SGOV) → ORANGE (100% SCHD) → EXTENDED (100% SCHD) → GREEN (100% TQQQ).</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
