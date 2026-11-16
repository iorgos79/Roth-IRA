import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==============================================================================
# STRATEGY DOCUMENTATION — FROZEN SPEC v1 (OFFICIAL LOCKED SPECIFICATION)
# ==============================================================================
STRATEGY_DOCS = """
**FROZEN SPEC v1: TQQQ / SCHD / SGOV, 70/30 core-satellite (Roth IRA)**
Check daily at 3:45 PM ET using DIVIDEND ADJUSTED (Adj Close in yfinance) prices.

1. **MACRO BREAKER**
   Raw trigger = (VIX > VIX3M) AND (HYG 20-day return < IEI 20-day return).
   Turns ON the day it fires. Turns OFF after 2 consecutive days with the trigger false.

2. **TREND GATE (QQQ vs its 200-day SMA)**
   Exit only if QQQ close < 0.98 x SMA200.
   Re-enter after 2 consecutive closes above SMA200.

3. **VOLATILITY LATCH**
   vol20 = 20-day sample std of QQQ daily returns x sqrt(252), tracked every day.
   ON when vol20 > 26%. OFF only when vol20 < 21%.

4. **SLEEVE** (the strategy part of the account holds ONE asset at a time)
   Macro ON or trend gate off -> 100% SGOV
   Else latch ON             -> 100% SCHD
   Else                      -> 100% TQQQ

5. **CORE:** 30% of the account in SCHD, permanently.

6. **REBALANCE & DRIFT MANAGEMENT (State-Switch Reset + 80% Upper Cap)**
   - **State-Switch Reset:** Whenever the sleeve switches into TQQQ (from SCHD or SGOV) or into SGOV (from TQQQ or SCHD), allocate exactly 70.0% of total account value to the new sleeve asset (TQQQ or SGOV) and set SCHD to 30.0% of total account value.
   - **Upper Guardrail:** While holding TQQQ, if strong momentum pushes TQQQ to >= 80.0% of the total account at the 3:45 PM check, trim TQQQ back to 70.0% and add the proceeds to SCHD.
   - **Downside / Full SCHD:** No lower-band rebalancing needed (the volatility latch and trend gate exit TQQQ before 60% can be reached). While the sleeve holds SCHD (account is 100% SCHD), no rebalance trade is made.
   - **Cash Flows:** Set dividends to Pay-to-Cash (DRIP OFF); route cash distributions and new contributions to whichever asset is underweight relative to the active target state.

7. **FREEZE:** No parameter or rule changes for 12 months.
"""

TICKERS = ['QQQ', 'TQQQ', 'SCHD', 'SGOV', 'HYG', 'IEI', '^VIX', '^VIX3M']

st.set_page_config(
    page_title="ROTH IRA — FROZEN SPEC v1 (70/30 TQQQ / SCHD / SGOV)",
    layout="wide"
)

def get_est_time():
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone('US/Eastern')
    return utc_now.astimezone(est)

@st.cache_data(ttl=300)
def fetch_data_with_retry(tickers):
    """Fetches 5y daily bars, strictly enforcing Adjusted Close (Adj Close) for signals."""
    try:
        data = yf.download(tickers, period="5y", progress=False, auto_adjust=False)

        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if 'Adj Close' in lvl0:
                data = data['Adj Close'].copy()
            elif 'Close' in lvl0:
                data = data['Close'].copy()
            else:
                raise ValueError("Source data missing 'Adj Close' and 'Close'.")
        else:
            if 'Adj Close' in data.columns:
                data = data['Adj Close'].copy()
            elif 'Close' in data.columns:
                data = data['Close'].copy()
            else:
                raise ValueError("No valid price columns found.")

        missing = [t for t in tickers if t not in data.columns]
        if data.empty or missing:
            raise ValueError(f"Missing ticker columns from bulk download: {missing}")

        return data[tickers]

    except Exception as e:
        combined_data = {}
        for t in tickers:
            try:
                df = yf.download(t, period="5y", progress=False, auto_adjust=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        lvl0 = df.columns.get_level_values(0)
                        col_name = 'Adj Close' if 'Adj Close' in lvl0 else 'Close'
                        series = df[col_name].squeeze()
                    else:
                        col_name = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                        series = df[col_name].squeeze()
                    if isinstance(series, pd.Series) and not series.empty:
                        combined_data[t] = series
            except Exception:
                pass

        if not combined_data or len(combined_data) < len(tickers):
            return None
        return pd.DataFrame(combined_data)[tickers]


# ==============================================================================
# UI HEADER & DOCUMENTATION EXPANDER
# ==============================================================================
st.title("ROTH IRA — (70/30 TQQQ / SCHD / SGOV)")
st.caption(f"Server Time (EST): {get_est_time().strftime('%Y-%m-%d %I:%M:%S %p')} | Evaluated daily at 3:45 PM ET via Dividend-Adjusted Prices")

with st.expander("📄 View Official Locked Strategy Documentation (FROZEN SPEC v1)"):
    st.markdown(STRATEGY_DOCS)

# ==============================================================================
# MAIN BACKEND & SIMULATION ENGINE
# ==============================================================================
with st.spinner("Fetching Market Data and Running Simulation Engine..."):
    data = fetch_data_with_retry(TICKERS)

if data is None or data.empty:
    st.error("Connection Failed: No data returned from Yahoo Finance.")
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(0)

data = data.ffill().dropna(subset=TICKERS)

if data.empty:
    st.error("Data is empty after aligning tickers.")
    st.stop()

last_row = data.iloc[-1]
nan_tickers = last_row[last_row.isna()].index.tolist()
if nan_tickers:
    st.error(f"CRITICAL DATA MISSING (NaN): {', '.join(nan_tickers)}\nTry again in 15 minutes.")
    st.stop()

MIN_HISTORY = 205
for t in ['QQQ', 'TQQQ', 'SCHD', 'SGOV', 'HYG', 'IEI']:
    if data[t].notna().sum() < MIN_HISTORY:
        st.error(f"INSUFFICIENT HISTORY: {t} needs {MIN_HISTORY}+ trading days.")
        st.stop()

# ---------------------------------------------------------
# TECHNICAL & MACRO SERIES COMPUTATION (DIVIDEND ADJUSTED)
# ---------------------------------------------------------
sma_200_series = data['QQQ'].rolling(200).mean()
qqq_ret_series = data['QQQ'].pct_change()
vol_20_series = qqq_ret_series.rolling(20).std(ddof=1) * np.sqrt(252.0)

hyg_ret_series = data['HYG'].pct_change(20)
iei_ret_series = data['IEI'].pct_change(20)
credit_stress_series = hyg_ret_series < iei_ret_series
vix_panic_series = data['^VIX'] > data['^VIX3M']
macro_raw_series = vix_panic_series & credit_stress_series

# ---------------------------------------------------------
# STATEFUL SIMULATION ENGINE (FROZEN SPEC v1)
# ---------------------------------------------------------
start_idx = 200
trend_gate_on = bool(data['QQQ'].iloc[start_idx - 1] >= sma_200_series.iloc[start_idx - 1])
above_count = 2 if trend_gate_on else 0

macro_on = False
macro_clear_streak = 2

vol_latch_on = bool(vol_20_series.iloc[start_idx - 1] > 0.26)

hist_sleeve = []
hist_w_tqqq = []
hist_w_schd = []
hist_w_sgov = []
hist_sim_tqqq = []
hist_guardrail_trim = []

sim_w_tqqq = 0.0
sim_w_schd = 1.0
sim_w_sgov = 0.0
prev_sleeve = None

for i in range(start_idx, len(data)):
    p_q = float(data['QQQ'].iloc[i])
    sma_q = float(sma_200_series.iloc[i])
    v_20 = float(vol_20_series.iloc[i])
    m_raw = bool(macro_raw_series.iloc[i])

    # 1. MACRO BREAKER: Turns ON the day it fires; turns OFF after 2 consecutive false days.
    if m_raw:
        macro_on = True
        macro_clear_streak = 0
    else:
        if macro_on:
            macro_clear_streak += 1
            if macro_clear_streak >= 2:
                macro_on = False
        else:
            macro_clear_streak += 1

    # 2. TREND GATE: Exit if QQQ < 0.98 x SMA200; re-enter after 2 consecutive closes > SMA200.
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

    # 3. VOLATILITY LATCH: Tracked every day. ON when vol20 > 26%; OFF only when vol20 < 21%.
    if not vol_latch_on and v_20 > 0.26:
        vol_latch_on = True
    elif vol_latch_on and v_20 < 0.21:
        vol_latch_on = False

    # 4. SLEEVE & 5. CORE (30% SCHD permanently)
    if macro_on or (not trend_gate_on):
        sleeve = "SGOV"
        tw_tqqq, tw_schd, tw_sgov = 0.00, 0.30, 0.70
    elif vol_latch_on:
        sleeve = "SCHD"
        tw_tqqq, tw_schd, tw_sgov = 0.00, 1.00, 0.00
    else:
        sleeve = "TQQQ"
        tw_tqqq, tw_schd, tw_sgov = 0.70, 0.30, 0.00

    # 6. SIMULATED DRIFT & 80% UPPER GUARDRAIL TRACKING
    guardrail_trim_today = False
    if i == start_idx or sleeve != prev_sleeve:
        sim_w_tqqq, sim_w_schd, sim_w_sgov = tw_tqqq, tw_schd, tw_sgov
    else:
        r_tqqq = float(data['TQQQ'].iloc[i] / data['TQQQ'].iloc[i - 1])
        r_schd = float(data['SCHD'].iloc[i] / data['SCHD'].iloc[i - 1])
        r_sgov = float(data['SGOV'].iloc[i] / data['SGOV'].iloc[i - 1])
        tot = (sim_w_tqqq * r_tqqq) + (sim_w_schd * r_schd) + (sim_w_sgov * r_sgov)
        if tot > 0:
            sim_w_tqqq = (sim_w_tqqq * r_tqqq) / tot
            sim_w_schd = (sim_w_schd * r_schd) / tot
            sim_w_sgov = (sim_w_sgov * r_sgov) / tot

        if sleeve == "TQQQ" and sim_w_tqqq >= 0.80:
            guardrail_trim_today = True
            sim_w_tqqq, sim_w_schd, sim_w_sgov = 0.70, 0.30, 0.00

    prev_sleeve = sleeve
    hist_sleeve.append(sleeve)
    hist_w_tqqq.append(tw_tqqq)
    hist_w_schd.append(tw_schd)
    hist_w_sgov.append(tw_sgov)
    hist_sim_tqqq.append(sim_w_tqqq)
    hist_guardrail_trim.append(guardrail_trim_today)

# ---------------------------------------------------------
# EXTRACT CURRENT BAR & STATE TRANSITION CHECK
# ---------------------------------------------------------
cur = data.iloc[-1]
last_bar_date = data.index[-1].strftime('%Y-%m-%d')
q_price = float(cur['QQQ'])
sma_200 = float(sma_200_series.iloc[-1])
exit_band_98 = sma_200 * 0.98
vol_20 = float(vol_20_series.iloc[-1])
hyg_ret = float(hyg_ret_series.iloc[-1])
iei_ret = float(iei_ret_series.iloc[-1])
vix_panic = bool(vix_panic_series.iloc[-1])
credit_stress = bool(credit_stress_series.iloc[-1])
macro_raw_today = bool(macro_raw_series.iloc[-1])

active_sleeve = hist_sleeve[-1]
prev_day_sleeve = hist_sleeve[-2] if len(hist_sleeve) > 1 else active_sleeve
tw_tqqq = hist_w_tqqq[-1]
tw_schd = hist_w_schd[-1]
tw_sgov = hist_w_sgov[-1]
sim_tqqq_weight = hist_sim_tqqq[-1]
guardrail_trim_today = hist_guardrail_trim[-1]

state_changed = (active_sleeve != prev_day_sleeve)

if state_changed:
    if active_sleeve in ("TQQQ", "SGOV"):
        action_note = f"STATE-SWITCH RESET TODAY ({prev_day_sleeve} -> {active_sleeve}: Set 70% {active_sleeve} / 30% SCHD)"
    else:
        action_note = f"STATE SWITCH TODAY ({prev_day_sleeve} -> SCHD: Set 100% SCHD)"
elif guardrail_trim_today:
    action_note = "80% UPPER GUARDRAIL HIT TODAY (Trim TQQQ to 70.0%, Add Proceeds to SCHD!)"
else:
    action_note = "NO STATE SWITCH (Hold Existing Positions; Check 80% Cap if Holding TQQQ)"

# Format Top Signal Banner
if active_sleeve == "SGOV":
    title = "🔴 RED SIGNAL: DEFENSIVE CASH SLEEVE (70% SGOV / 30% SCHD)"
    alloc = "TARGET STATE: 0.0% TQQQ  |  30.0% SCHD (Core)  |  70.0% SGOV (Sleeve)"
    drivers = []
    if macro_on:
        drivers.append("Macro Breaker ON")
    if not trend_gate_on:
        drivers.append("Trend Gate OFF (< 0.98 x SMA200)")
    reason = f"{' & '.join(drivers)} | {action_note}"
    box_color = "#ffcccc"
elif active_sleeve == "SCHD":
    title = "🟡 YELLOW SIGNAL: VOLATILITY LATCH ON (100% SCHD)"
    alloc = "TARGET STATE: 0.0% TQQQ  |  100.0% SCHD (30% Core + 70% Sleeve)  |  0.0% SGOV"
    reason = f"Trend Gate ON + Macro OFF, but Vol Latch ON (vol20 = {vol_20*100:.2f}%) | {action_note}"
    box_color = "#fff2cc"
else:
    if guardrail_trim_today:
        title = "🟠 ORANGE SIGNAL: 80% GUARDRAIL TRIM (RESET TO 70% TQQQ / 30% SCHD)"
        box_color = "#ffe5cc"
    else:
        title = "🟢 GREEN SIGNAL: RISK-ON GROWTH SLEEVE (70% TQQQ / 30% SCHD)"
        box_color = "#ccffcc"
    alloc = "TARGET STATE: 70.0% TQQQ (Sleeve)  |  30.0% SCHD (Core)  |  0.0% SGOV"
    reason = f"Trend Gate ON + Macro OFF + Vol Latch OFF (vol20 = {vol_20*100:.2f}%) | {action_note}"


# ==============================================================================
# STREAMLIT UI LAYOUT & RENDERING
# ==============================================================================

# 1. TOP SIGNAL BANNER
st.markdown(
    f"""
    <div style="background-color: {box_color}; padding: 18px; border-radius: 8px; border: 2px solid #333333; margin-bottom: 20px;">
        <h2 style="color: #111111; margin-top: 0; margin-bottom: 8px; font-size: 22px;">{title}</h2>
        <p style="color: #111111; font-weight: bold; font-size: 15px; margin-bottom: 6px;">{alloc}</p>
        <p style="color: #222222; font-size: 13px; margin-bottom: 0;">{reason}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. THREE-COLUMN METRICS GRID (RULES 1, 2, 3)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Macro Breaker (Rule 1)")
    st.metric("VIX (Spot)", f"{cur['^VIX']:.2f}")
    st.metric("VIX3M (3-Month)", f"{cur['^VIX3M']:.2f}")
    st.markdown(f"**Term Structure:** `{'INVERTED (VIX > VIX3M)' if vix_panic else 'NORMAL'}`")
    
    st.divider()
    st.metric("HYG (20d Ret)", f"{hyg_ret*100:+.2f}%")
    st.metric("IEI (20d Ret)", f"{iei_ret*100:+.2f}%")
    st.markdown(f"**Credit Spread:** `{'STRESS (HYG < IEI)' if credit_stress else 'HEALTHY'}`")
    
    st.divider()
    if macro_on:
        if macro_raw_today:
            macro_txt = "ON (FIRED TODAY — 0/2d Clear)"
        else:
            macro_txt = f"ON (Cooling: {macro_clear_streak}/2d False)"
        st.markdown(f":red[**MACRO BREAKER: {macro_txt}**]")
    else:
        st.markdown(":green[**MACRO BREAKER: OFF (CLEAR)**]")

with col2:
    st.subheader("2. Trend Gate (Rule 2)")
    st.metric("QQQ (Adj Close)", f"${q_price:.2f}")
    st.metric("200 SMA (Re-Entry)", f"${sma_200:.2f}")
    st.metric("0.98 x SMA200 (Exit)", f"${exit_band_98:.2f}")
    
    st.divider()
    if trend_gate_on:
        pct_above_exit = ((q_price / exit_band_98) - 1.0) * 100.0
        st.markdown(f":green[**GATE ON (+{pct_above_exit:.2f}% vs 0.98 Exit)**]")
    else:
        st.markdown(f":red[**GATE OFF (Closes > SMA200: {above_count}/2)**]")

with col3:
    st.subheader("3. Vol Latch & Sleeve (Rules 3-6)")
    st.metric("QQQ 20d Sample Vol", f"{vol_20*100:.2f}%")
    st.markdown("**Thresholds:** `ON > 26%` | `OFF < 21%`")
    
    if vol_latch_on:
        st.markdown(":orange[**LATCH ON (-> 100% SCHD)**]")
    else:
        st.markdown(":green[**LATCH OFF (Normal Vol)**]")
        
    st.divider()
    st.markdown(f"**TQQQ:** `${cur['TQQQ']:.2f}`")
    st.markdown(f"**SCHD:** `${cur['SCHD']:.2f}`")
    st.markdown(f"**SGOV:** `${cur['SGOV']:.2f}`")
    
    if state_changed:
        st.markdown(f":red[**STATE SWITCH: {prev_day_sleeve} -> {active_sleeve}**]")
    elif guardrail_trim_today:
        st.markdown(":orange[**80% GUARDRAIL HIT**]")
    else:
        st.markdown(f":green[**SLEEVE STATUS: HOLD {active_sleeve}**]")


# ==============================================================================
# 4. ROTH IRA DOLLAR, SHARE & 80% GUARDRAIL CALCULATOR
# ==============================================================================
st.markdown("---")
st.subheader("4. Roth IRA Position, 80% Guardrail & Cash-Flow Calculator")

c_in1, c_in2 = st.columns([1, 1])
with c_in1:
    account_balance = st.number_input("Total Roth IRA Balance ($):", min_value=0.0, value=230000.0, step=1000.0, format="%.2f")
with c_in2:
    current_tqqq_dlr = st.number_input("Current TQQQ Value ($) [Optional 80% Check]:", min_value=0.0, value=0.0, step=500.0, format="%.2f")

p_tqqq = float(cur['TQQQ'])
p_schd = float(cur['SCHD'])
p_sgov = float(cur['SGOV'])

dlr_tqqq = account_balance * tw_tqqq
dlr_schd = account_balance * tw_schd
dlr_sgov = account_balance * tw_sgov

sh_tqqq = dlr_tqqq / p_tqqq if p_tqqq > 0 else 0.0
sh_schd = dlr_schd / p_schd if p_schd > 0 else 0.0
sh_sgov = dlr_sgov / p_sgov if p_sgov > 0 else 0.0

# Display Table via Pandas DataFrame
calc_df = pd.DataFrame([
    {
        "Ticker / Role": "TQQQ (70% Growth Sleeve)",
        "Target Weight": f"{tw_tqqq*100:.1f}%",
        "Live Price": f"${p_tqqq:.2f}",
        "Target Dollars ($)": f"${dlr_tqqq:,.2f}",
        "Target Shares (Exact / Whole)": f"{sh_tqqq:,.2f} sh ({int(sh_tqqq):,} whole)"
    },
    {
        "Ticker / Role": "SCHD (30% Core / 100% Vol Latch)",
        "Target Weight": f"{tw_schd*100:.1f}%",
        "Live Price": f"${p_schd:.2f}",
        "Target Dollars ($)": f"${dlr_schd:,.2f}",
        "Target Shares (Exact / Whole)": f"{sh_schd:,.2f} sh ({int(sh_schd):,} whole)"
    },
    {
        "Ticker / Role": "SGOV (70% Macro/Trend-Off Sleeve)",
        "Target Weight": f"{tw_sgov*100:.1f}%",
        "Live Price": f"${p_sgov:.2f}",
        "Target Dollars ($)": f"${dlr_sgov:,.2f}",
        "Target Shares (Exact / Whole)": f"{sh_sgov:,.2f} sh ({int(sh_sgov):,} whole)"
    }
])

st.table(calc_df)

# Guardrail & Cash Flow Messages
if account_balance > 0 and current_tqqq_dlr > 0 and active_sleeve == "TQQQ":
    live_tqqq_pct = (current_tqqq_dlr / account_balance) * 100.0
    if live_tqqq_pct >= 80.0:
        trim_dlr = current_tqqq_dlr - dlr_tqqq
        trim_sh = trim_dlr / p_tqqq if p_tqqq > 0 else 0.0
        st.error(f"⚠️ **80% CAP HIT ({live_tqqq_pct:.1f}%):** Trim `${trim_dlr:,.2f}` (`{trim_sh:,.1f}` shares) TQQQ -> Buy SCHD!")
    elif live_tqqq_pct < 70.0:
        st.success(f"Live TQQQ: `{live_tqqq_pct:.1f}%` (<70%) | **Cash Routing (DRIP OFF):** Buy TQQQ (Underweight)")
    else:
        st.info(f"Live TQQQ: `{live_tqqq_pct:.1f}%` (70%-80% Band) | Hold & Route New Cash/Divs to SCHD")
else:
    st.caption(f"Calculated for `${account_balance:,.2f}` Total Account Value (Active Sleeve: `{active_sleeve}`)")


# ==============================================================================
# RULES LEGEND FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="background-color: #f8f8f8; padding: 12px; border-radius: 5px; border: 1px solid #dcdcdc;">
        <p style="font-weight: bold; color: #111111; margin-bottom: 4px;">LOCKED EXECUTION RULES — ROTH IRA 70/30 TQQQ/SCHD/SGOV (CHECK DAILY @ 3:45 PM ET, DIVIDEND-ADJUSTED PRICES)</p>
        <ul style="font-size: 12px; color: #333333; margin-bottom: 0; padding-left: 20px;">
            <li><b>1. MACRO BREAKER:</b> Raw = (VIX > VIX3M) AND (HYG 20d ret < IEI 20d ret). Turns ON day it fires; turns OFF after 2 consecutive false days.</li>
            <li><b>2. TREND GATE:</b> Exit only if QQQ close < 0.98 x SMA200. Re-enter after 2 consecutive closes above SMA200.</li>
            <li><b>3. VOLATILITY LATCH:</b> vol20 = 20d sample std of QQQ daily returns x sqrt(252), tracked daily. ON when vol20 > 26%; OFF only when vol20 < 21%.</li>
            <li><b>4 & 5. SLEEVE (70%) + CORE (30% SCHD):</b> Macro ON or Trend OFF -> 70% SGOV / 30% SCHD | Else Latch ON -> 100% SCHD | Else -> 70% TQQQ / 30% SCHD.</li>
            <li><b>6 & 7. REBALANCE & FREEZE:</b> Reset 70/30 on switch into TQQQ or SGOV. Trim TQQQ to 70% if >= 80% of account. DRIP OFF (route cash to underweight). Frozen 12 mos.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
