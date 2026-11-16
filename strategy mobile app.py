import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==============================================================================
# STRATEGY DETAILS (FROZEN SPEC v1)
# ==============================================================================
STRATEGY_DOCS = """
**FREQUENCY:** Execute DAILY at 3:45 PM EST using Dividend-Adjusted Close prices.

**OBJECTIVE:** Maximize long-term 3x Nasdaq compounding while mathematically cutting tail risk and drawdown duration via an acute macro circuit breaker, volatility governor, and permanent dividend anchor.

**LOGIC TREE:**

**1. MACRO BREAKER (MACRO_ACTIVE)**
- **Raw Trigger:** (Spot VIX > 3M VIX) AND (HYG 20-day return < IEI 20-day return).
- **Persistence:** Turns ON the day it fires. Turns OFF only after 2 consecutive days with the trigger false.

**2. TREND GATE (TREND_BULL)**
- **Exit Rule:** Exit TQQQ only if QQQ close < 0.98 × 200-day SMA (-2.0% breakdown buffer).
- **Re-Entry Rule:** Re-enter only after 2 consecutive daily closes above the 200-day SMA.

**3. VOLATILITY LATCH (VOL_HIGH)**
- **Metric:** 20-day annualized realized volatility of QQQ daily returns ($vol_{20} = \sigma_{20} \times \sqrt{252}$).
- **Latch Rules:** Flips ON when $vol_{20} > 26.0\%$. Stays ON until $vol_{20} < 21.0\%$.

**4. SLEEVE ALLOCATION (Mutually Exclusive Active Asset)**
- **If Macro ON or Trend Gate OFF:** $\rightarrow$ 100% SGOV (Cash Defense)
- **Else if Volatility Latch ON:** $\rightarrow$ 100% SCHD (Value Equity Defense)
- **Else:** $\rightarrow$ 100% TQQQ (Growth Engine)

**5. PERMANENT CORE**
- **SCHD Core:** 30% of total account value held permanently in SCHD.
- **Active Sleeve:** 70% of total account value allocated to the tactical asset (TQQQ, SCHD, or SGOV).

**6. REBALANCE & DRIFT MANAGEMENT**
- **State-Switch Reset:** Whenever the active sleeve switches assets, reset the active sleeve to exactly 70.0% and SCHD core to 30.0%.
- **Upper Guardrail:** While holding TQQQ, if strong momentum pushes TQQQ to $\ge 80.0\%$ of total account equity at 3:45 PM, trim TQQQ back to 70.0% and sweep proceeds into SCHD.
"""

st.set_page_config(page_title="FROZEN SPEC v1 — Roth IRA Strategy", layout="centered")

TICKERS = ['QQQ', 'TQQQ', 'SCHD', 'SGOV', 'HYG', 'IEI', '^VIX', '^VIX3M']

def get_est_time():
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone('US/Eastern')
    return utc_now.astimezone(est)

@st.cache_data(ttl=300)
def fetch_data_with_retry(tickers):
    try:
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                data = data['Adj Close']
            else:
                raise ValueError("Source data missing 'Adj Close'.")
        else:
            if 'Adj Close' in data:
                data = data['Adj Close']
            elif 'Close' in data:
                data = data['Close']
            else:
                raise ValueError("No valid price data found.")

        if data.empty or data.shape[1] < len(tickers):
            raise ValueError("Incomplete data returned")
        return data
    except Exception as e:
        combined_data = {}
        for t in tickers:
            try:
                df = yf.download(t, period="2y", progress=False, auto_adjust=True)
                if not df.empty:
                    combined_data[t] = df['Close']
            except Exception:
                pass
        if not combined_data:
            return None
        return pd.DataFrame(combined_data)

# --- MAIN UI ---
st.title("🛡️ FROZEN SPEC v1: 70/30 ROTH IRA STRATEGY")
st.caption(f"Server Time: {get_est_time().strftime('%Y-%m-%d %I:%M %p EST')}")

with st.expander("📄 Strategy Documentation (Click to Expand)"):
    st.markdown(STRATEGY_DOCS)

if st.button("RUN 3:45 PM ANALYSIS", type="primary", use_container_width=True):
    status_placeholder = st.empty()
    status_placeholder.info("Fetching Dividend-Adjusted Market Data...")

    data = fetch_data_with_retry(TICKERS)
    
    if data is None or data.empty:
        status_placeholder.empty()
        st.error("Connection Failed: No data returned from API.")
        st.stop()
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)

    data = data.ffill()

    last_row = data.iloc[-1]
    nan_tickers = last_row[last_row.isna()].index.tolist()
    if nan_tickers:
        st.error(f"CRITICAL DATA MISSING (NaN): {', '.join(nan_tickers)}\n\nTry again in 15 minutes.")
        st.stop()

    MIN_HISTORY = 220
    if len(data) < MIN_HISTORY:
        st.error(f"⚠️ INSUFFICIENT DATA: Requires 220+ trading days for 200 SMA calculation.")
        st.stop()

    # --- CALCULATE TECHNICAL & MACRO SERIES ---
    data['QQQ_SMA200'] = data['QQQ'].rolling(200).mean()
    data['QQQ_RET'] = data['QQQ'].pct_change()
    data['VOL20'] = data['QQQ_RET'].rolling(20).std() * np.sqrt(252)
    
    data['HYG_RET20'] = data['HYG'] / data['HYG'].shift(20) - 1.0
    data['IEI_RET20'] = data['IEI'] / data['IEI'].shift(20) - 1.0
    
    # 1. Macro Breaker State Machine (Turns ON immediately, turns OFF after 2 consecutive clean days)
    macro_raw = (data['^VIX'] > data['^VIX3M']) & (data['HYG_RET20'] < data['IEI_RET20'])
    macro_active = False
    false_streak = 0
    macro_active_list = []
    for raw in macro_raw:
        if raw:
            macro_active = True
            false_streak = 0
        else:
            if macro_active:
                false_streak += 1
                if false_streak >= 2:
                    macro_active = False
        macro_active_list.append(macro_active)
    data['MACRO_ACTIVE'] = macro_active_list

    # 2. Trend Gate State Machine (Exit if QQQ < 0.98 * SMA200; Re-enter after 2 consecutive closes > SMA200)
    trend_bull = True
    above_count = 2
    trend_bull_list = []
    for p, sma in zip(data['QQQ'], data['QQQ_SMA200']):
        if p > sma:
            above_count += 1
        else:
            above_count = 0
        
        if not trend_bull:
            if above_count >= 2:
                trend_bull = True
        else:
            if p < (0.98 * sma):
                trend_bull = False
        trend_bull_list.append(trend_bull)
    data['TREND_BULL'] = trend_bull_list

    # 3. Volatility Latch State Machine (ON when vol20 > 26%, OFF when vol20 < 21%)
    vol_high = False
    vol_high_list = []
    for v in data['VOL20']:
        if pd.isna(v):
            vol_high_list.append(False)
            continue
        if not vol_high:
            if v > 0.26:
                vol_high = True
        else:
            if v < 0.21:
                vol_high = False
        vol_high_list.append(vol_high)
    data['VOL_HIGH'] = vol_high_list

    # Get latest evaluated state
    cur = data.iloc[-1]
    is_macro_active = cur['MACRO_ACTIVE']
    is_trend_bull = cur['TREND_BULL']
    is_vol_high = cur['VOL_HIGH']
    cur_vol20 = cur['VOL20']
    q_price = cur['QQQ']
    sma_200 = cur['QQQ_SMA200']

    status_placeholder.empty()

    # --- DETERMINE ACTIVE SLEEVE ---
    if is_macro_active or not is_trend_bull:
        sleeve_asset = "SGOV"
        sleeve_reason = "Macro Circuit Breaker ACTIVE or Trend Gate BEAR (-2% breach)."
        status_color = "error"
    elif is_vol_high:
        sleeve_asset = "SCHD"
        sleeve_reason = "Volatility Latch ON (Realized Vol > 26%). Shifting sleeve to SCHD."
        status_color = "warning"
    else:
        sleeve_asset = "TQQQ"
        sleeve_reason = "Normal Bull Regime. Growth compounding active."
        status_color = "success"

    # --- RENDER PRIMARY ACTION SIGNAL ---
    if status_color == "error":
        st.error(f"### 🛡️ DEFENSIVE ROTATION: 100% {sleeve_asset}\n\n**Active Sleeve:** 70% {sleeve_asset} | **Core:** 30% SCHD\n\n*{sleeve_reason}*")
    elif status_color == "warning":
        st.warning(f"### ⚠️ HIGH VOLATILITY PAUSE: 100% SCHD\n\n**Active Sleeve:** 70% SCHD | **Core:** 30% SCHD (100% Total SCHD)\n\n*{sleeve_reason}*")
    else:
        st.success(f"### 🟢 RISK-ON: 100% TQQQ\n\n**Active Sleeve:** 70% TQQQ | **Core:** 30% SCHD\n\n*{sleeve_reason}*", icon="🚀")

    # --- DATA GRID METRICS ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Macro Breaker")
        st.metric("VIX Spot", f"{cur['^VIX']:.2f}")
        st.metric("VIX 3M", f"{cur['^VIX3M']:.2f}")
        vix_inv = cur['^VIX'] > cur['^VIX3M']
        st.markdown(f"**Inversion:** `{'YES' if vix_inv else 'NO'}`")
        
        st.divider()
        st.metric("HYG 20d Ret", f"{cur['HYG_RET20']:.2%}")
        st.metric("IEI 20d Ret", f"{cur['IEI_RET20']:.2%}")
        credit_lag = cur['HYG_RET20'] < cur['IEI_RET20']
        st.markdown(f"**Credit Stress:** `{'YES' if credit_lag else 'NO'}`")
        
        st.divider()
        if is_macro_active:
            st.markdown(":red[**MACRO: ACTIVE (RED)**]")
        else:
            st.markdown(":green[**MACRO: CLEAR**]")

    with col2:
        st.subheader("2. Trend Gate")
        st.metric("QQQ Price", f"${q_price:.2f}")
        st.metric("200-day SMA", f"${sma_200:.2f}")
        buffer_price = sma_200 * 0.98
        st.metric("Exit Buffer (-2%)", f"${buffer_price:.2f}")
        
        if is_trend_bull:
            st.markdown(":green[**TREND: BULL (GREEN)**]")
        else:
            st.markdown(":red[**TREND: BEAR (-2% Breached)**]")

    with col3:
        st.subheader("3. Volatility Latch")
        st.metric("QQQ 20d Realized Vol", f"{cur_vol20:.1%}")
        st.markdown("**Thresholds:** `ON > 26%` | `OFF < 21%`")
        
        if is_vol_high:
            st.markdown(":orange[**VOL LATCH: ACTIVE (SCHD)**]")
        else:
            st.markdown(":green[**VOL LATCH: NORMAL**]")

st.markdown("---")
st.subheader("Execution & Maintenance Checklist")
st.info("EXECUTION WINDOW: Daily @ 3:45 PM EST (Market-on-Close / Limit Orders)")
st.markdown("""
- **State-Switch Reset:** When the active sleeve asset changes (e.g., TQQQ $\leftrightarrow$ SGOV), rebalance portfolio to exactly **70% Active Sleeve / 30% SCHD Core**.
- **Upper Guardrail:** If TQQQ momentum pushes the tactical sleeve to $\ge 80\%$ of total portfolio value at check time, trim back to 70% and sweep excess into SCHD.
- **Cash Flow Routing:** Set dividends to Pay-to-Cash (DRIP OFF); route cash distributions and new contributions to whichever asset is underweight relative to target state.
""")
