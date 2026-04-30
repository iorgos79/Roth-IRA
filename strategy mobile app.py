import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

# ==============================================================================
# STRATEGY DETAILS
# ==============================================================================
STRATEGY_DOCS = """
**FREQUENCY:** Execute DAILY at 3:45 PM EST.

**OBJECTIVE:** Maximize absolute CAGR utilizing 3X leverage while enforcing a 
hard mathematical floor against systemic crashes.

**LOGIC TREE:**

**STEP 1. SAFETY CHECK (MACRO FILTER)**
* **Triggers:** a) Spot VIX > 3M VIX (^VIX > ^VIX3M)
  b) HYG underperforms IEI over 20 days.
* **RULE:** If BOTH a AND b are True -> STATUS = RED (RISK OFF).

**STEP 2. TREND CHECK (BINARY PRICE FILTER)**
* **GREEN (RISK ON):** QQQ > 200 SMA -OR- (QQQ > 50 SMA AND MACD > Signal Line).
* **RED (RISK OFF):** QQQ < 200 SMA AND (QQQ < 50 SMA OR MACD < Signal Line).

**STEP 3. ALLOCATION ENGINE**
* **IF SIGNAL IS RED (OR MACRO IS RED):**
  * Scenario A (Strong Dollar): UUP > 63 SMA -> Buy: 40% KMLM / 40% BTAL / 20% USDU
  * Scenario B (Stagflation): UUP < 63 SMA AND GLD > 200 SMA -> Buy: 40% KMLM / 40% BTAL / 20% GLDM
  * Scenario C (Deflation): Both below SMAs -> Sell everything, 100% CASH (Schwab Sweep)
* **IF SIGNAL IS GREEN:**
  * Monday-Thursday: HOLD current position. Do not buy TQQQ.
  * Friday: Buy/Hold 100% TQQQ.
"""

st.set_page_config(page_title="Roth IRA Strategy", layout="centered")

TICKERS = ['QQQ', 'HYG', 'IEI', 'UUP', 'GLD', '^VIX', '^VIX3M']

def get_est_time():
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone('US/Eastern')
    return utc_now.astimezone(est)

@st.cache_data(ttl=300) # Cache for 5 mins to prevent API spam
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
            else:
                if 'Close' in data:
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
st.title("ROTH IRA STRATEGY")
st.caption(f"Server Time: {get_est_time().strftime('%Y-%m-%d %I:%M %p EST')}")

with st.expander("📄 Strategy Documentation (Click to Expand)"):
    st.markdown(STRATEGY_DOCS)

if st.button("RUN 3:45 PM ANALYSIS", type="primary", use_container_width=True):
    status_placeholder = st.empty()
    status_placeholder.info("Fetching Market Data...")

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

    MIN_HISTORY = 205
    for t in ['QQQ', 'GLD']:
        if data[t].notna().sum() < MIN_HISTORY:
            st.error(f"⚠️ INSUFFICIENT DATA: {t} requires 205+ days.")
            st.stop()

    cur = data.iloc[-1]
    prev_20 = data.iloc[-21]

    est_now = get_est_time()
    today_weekday = est_now.weekday()

    # STEP 1: MACRO FILTER
    vix_panic = cur['^VIX'] > cur['^VIX3M']
    hyg_ret = (cur['HYG'] - prev_20['HYG']) / prev_20['HYG']
    iei_ret = (cur['IEI'] - prev_20['IEI']) / prev_20['IEI']
    credit_stress = hyg_ret < iei_ret
    
    macro_red = vix_panic and credit_stress

    # STEP 2: TREND FILTER (QQQ)
    q_price = cur['QQQ']
    sma_200 = data['QQQ'].rolling(200).mean().iloc[-1]
    sma_50 = data['QQQ'].rolling(50).mean().iloc[-1]
    
    exp12 = data['QQQ'].ewm(span=12, adjust=False).mean()
    exp26 = data['QQQ'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    sig = macd.ewm(span=9, adjust=False).mean()
    macd_bullish = macd.iloc[-1] > sig.iloc[-1]

    green_cond = (q_price > sma_200) or ((q_price > sma_50) and macd_bullish)
    red_cond = (q_price < sma_200) and ((q_price < sma_50) or not macd_bullish)

    if macro_red:
        trend_status = "RED"
    elif red_cond and not green_cond:
        trend_status = "RED"
    else:
        trend_status = "GREEN"

    # STEP 3: ALLOCATION ENGINE
    sma_uup_63 = data['UUP'].rolling(63).mean().iloc[-1]
    sma_gld_200 = data['GLD'].rolling(200).mean().iloc[-1]
    
    uup_up = cur['UUP'] > sma_uup_63
    gld_up = cur['GLD'] > sma_gld_200

    status_placeholder.empty()

    if trend_status == "RED":
        if uup_up:
            st.error("### 🔴 RED SIGNAL: HEDGE A\n\n**BUY: 40% KMLM / 40% BTAL / 20% USDU**\n\n*Scenario A (Cash Crunch / Strong Dollar)*")
        elif not uup_up and gld_up:
            st.warning("### 🔴 RED SIGNAL: HEDGE B\n\n**BUY: 40% KMLM / 40% BTAL / 20% GLDM**\n\n*Scenario B (Stagflation / Weak Dollar)*")
        else:
            st.error("### 🔴 RED SIGNAL: CASH\n\n**SELL EVERYTHING -> 100% CASH (Schwab Sweep)**\n\n*Scenario C (Deflationary Cash)*")
    else:
        if today_weekday == 4:
            st.success("### 🟢 GREEN SIGNAL: RISK ON\n\n**BUY/HOLD 100% TQQQ**\n\n*Trend is GREEN. Today is Friday (Execution Day).*")
        else:
            st.success("### 🟢 GREEN SIGNAL: HOLD (Waiting for Friday)\n\n**HOLD CURRENT POSITION**\n\n*Trend is GREEN, but today is not Friday. Do not buy TQQQ.*", icon="⏳")

    # --- DATA GRID ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Macro (Step 1)")
        st.metric("VIX (Spot)", f"{cur['^VIX']:.2f}")
        st.metric("VIX (3M)", f"{cur['^VIX3M']:.2f}")
        if vix_panic:
            st.markdown(":red[**VIX: PANIC**]")
        else:
            st.markdown(":green[**VIX: NORMAL**]")
        
        st.divider()
        st.metric("HYG (Risk)", f"{hyg_ret:.2%}")
        st.metric("IEI (Safe)", f"{iei_ret:.2%}")
        if credit_stress:
            st.markdown(":red[**CREDIT: STRESS**]")
        else:
            st.markdown(":green[**CREDIT: HEALTHY**]")

        st.divider()
        if macro_red:
            st.markdown(":red[**MACRO: FAIL (RED)**]")
        else:
            st.markdown(":green[**MACRO: PASS (GREEN)**]")

    with col2:
        st.subheader("2. Trend (Step 2)")
        st.metric("QQQ Price", f"${q_price:.2f}")
        st.metric("200 SMA", f"${sma_200:.2f}")
        st.metric("50 SMA", f"${sma_50:.2f}")
        if macd_bullish:
            st.markdown(":green[**MACD: BULLISH**]")
        else:
            st.markdown(":red[**MACD: BEARISH**]")

    with col3:
        st.subheader("3. Defense Select")
        uup_stat = "UP (> 63 SMA)" if uup_up else "DOWN"
        uup_col = "green" if uup_up else "red"
        st.metric("Dollar (UUP)", f"${cur['UUP']:.2f}")
        st.markdown(f":{uup_col}[**TREND: {uup_stat}**]")
        
        st.divider()
        
        gld_stat = "UP (> 200 SMA)" if gld_up else "DOWN"
        gld_col = "green" if gld_up else "red"
        st.metric("Gold (GLD)", f"${cur['GLD']:.2f}")
        st.markdown(f":{gld_col}[**TREND: {gld_stat}**]")

st.markdown("---")
st.subheader("Execution Rules")
st.info("EXECUTION: DAILY @ 3:45 PM EST")
st.markdown("""
* **MACRO:** Requires BOTH VIX Inversion AND Credit Stress to trigger RED.
* **TREND:** QQQ requires dual-confirmation (Moving Averages + MACD) to exit.
* **OFFENSE:** If GREEN, hold current positions Mon-Thu. BUY TQQQ on Fridays ONLY.
* **DEFENSE:** If RED, instantly rotate to specified Hedge Scenario or Cash.
""")
