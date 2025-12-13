import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ASSETS = {
    'TECH_3X': 'TQQQ', 'TECH_2X': 'QLD',
    'SPY_3X':  'UPRO', 'SPY_2X':  'SSO',
    'HEDGE':   '40% KMLM / 40% BTAL / 20% UUP',
    'GOLD':    '100% GLD (Gold Trust)',
    'CASH':    '100% SGOV (Treasury Bills)'
}

TICKERS = ['SPY', 'QQQ', 'HYG', 'IEI', 'UUP', 'GLD', '^VIX', '^VIX3M']

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def get_eastern_time():
    """Returns the current time in US/Eastern."""
    utc_now = datetime.now(pytz.utc)
    eastern = pytz.timezone('US/Eastern')
    return utc_now.astimezone(eastern)

def fetch_data_with_retry(tickers, retries=3):
    """Robust fetcher that handles yfinance timeouts."""
    for i in range(retries):
        try:
            # Download 2 years to ensure valid 200 SMA
            data = yf.download(tickers, period="2y", progress=False)['Close']
            if not data.empty:
                return data
        except Exception as e:
            time.sleep(1)
    return pd.DataFrame()

# ==============================================================================
# STREAMLIT UI LAYOUT
# ==============================================================================
st.set_page_config(page_title="Roth IRA Strategy", layout="wide")

# Header
st.title("ΧΡΗΜΑΤΙΣΤΗΡΙΟ ROTH IRA")
st.subheader("Friday Close Strategy (MACD + Macro Filter)")

# Time Status
est_now = get_eastern_time()
st.caption(f"Server Time (EST): {est_now.strftime('%A, %Y-%m-%d %I:%M:%S %p')}")

# Run Button
if st.button("RUN ANALYSIS", type="primary"):
    
    with st.spinner("Fetching Market Data..."):
        try:
            # 1. Fetch Data
            data = fetch_data_with_retry(TICKERS)
            
            if data.empty:
                st.error("Connection Failed: No data returned from API.")
                st.stop()
            
            # Handle MultiIndex if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)

            # Check for NaNs in today's data
            last_row = data.iloc[-1]
            nan_tickers = last_row[last_row.isna()].index.tolist()
            if nan_tickers:
                st.warning(f"Warning: Data missing for {', '.join(nan_tickers)}. Using previous close.")
                data = data.ffill()

            # 2. Extract Time Slices
            cur = data.iloc[-1]       # Today
            prev_20 = data.iloc[-21]  # 20 Trading Days ago
            prev_63 = data.iloc[-63]  # 63 Trading Days ago

            # ==================================================================
            # CALCULATIONS (Identical to strategy.py)
            # ==================================================================
            
            # A. Volatility Structure (Panic Check)
            panic_active = cur['^VIX'] > cur['^VIX3M']
            
            # B. Credit Stress (HYG vs IEI)
            hyg_ret = (cur['HYG'] - prev_20['HYG']) / prev_20['HYG']
            iei_ret = (cur['IEI'] - prev_20['IEI']) / prev_20['IEI']
            credit_stress = hyg_ret < iei_ret
            
            # MACRO SAFE SWITCH
            macro_safe = not (panic_active or credit_stress)

            # C. Trend & Asset Selection
            tech_perf = (cur['QQQ'] - prev_63['QQQ']) / prev_63['QQQ']
            spy_perf = (cur['SPY'] - prev_63['SPY']) / prev_63['SPY']
            tech_leads = tech_perf > spy_perf

            track_ticker = "QQQ" if tech_leads else "SPY"
            track_price = cur[track_ticker]
            
            # Moving Average
            sma_200 = data[track_ticker].rolling(200).mean().iloc[-1]
            
            # MACD Calculation
            exp12 = data[track_ticker].ewm(span=12, adjust=False).mean()
            exp26 = data[track_ticker].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

            # D. Defensive Trends
            sma_uup_63 = data['UUP'].rolling(63).mean().iloc[-1] 
            sma_gold_200 = data['GLD'].rolling(200).mean().iloc[-1] 
            uup_trending_up = cur['UUP'] > sma_uup_63
            gold_trending_up = cur['GLD'] > sma_gold_200

            # ==================================================================
            # LOGIC ENGINE
            # ==================================================================

            # 1. Determine Trend Status
            is_above_sma = track_price > sma_200
            
            if is_above_sma and macd_bullish:
                trend_status = "GREEN"
            elif not is_above_sma:
                trend_status = "RED"
            else:
                trend_status = "YELLOW"

            # 2. Determine Time Warning
            # 0 = Monday, 4 = Friday
            today_weekday = est_now.weekday()
            
            if not macro_safe:
                time_msg = "⚠️ EXECUTE NOW (Macro Alarm)"
            elif today_weekday != 4:
                time_msg = "⏳ WAIT FOR FRIDAY"
            else:
                time_msg = "✅ FRIDAY EXECUTION"

            # 3. Decision Matrix
            signal_type = "" # Green, Yellow, Red
            allocation_text = ""
            reason_text = ""

            # --- RED LOGIC (Risk Off) ---
            if (not macro_safe) or (trend_status == "RED"):
                signal_type = "RED"
                if uup_trending_up:
                    allocation_text = f"BUY: {ASSETS['HEDGE']}"
                    reason_text = "Risk Off + Dollar Rising (Deflation Defense)."
                elif gold_trending_up:
                    allocation_text = f"BUY: {ASSETS['GOLD']}"
                    reason_text = "Risk Off + Dollar Falling + Gold Up (Stagflation Defense)."
                else:
                    allocation_text = f"BUY: {ASSETS['CASH']}"
                    reason_text = "Risk Off + No Clear Trend (Cash Preservation)."
                
                reason_text += "\n(Macro Unsafe)" if not macro_safe else "\n(Price < SMA)"

            # --- GREEN LOGIC (Risk On) ---
            elif trend_status == "GREEN":
                signal_type = "GREEN"
                target_idx = "TECH" if tech_leads else "SPY"
                vix_spot = cur['^VIX']
                
                if vix_spot < 20:
                    ticker = ASSETS[f'{target_idx}_3X']
                    lev = "3x"
                else:
                    ticker = ASSETS[f'{target_idx}_2X']
                    lev = "2x"
                
                allocation_text = f"BUY 100% {ticker} ({lev})"
                reason_text = "Price > SMA and MACD Bullish.\nSet 30% Trailing Stop GTC."

            # --- YELLOW LOGIC (Hold) ---
            else:
                signal_type = "YELLOW"
                allocation_text = "HOLD CURRENT POSITION"
                reason_text = "Price > SMA but MACD Bearish (Weak Momentum)."

            # ==================================================================
            # DISPLAY RESULTS
            # ==================================================================
            
            # Main Signal Banner
            if signal_type == "RED":
                st.error(f"### 🔴 RED SIGNAL: {allocation_text}")
            elif signal_type == "GREEN":
                st.success(f"### 🟢 GREEN SIGNAL: {allocation_text}")
            else:
                st.warning(f"### 🟡 YELLOW SIGNAL: {allocation_text}")
            
            st.info(f"**Logic:** {reason_text} | **Status:** {time_msg}")

            st.divider()

            # Detailed Metrics (3 Columns)
            col1, col2, col3 = st.columns(3)

            # Column 1: Macro Safety
            with col1:
                st.subheader("1. Macro Safety")
                
                # VIX Logic
                vix_delta = cur['^VIX'] - cur['^VIX3M']
                vix_color = "normal" if not panic_active else "inverse"
                st.metric("VIX Term Structure", 
                          f"{cur['^VIX']:.2f} vs {cur['^VIX3M']:.2f}",
                          f"{'PANIC' if panic_active else 'NORMAL'}",
                          delta_color=vix_color)
                
                # Credit Logic
                st.metric("HYG vs IEI (20d)", 
                          f"{(hyg_ret - iei_ret)*100:.2f}% Spread",
                          f"{'STRESS' if credit_stress else 'HEALTHY'}",
                          delta_color="normal" if not credit_stress else "inverse")

            # Column 2: Trend
            with col2:
                st.subheader("2. Trend & Momentum")
                st.metric(f"Asset ({track_ticker})", 
                          f"${track_price:.2f}",
                          f"{'Above' if is_above_sma else 'Below'} SMA 200")
                
                st.metric("Momentum (MACD)", 
                          f"{'BULLISH' if macd_bullish else 'BEARISH'}",
                          f"{trend_status} STATE")

            # Column 3: Defense
            with col3:
                st.subheader("3. Defense Select")
                st.metric("US Dollar (UUP)", 
                          f"${cur['UUP']:.2f}",
                          f"{'UPTREND' if uup_trending_up else 'DOWNTREND'}")
                
                st.metric("Gold (GLD)", 
                          f"${cur['GLD']:.2f}",
                          f"{'UPTREND' if gold_trending_up else 'DOWNTREND'}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer Logic Legend
with st.expander("Strategy Rules & Legend"):
    st.markdown("""
    **EXECUTION:** Fridays 3:30PM - 4:00PM EST.
    
    * **Daily Check (3:45 PM):** If Macro is Unsafe (Red), Exit Immediately.
    * **Weekly Check (Friday):**
        * 🟢 **GREEN:** Price > SMA + MACD Bullish. (Buy 3x or 2x Leverage).
        * 🟡 **YELLOW:** Price > SMA + MACD Bearish. (Hold Position).
        * 🔴 **RED:** Price < SMA. (Go to Defense).
    
    **DEFENSE ROTATION:**
    1. **Hedge:** If Stocks Red + Dollar UP.
    2. **Gold:** If Stocks Red + Dollar DOWN + Gold UP.
    3. **Cash:** If Stocks Red + Dollar DOWN + Gold DOWN.
    """)
