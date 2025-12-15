import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import time

# ==============================================================================
# STRATEGY DETAILS
# ==============================================================================
STRATEGY_DOCS = """
**FREQUENCY:** Execute WEEKLY on Fridays between 3:30 PM and 4:00 PM EST.  
**EXCEPTION:** DAILY Safety Check (Macro Filter) ONLY between 3:45PM and 4:00PM.

**OBJECTIVE:** Capture aggressive growth in Bull Markets while avoiding major drawdowns using Macro Filters, Asset Rotation, and Momentum Confirmation.

**LOGIC TREE:**

**1. SAFETY CHECK (MACRO FILTER) - Evaluated Daily at Close** * **Triggers:** a) Volatility Structure: Spot VIX > 3M VIX (^VIX > ^VIX3M) (Backwardation/Panic)  
    b) Credit Stress: High Yield (HYG) underperforms Treasuries (IEI) over 20 days.  
* **RULE:** If EITHER is True -> STATUS = RED (RISK OFF).

**2. TREND CHECK (PRICE & MOMENTUM FILTER)  - Evaluated Weekly at Close** * **Asset Selection:** Track QQQ (Tech) if it outperforms SPY over 63 days, else Track SPY.  
* **Triggers:** a) GREEN (BUY): Price > 200 SMA AND MACD > Signal Line (Positive Momentum).  
    b) YELLOW (HOLD): Price > 200 SMA BUT MACD < Signal Line (Weak Momentum/Whipsaw Risk).  
    c) RED (EXIT): Price < 200 SMA.

**3. ALLOCATION ENGINE (THE "WHAT TO BUY")**

* **IF SIGNAL IS GREEN (RISK ON):** * IF VIX < 20: Buy 3x Leverage (TQQQ or UPRO).  
    * IF VIX >= 20: Buy 2x Leverage (QLD or SSO).  
    * SAFETY: Set 30% Trailing Stop Loss (GTC) immediately. (Only for Black Swan events. Do not touch otherwise).

* **IF SIGNAL IS YELLOW (TRANSITION):** * HOLD current position. Do not buy, do not sell.

* **IF SIGNAL IS RED (DEFENSE ROTATION):** * Check US Dollar (UUP) Trend (vs 63 SMA).  
    * Check Gold (GLD) Trend (vs 200 SMA).
    * **SCENARIO A (CRASH/DEFLATION):** Stocks RED + Dollar UP (Flight to Safety) -> ACTION: Buy HEDGE BASKET (40% KMLM / 40% BTAL / 20% UUP)  
    * **SCENARIO B (STAGFLATION / DEVALUATION):** Stocks RED + Dollar DOWN + Gold UP -> ACTION: Buy GOLD (GLD) - "The Golden Parachute"  
    * **SCENARIO C (TOTAL APATHY / CHOP):** Stocks RED + Dollar DOWN + Gold DOWN -> ACTION: Buy CASH (SGOV)
"""

# --- CONFIGURATION ---
st.set_page_config(page_title="Roth Strategy", layout="centered")

ASSETS = {
    'TECH_3X': 'TQQQ', 'TECH_2X': 'QLD',
    'SPY_3X':  'UPRO', 'SPY_2X':  'SSO',
    'HEDGE':   '40% KMLM / 40% BTAL / 20% UUP',
    'GOLD':    '100% GLD (Gold)',
    'CASH':    '100% SGOV (Treasury Bills)'
}

TICKERS = ['SPY', 'QQQ', 'HYG', 'IEI', 'UUP', 'GLD', '^VIX', '^VIX3M']

# --- HELPER FUNCTIONS ---
def get_est_time():
    """Returns current time in US/Eastern."""
    utc_now = datetime.now(pytz.utc)
    est = pytz.timezone('US/Eastern')
    return utc_now.astimezone(est)

def fetch_data_with_retry(tickers):
    # Try fetching all at once first (fastest)
    try:
        # Use auto_adjust=False but grab 'Adj Close' to be safe
        data = yf.download(tickers, period="2y", progress=False, auto_adjust=False)
        
        # Check if we got a MultiIndex (common with multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # If 'Adj Close' exists, use it. Otherwise fallback to 'Close'
            if 'Adj Close' in data.columns.levels[0]:
                data = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                print("WARNING: 'Adj Close' missing. Using 'Close' (Dividends will skew signals).")
                data = data['Close']
        else:
            # Single level columns, sometimes yfinance returns this structure
            if 'Adj Close' in data:
                data = data['Adj Close']
            else:
                data = data['Close']

        # Verify we actually have data
        if data.empty or data.shape[1] < len(tickers):
            raise ValueError("Incomplete data returned")
            
        return data

    except Exception as e:
        print(f"Bulk download failed: {e}. Retrying individually...")
        
        # Fallback: Download one by one and combine (Slower but 99% reliable)
        combined_data = {}
        for t in tickers:
            try:
                # auto_adjust=True makes 'Close' = Adjusted Close automatically
                df = yf.download(t, period="2y", progress=False, auto_adjust=True)
                if not df.empty:
                    combined_data[t] = df['Close'] # Because auto_adjust=True, 'Close' IS Adjusted
                else:
                    print(f"Failed to fetch {t}")
            except Exception as e2:
                print(f"Error fetching {t}: {e2}")
        
        if not combined_data:
            return None
            
        return pd.DataFrame(combined_data)

# --- MAIN UI ---
st.title("ROTH STRATEGY: Friday 3:30PM")
st.caption(f"Server Time: {get_est_time().strftime('%Y-%m-%d %I:%M %p EST')}")

with st.expander("📄 Strategy Documentation (Click to Expand)"):
    st.markdown(STRATEGY_DOCS)

# Button to Run
if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    
    status_placeholder = st.empty()
    status_placeholder.info("Fetching Market Data...")

    try:
        with st.spinner("Fetching Market Data..."):
        # 1. Get Data
        data = fetch_data_with_retry(TICKERS)
        
        # --- DATA INTEGRITY CHECK ---
        if data is None or data.empty:
            status_placeholder.empty()
            st.error("Connection Failed: No data returned from API.")
            st.stop()
            
        # Handle MultiIndex (yfinance update standard)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)

        # Check for NaNs in the LAST row specifically (Today's Data)
        last_row = data.iloc[-1]
        nan_tickers = last_row[last_row.isna()].index.tolist()
        
        if nan_tickers:
            missing_str = ", ".join(nan_tickers)
            st.error(f"CRITICAL DATA MISSING (NaN): {missing_str}\n\nMarket data may be delayed or unavailable. Please try again in 15 minutes.")
            st.stop()

        # --- TIMESTAMP VALIDATION ---
        # 1. Get correct dates
        last_market_date = data.index[-1].date()
        est_now = get_est_time()
        current_est_date = est_now.date()

        # 2. Only check freshness on Weekdays (Mon-Fri)
        # (weekday 0=Mon, 4=Fri. So < 5 means it is a weekday)
        if current_est_date.weekday() < 5:
            if last_market_date != current_est_date:
                st.error(f"⚠️ DATA IS STALE! \n\nLast Market Date: {last_market_date}\nToday: {current_est_date}\n\nThe API has not returned today's price yet. Please wait.")
                st.stop()

        # 2. Extract Time Slices
        cur = data.iloc[-1]       # Today
        prev_20 = data.iloc[-21]  # 20 Trading Days ago
        prev_63 = data.iloc[-63]  # 63 Trading Days ago

        # --- CALCULATIONS ---

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

        # --- LOGIC ENGINE ---

        # 1. Determine Trend Status
        is_above_sma = track_price > sma_200
        
        if is_above_sma and macd_bullish:
            trend_status = "GREEN"
        elif not is_above_sma:
            trend_status = "RED"
        else:
            trend_status = "YELLOW"

        # 2. Determine Time Warning Suffix
        # 0 = Monday, 4 = Friday
        est_now = get_est_time()
        today_weekday = est_now.weekday()
        
        # Check if it's currently the Daily Macro Execution Time (3:45 PM EST)
        # We give a window of 3:40 PM - 4:00 PM for the "Execute Now" logic
        is_daily_close_window = (est_now.hour == 15 and est_now.minute >= 40)
        
        time_suffix = ""
        
        if not macro_safe:
            # PRIORITY 1: Macro Fire Alarm (Applies Every Day)
            if is_daily_close_window:
                time_suffix = " (⚠️ EXECUTE NOW - MACRO PANIC)"
            else:
                time_suffix = " (ONLY Execute if Red at 3:45PM EST)"
        
        elif today_weekday != 4:
            # PRIORITY 2: Not Friday (Wait)
            time_suffix = " (WAIT FOR FRIDAY)"
        
        else:
            # PRIORITY 3: Friday (Execute)
            time_suffix = ""

        # 3. Decision Matrix
        status_placeholder.empty()

        # --- RED LOGIC (Risk Off) ---
        if (not macro_safe) or (trend_status == "RED"):
            # Sub-Logic: Which defense?
            if uup_trending_up:
                asset_name = "HEDGE"
                asset_desc = ASSETS['HEDGE']
                why = "Risk Off + Dollar Rising (Deflation Defense)."
            elif gold_trending_up:
                asset_name = "GOLD"
                asset_desc = ASSETS['GOLD']
                why = "Risk Off + Dollar Falling + Gold Up (Stagflation Defense)."
            else:
                asset_name = "CASH"
                asset_desc = ASSETS['CASH']
                why = "Risk Off. No clear trend (Capital Preservation)."
            
            st.error(f"### 🔴 RED SIGNAL: {asset_name}{time_suffix}\n\n**BUY:** {asset_desc}\n\n*{why}*\n\nCheck Macro triggers.")

        # --- GREEN LOGIC (Risk On) ---
        elif trend_status == "GREEN":
            target_idx = "TECH" if tech_leads else "SPY"
            vix_spot = cur['^VIX']
            
            if vix_spot < 20:
                ticker = ASSETS[f'{target_idx}_3X']
                lev = "3x"
            else:
                ticker = ASSETS[f'{target_idx}_2X']
                lev = "2x"
            
            msg = f"### 🟢 GREEN SIGNAL: BUY{time_suffix}\n\n**BUY 100% {ticker} ({lev})**\n\n*Price > SMA and MACD Bullish. Set 30% Trailing Stop GTC.*"
            
            if "WAIT" in time_suffix:
                st.success(msg, icon="⏳") # Show as green but with hourglass if waiting
            else:
                st.success(msg)

        # --- YELLOW LOGIC (Hold) ---
        else:
            st.warning(f"### 🟡 YELLOW SIGNAL: HOLD{time_suffix}\n\n**HOLD CURRENT POSITION**\n\n*Price > SMA but MACD Bearish (Weak Momentum).*")

        # --- DATA GRID ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        # Col 1: Safety (VIX & Credit)
        with col1:
            st.subheader("1. Macro Safety")
            
            # VIX
            st.metric("VIX (Spot)", f"{cur['^VIX']:.2f}")
            st.metric("VIX (3M)", f"{cur['^VIX3M']:.2f}")
            
            if panic_active:
                st.markdown(":red[**STATUS: PANIC (Inverted)**]")
            else:
                st.markdown(":green[**STATUS: NORMAL**]")
            
            st.divider()
            
            # Credit
            st.metric("HYG (Risk)", f"{hyg_ret:.2%}")
            st.metric("IEI (Safe)", f"{iei_ret:.2%}")
            
            if credit_stress:
                st.markdown(":red[**STATUS: STRESS (Risk Off)**]")
            else:
                st.markdown(":green[**STATUS: HEALTHY**]")

        # Col 2: Trend
        with col2:
            st.subheader("2. Trend & Mom.")
            
            st.metric(f"Asset: {track_ticker}", f"${track_price:.2f}")
            st.metric("200 SMA", f"${sma_200:.2f}")
            
            macd_txt = "MACD UP" if macd_bullish else "MACD DOWN"
            
            if trend_status == "GREEN":
                st.markdown(f":green[**{trend_status} ({macd_txt})**]")
            elif trend_status == "RED":
                st.markdown(f":red[**{trend_status} ({macd_txt})**]")
            else:
                st.markdown(f":orange[**{trend_status} ({macd_txt})**]")

        # Col 3: Defense
        with col3:
            st.subheader("3. Defense Select")
            
            # Dollar
            uup_stat_txt = "UP" if uup_trending_up else "DOWN"
            uup_color = "green" if uup_trending_up else "red"
            st.metric("Dollar ($UUP)", f"${cur['UUP']:.2f}")
            st.markdown(f":{uup_color}[**TREND: {uup_stat_txt}**]")
            
            st.divider()
            
            # Gold
            gld_stat_txt = "UP" if gold_trending_up else "DOWN"
            gld_color = "green" if gold_trending_up else "red"
            st.metric("Gold ($GLD)", f"${cur['GLD']:.2f}")
            st.markdown(f":{gld_color}[**TREND: {gld_stat_txt}**]")

    except Exception as e:
        st.error(f"Data Error: {e}")
        
# --- LEGEND ---
st.markdown("---")
st.subheader("Strategy Rules & Legend")
st.info("EXECUTION: Fridays 3:30PM - 4:00PM EST. EXCEPT for Daily Macro 3:45PM - 4:00PM")

with st.expander("Show Detailed Legend", expanded=True):
    st.markdown("""
    * **DAILY CHECK (3:45 PM):** :red[**RED**] (Macro Unsafe) = VIX Inverted OR Credit Stress (Exit Immediately).
    * **FRIDAY CHECK (3:30 PM):**
        * :green[**GREEN**] = Price > SMA + MACD Bullish (Positive Momentum).
        * :orange[**YELLOW**] = Price > SMA but MACD Bearish (Weak Trend). Hold Position.
        * :red[**RED (HEDGE)**] = Price < SMA (Check Defense: Hedge -> Gold -> Cash).
    * :red[**SAFETY**]: Always maintain 30% Trailing Stop GTC for Black Swans.
    """)
