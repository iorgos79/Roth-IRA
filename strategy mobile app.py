import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIGURATION & ASSETS
# ==============================================================================
st.set_page_config(page_title="Strategy Roth IRA: Friday Close", layout="wide")

ASSETS = {
    'TECH_3X': 'TQQQ', 'TECH_2X': 'QLD',
    'SPY_3X':  'UPRO', 'SPY_2X':  'SSO',
    'HEDGE':   '40% KMLM / 40% BTAL / 20% UUP',
    'GOLD':    '100% GLD (Gold)',
    'CASH':    '100% SGOV (Treasury Bills)'
}

# Tickers needed for calculation
TICKERS = ['SPY', 'QQQ', 'HYG', 'IEI', 'UUP', 'GLD', '^VIX', '^VIX3M']

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def fetch_data(tickers):
    """
    Fetches the last 2 years of data for the required tickers.
    Uses st.cache_data to prevent re-downloading on minor UI interactions,
    but sets a short TTL (time to live) so data stays fresh.
    """
    try:
        data = yf.download(tickers, period="2y", progress=False)
        
        # Handle MultiIndex (yfinance update standard)
        if isinstance(data.columns, pd.MultiIndex):
            # We only want 'Close' prices usually, but yf.download(group_by='ticker') 
            # might change structure. Standard yf.download returns (Price, Ticker).
            # We select 'Close' level if it exists.
            try:
                data = data['Close']
            except KeyError:
                pass 
                
        if data.empty:
            return None
            
        return data
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

def analyze_strategy(data):
    """
    Executes the exact logic tree provided in the original script.
    """
    # Check for NaNs in the LAST row specifically (Today's Data)
    last_row = data.iloc[-1]
    nan_tickers = last_row[last_row.isna()].index.tolist()
    
    if nan_tickers:
        st.error(f"CRITICAL DATA MISSING (NaN): {', '.join(nan_tickers)}. Market data may be delayed.")
        return None

    # 3. Extract Time Slices
    cur = data.iloc[-1]       # Today (Friday Close)
    prev_20 = data.iloc[-21]  # 20 Trading Days ago
    prev_63 = data.iloc[-63]  # 63 Trading Days ago (Quarter)
    
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
    
    # 1. Performance Check (Who is leading?)
    tech_perf = (cur['QQQ'] - prev_63['QQQ']) / prev_63['QQQ']
    spy_perf = (cur['SPY'] - prev_63['SPY']) / prev_63['SPY']
    tech_leads = tech_perf > spy_perf

    track_ticker = "QQQ" if tech_leads else "SPY"
    track_price = cur[track_ticker]
    
    # 2. Moving Average
    sma_200 = data[track_ticker].rolling(200).mean().iloc[-1]
    
    # 3. MACD Calculation (Momentum Check)
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
    is_above_sma = track_price > sma_200
    
    trend_status = "YELLOW" # Default
    if is_above_sma and macd_bullish:
        trend_status = "GREEN"
    elif not is_above_sma:
        trend_status = "RED"
    else:
        trend_status = "YELLOW"

    # Decision Matrix Construction
    result = {
        'status_color': 'grey',
        'title': 'WAITING',
        'allocation': '-',
        'reason': '-',
        'metrics': {
            'vix': cur['^VIX'], 'vix3m': cur['^VIX3M'], 'panic': panic_active,
            'hyg_ret': hyg_ret, 'iei_ret': iei_ret, 'credit_stress': credit_stress,
            'track_ticker': track_ticker, 'track_price': track_price, 'sma': sma_200,
            'trend_status': trend_status, 'macd_bullish': macd_bullish,
            'uup': cur['UUP'], 'uup_up': uup_trending_up,
            'gld': cur['GLD'], 'gld_up': gold_trending_up
        }
    }

    # RED LOGIC
    if (not macro_safe) or (trend_status == "RED"):
        if uup_trending_up:
            result['title'] = "🔴 RED SIGNAL: HEDGE"
            result['allocation'] = f"BUY: {ASSETS['HEDGE']}"
            result['reason'] = "Risk Off (Macro or Trend) + Dollar Rising. Use Managed Futures to hedge crash."
            result['status_color'] = 'red'
        elif gold_trending_up:
            result['title'] = "🔴 RED SIGNAL: GOLD"
            result['allocation'] = f"BUY: {ASSETS['GOLD']}"
            result['reason'] = "Risk Off + Dollar Falling. Gold is trending up (Stagflation Defense)."
            result['status_color'] = 'gold' # Custom handling
        else:
            result['title'] = "🔴 RED SIGNAL: CASH"
            result['allocation'] = f"BUY: {ASSETS['CASH']}"
            result['reason'] = "Risk Off. No clear trend in Dollar or Gold. Preserve Capital in Cash."
            result['status_color'] = 'orange' # Custom handling

    # GREEN LOGIC
    elif trend_status == "GREEN":
        result['title'] = "🟢 GREEN SIGNAL: BUY"
        result['status_color'] = 'green'
        
        target_idx = "TECH" if tech_leads else "SPY"
        vix_spot = cur['^VIX']
        if vix_spot < 20:
            ticker = ASSETS[f'{target_idx}_3X']
            lev_desc = "3x Leverage"
        else:
            ticker = ASSETS[f'{target_idx}_2X']
            lev_desc = "2x Leverage"
        
        result['allocation'] = f"BUY 100% {ticker} ({lev_desc})"
        result['reason'] = f"Price > SMA and MACD Bullish.\nIMPORTANT: Set 30% Trailing Stop (Black Swan protection)."

    # YELLOW LOGIC
    else:
        result['title'] = "🟡 YELLOW SIGNAL: HOLD"
        result['status_color'] = 'yellow'
        result['allocation'] = "HOLD CURRENT POSITION"
        result['reason'] = "Price > SMA but MACD Bearish (Weak Momentum). Do not Buy new positions. Do not Sell yet."

    return result

# ==============================================================================
# MAIN UI LAYOUT
# ==============================================================================

st.title("Roth IRA Strategy ΠΟΛΛΑ ΛΕΦΤΑ: Friday Close")
st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}*")

# Button to refresh data
if st.button("Run Strategy Analysis", type="primary"):
    with st.spinner('Fetching market data...'):
        data = fetch_data(TICKERS)
        
        if data is not None:
            res = analyze_strategy(data)
            
            if res:
                # --- MAIN SIGNAL DISPLAY ---
                # We use custom Markdown for specific coloring similar to the Tkinter app
                color_map = {
                    'green': '#e6fffa', 'red': '#fff5f5', 'yellow': '#fffff0', 
                    'gold': '#fffaf0', 'orange': '#fffaf0'
                }
                border_map = {
                    'green': '#38a169', 'red': '#e53e3e', 'yellow': '#d69e2e',
                    'gold': '#d69e2e', 'orange': '#dd6b20'
                }
                
                bg_color = color_map.get(res['status_color'], '#f0f0f0')
                border_color = border_map.get(res['status_color'], '#ccc')
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 10px solid {border_color}; margin-bottom: 25px;">
                    <h2 style="color: #333; margin:0;">{res['title']}</h2>
                    <h3 style="color: #000;">{res['allocation']}</h3>
                    <p style="color: #555;">{res['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # --- METRICS GRID ---
                m = res['metrics']
                
                # Column 1: Macro
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.subheader("1. Macro Safety")
                    st.metric("VIX (Spot)", f"{m['vix']:.2f}", delta=f"{m['vix'] - m['vix3m']:.2f} vs 3M", delta_color="inverse")
                    
                    if m['panic']:
                        st.error("VOLATILITY: PANIC (Inverted)")
                    else:
                        st.success("VOLATILITY: NORMAL")
                        
                    st.divider()
                    st.metric("HYG Return (20d)", f"{m['hyg_ret']*100:.2f}%")
                    st.metric("IEI Return (20d)", f"{m['iei_ret']*100:.2f}%")
                    
                    if m['credit_stress']:
                        st.error("CREDIT: STRESS (Risk Off)")
                    else:
                        st.success("CREDIT: HEALTHY")

                # Column 2: Trend
                with c2:
                    st.subheader("2. Trend & Momentum")
                    st.metric(f"Tracked Asset ({m['track_ticker']})", f"${m['track_price']:.2f}")
                    st.metric("200 SMA", f"${m['sma']:.2f}")
                    
                    trend_txt = f"{m['trend_status']} ({'MACD UP' if m['macd_bullish'] else 'MACD DOWN'})"
                    if m['trend_status'] == "GREEN":
                        st.success(f"STATUS: {trend_txt}")
                    elif m['trend_status'] == "YELLOW":
                        st.warning(f"STATUS: {trend_txt}")
                    else:
                        st.error(f"STATUS: {trend_txt}")

                # Column 3: Defense
                with c3:
                    st.subheader("3. Defense Select")
                    st.metric("Dollar ($UUP)", f"${m['uup']:.2f}", delta="Trending Up" if m['uup_up'] else "Trending Down")
                    st.metric("Gold ($GLD)", f"${m['gld']:.2f}", delta="Trending Up" if m['gld_up'] else "Trending Down")

# --- STRATEGY LEGEND (EXPANDER) ---
with st.expander("Strategy Rules & Documentation"):
    st.markdown("""
    **EXECUTION:** Fridays 3:30PM - 4:00PM EST.
    
    1. **🔴 RED (Macro Unsafe):** - VIX Inverted (^VIX > ^VIX3M) OR Credit Stress (HYG < IEI).
       - *Action:* Exit Stocks Immediately.
       
    2. **🟢 GREEN (Buy):**
       - Price > 200 SMA **AND** MACD Bullish.
       - *Action:* Buy Leverage (3x if VIX < 20, else 2x).
       
    3. **🟡 YELLOW (Hold):**
       - Price > 200 SMA but MACD Bearish (Weak Trend).
       - *Action:* Hold Position. Do not buy/sell.
       
    4. **🔴 RED (Defense Rotation):**
       - If Price < 200 SMA or Macro Unsafe:
       - **Scenario A:** Dollar Up -> Buy **HEDGE** (KMLM/BTAL/UUP).
       - **Scenario B:** Dollar Down + Gold Up -> Buy **GOLD**.
       - **Scenario C:** Both Down -> Buy **CASH**.
       
    *SAFETY:* Always maintain 30% Trailing Stop GTC.
    """)
