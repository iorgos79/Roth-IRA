import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
ASSETS = {
    'TECH_3X': 'TQQQ', 'TECH_2X': 'QLD',
    'SPY_3X': 'UPRO',  'SPY_2X': 'SSO',
    'HEDGE': '40% KMLM / 40% BTAL / 20% UUP',
    'CASH': '100% SGOV (Treasury Bills)'
}

# Set page config for mobile friendliness
st.set_page_config(page_title="Roth Strategy", layout="centered")

# --- HEADER ---
st.title("🛡️ Strategy ROTH IRA")
st.caption(f"Active Defense Dashboard | {datetime.now().strftime('%Y-%m-%d')}")

# --- LOGIC ---
if st.button("RUN ANALYSIS", type="primary"):
    with st.spinner("Fetching Market Data..."):
        try:
            # 1. Define Tickers
            tickers = ['SPY', 'QQQ', 'HYG', 'IEI', 'UUP', '^VIX', '^VIX3M']
            
            # 2. Download Data
            data = yf.download(tickers, period="2y", progress=False)['Close']
            
            if data.empty:
                st.error("No data returned.")
                st.stop()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)

            # 3. Extract Data
            cur = data.iloc[-1]
            prev_5 = data.iloc[-6]
            prev_20 = data.iloc[-21]
            prev_63 = data.iloc[-63]

            # --- CALCULATIONS ---
            # A. Volatility
            vix_spot = cur['^VIX']
            vix_future = cur['^VIX3M']
            panic_safe = vix_spot < vix_future
            vix_5d_chg = (vix_spot - prev_5['^VIX']) / prev_5['^VIX']
            vol_crush_active = vix_5d_chg < -0.20

            # B. Credit
            hyg_ret = (cur['HYG'] - prev_20['HYG']) / prev_20['HYG']
            iei_ret = (cur['IEI'] - prev_20['IEI']) / prev_20['IEI']
            credit_safe = hyg_ret > iei_ret
            macro_safe = credit_safe and panic_safe

            # C. Trend
            tech_perf = (cur['QQQ'] - prev_63['QQQ']) / prev_63['QQQ']
            spy_perf = (cur['SPY'] - prev_63['SPY']) / prev_63['SPY']
            tech_leads = tech_perf > spy_perf
            
            sma_spy = data['SPY'].rolling(200).mean().iloc[-1]
            sma_qqq = data['QQQ'].rolling(200).mean().iloc[-1]

            # D. Active Defense
            sma_uup = data['UUP'].rolling(63).mean().iloc[-1]
            uup_trending_up = cur['UUP'] > sma_uup

            # --- DECISION ENGINE ---
            use_qqq_trend = tech_leads and (vix_spot < 20)
            track_ticker = "QQQ" if use_qqq_trend else "SPY"
            track_price = cur['QQQ'] if use_qqq_trend else cur['SPY']
            track_sma = sma_qqq if use_qqq_trend else sma_spy

            if track_price > (track_sma * 1.04):
                trend_status = "GREEN"
            elif track_price < track_sma:
                trend_status = "RED"
            else:
                trend_status = "YELLOW"

            # --- SIGNALS ---
            if (not macro_safe) or (trend_status == "RED"):
                if uup_trending_up:
                    st.error("🔴 RED SIGNAL: DEFENSE")
                    st.info(f"**BUY:** {ASSETS['HEDGE']}")
                    st.write("Reason: Risk High + Dollar Rising. Hold Hedge Basket.")
                else:
                    st.warning("🟠 RED SIGNAL: CASH")
                    st.info(f"**BUY:** {ASSETS['CASH']}")
                    st.write("Reason: Risk High + Dollar Falling. Move to Cash.")

            elif trend_status == "GREEN":
                st.success("🟢 GREEN SIGNAL: BUY")
                target_idx = "TECH" if tech_leads else "SPY"
                if vix_spot < 20:
                    ticker = ASSETS[f'{target_idx}_3X']
                    lev_desc = "3x Lev"
                else:
                    ticker = ASSETS[f'{target_idx}_2X']
                    lev_desc = "2x Lev"
                st.info(f"**BUY 100% {ticker}** ({lev_desc})")
                st.write(f"Trend Green. Volatility {vix_spot:.1f}. Set 20% Trailing Stop.")

            else:
                st.warning("🟡 YELLOW SIGNAL: HOLD")
                st.info("HOLD CURRENT POSITION")
                st.write("Price in buffer zone. Maintain exposure.")

            st.divider()

            # --- METRICS GRID ---
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("VIX", f"{vix_spot:.2f}", delta=f"Future: {vix_future:.2f}", delta_color="inverse")
                st.write(f"Crush: **{'YES' if vol_crush_active else 'NO'}**")
            
            with c2:
                st.metric("HYG (20d)", f"{hyg_ret:.1%}")
                st.metric("IEI (20d)", f"{iei_ret:.1%}")
                st.write(f"Credit: **{'RISK ON' if credit_safe else 'RISK OFF'}**")

            with c3:
                st.metric(f"{track_ticker}", f"{track_price:.2f}")
                st.metric("SMA", f"{track_sma:.2f}")
                st.write(f"Trend: **{trend_status}**")

        except Exception as e:
            st.error(f"Error: {e}")

# --- LEGEND ---
with st.expander("Strategy Logic"):
    st.markdown("""
    * **🟢 GREEN:** 3x Lev (VIX < 20) or 2x Lev (VIX > 20)
    * **🟡 YELLOW:** Buffer Zone. Hold Position.
    * **🔴 RED (HEDGE):** Trend Down + Dollar UP -> Buy KMLM/BTAL/UUP
    * **🟠 RED (CASH):** Trend Down + Dollar DOWN -> Buy SGOV
    """)
