"""
StockSense AI — Streamlit Dashboard
=====================================
Run:
  streamlit run ui/app.py
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = "http://api:8000"

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .stApp { background: #0a0e1a; color: #e8eaf0; }

  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-value { font-family: 'Space Mono', monospace; font-size: 2em; font-weight: 700; }
  .metric-label { color: #7b8fa8; font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

  .signal-buy  { color: #00e676; }
  .signal-sell { color: #ff5252; }
  .signal-hold { color: #ffd740; }

  .factor-card {
    background: #111827;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9em;
  }
  .factor-bull { border-color: #00e676; }
  .factor-bear { border-color: #ff5252; }

  .section-title {
    font-family: 'Space Mono', monospace;
    color: #5c8adb;
    font-size: 0.78em;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 6px;
  }
  div[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e2d45; }
  .stButton button {
    background: linear-gradient(135deg, #1e4db7, #0f2d8a);
    color: white; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 600; width: 100%;
    transition: opacity 0.2s;
  }
  .stButton button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_health() -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def generate_demo_candles(n: int = 200, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0005, 0.014, n)
    close   = 150.0 * np.exp(np.cumsum(log_ret))
    noise   = rng.uniform(0.003, 0.018, n)
    high    = close * (1 + noise)
    low     = close * (1 - noise)
    open_   = close * (1 + rng.normal(0, 0.005, n))
    volume  = rng.lognormal(14.5, 0.6, n).astype(int)
    dates   = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return [
        {
            "date":   str(dates[i].date()),
            "open":   round(float(open_[i]), 4),
            "high":   round(float(high[i]),  4),
            "low":    round(float(low[i]),   4),
            "close":  round(float(close[i]), 4),
            "volume": int(volume[i]),
        }
        for i in range(n)
    ]


def call_predict(ticker: str, candles: List[Dict]) -> Optional[Dict]:
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json={"ticker": ticker, "candles": candles},
            timeout=30,
        )
        if r.ok:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the backend is running.")
        return None


def candlestick_chart(candles: List[Dict]) -> go.Figure:
    df = pd.DataFrame(candles)
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#00e676",
        decreasing_line_color="#ff5252",
        name="Price",
    ), row=1, col=1)

    # EMA overlays
    c = df["close"]
    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=ema9,  name="EMA 9",
                             line=dict(color="#5c8adb", width=1.2), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=ema21, name="EMA 21",
                             line=dict(color="#f39c12", width=1.2), opacity=0.8), row=1, col=1)

    # Volume bars
    colors = ["#00e676" if c >= o else "#ff5252"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=colors,
                         opacity=0.6, name="Volume"), row=2, col=1)

    fig.update_layout(
        plot_bgcolor  = "#0a0e1a",
        paper_bgcolor = "#0a0e1a",
        font_color    = "#7b8fa8",
        height        = 480,
        margin        = dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible = False,
        legend = dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        xaxis2 = dict(showgrid=False, color="#7b8fa8"),
        yaxis  = dict(gridcolor="#1e2d45", color="#7b8fa8"),
        yaxis2 = dict(gridcolor="#1e2d45", color="#7b8fa8"),
    )
    return fig


def shap_waterfall_chart(factors: List[Dict], baseline: float = 0.5) -> go.Figure:
    top = sorted(factors, key=lambda x: abs(x["shap_value"]), reverse=True)[:10]
    top = sorted(top, key=lambda x: x["shap_value"])

    names  = [f["display_name"] for f in top]
    vals   = [f["shap_value"]   for f in top]
    colors = ["#00e676" if v > 0 else "#ff5252" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=11),
    ))
    fig.add_vline(x=0, line_color="#7b8fa8", line_width=1, line_dash="dash")
    fig.update_layout(
        plot_bgcolor  = "#0a0e1a",
        paper_bgcolor = "#0a0e1a",
        font_color    = "#e8eaf0",
        height        = 360,
        margin        = dict(l=0, r=60, t=10, b=0),
        xaxis = dict(title="SHAP Contribution", gridcolor="#1e2d45", color="#7b8fa8"),
        yaxis = dict(gridcolor="#1e2d45", color="#e8eaf0"),
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSense AI")
    st.markdown("<div style='color:#7b8fa8;font-size:0.85em;margin-bottom:20px'>AI-Powered Stock Direction Prediction</div>", unsafe_allow_html=True)

    health = fetch_health()
    if health:
        st.markdown(f"""
        <div style='background:#0d1f12;border:1px solid #1a3d20;border-radius:8px;padding:12px;margin-bottom:16px'>
          <div style='color:#00e676;font-weight:600;margin-bottom:6px'>● API Online</div>
          <div style='color:#7b8fa8;font-size:0.8em'>Model v{health.get("model_version","?")}
          &nbsp;|&nbsp; AUC {health.get("test_auc", 0):.3f}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1f0d0d;border:1px solid #3d1a1a;border-radius:8px;padding:12px;margin-bottom:16px'>
          <div style='color:#ff5252;font-weight:600'>● API Offline</div>
          <div style='color:#7b8fa8;font-size:0.8em'>Start with docker-compose up</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=8).upper()
    n_candles = st.slider("Demo candles to generate", 100, 500, 200, step=20)
    seed = st.number_input("Random seed (demo data)", value=42, min_value=0, max_value=9999)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    <div style='color:#7b8fa8;font-size:0.82em;line-height:1.7'>
    <b>StockSense AI</b> predicts whether a stock will move <b>+1.5%</b> or more over the next <b>5 trading days</b>.<br><br>
    Features: 27 technical indicators including RSI, MACD, Bollinger Bands, ATR & momentum signals.<br><br>
    Model: XGBoost + SHAP explainability
    </div>""", unsafe_allow_html=True)


# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:24px 0 12px 0'>
  <span style='font-family:Space Mono,monospace;font-size:1.6em;font-weight:700;color:#e8eaf0'>
    StockSense <span style='color:#5c8adb'>AI</span>
  </span>
  <span style='color:#7b8fa8;font-size:0.9em;margin-left:14px'>
    Stock Direction Prediction · {ticker}
  </span>
</div>""", unsafe_allow_html=True)

# Generate demo data
candles = generate_demo_candles(n=n_candles, seed=seed)
df_chart = pd.DataFrame(candles)

# Chart
st.plotly_chart(candlestick_chart(candles[-120:]), use_container_width=True)

# Predict Button
col_btn, col_spacer = st.columns([1, 3])
with col_btn:
    run = st.button("🔍 Analyse & Predict")

if run:
    with st.spinner("Running AI analysis..."):
        result = call_predict(ticker, candles)

    if result:
        st.markdown("---")

        # ── Signal Banner ─────────────────────────────────────────────────────
        signal     = result["signal"]
        confidence = result["confidence"]
        sig_class  = "signal-buy" if signal == "BUY" else "signal-sell"
        sig_emoji  = "🟢" if signal == "BUY" else "🔴"
        sig_text   = "STRONG BUY" if confidence > 0.70 else ("BUY" if signal == "BUY" else "HOLD / SELL")

        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#111827,#1a2235);border:1px solid #1e2d45;
                    border-radius:14px;padding:28px 32px;margin:16px 0;text-align:center'>
          <div class='{sig_class}' style='font-family:Space Mono,monospace;font-size:2.4em;font-weight:700'>
            {sig_emoji} {sig_text}
          </div>
          <div style='color:#7b8fa8;margin-top:8px;font-size:0.95em'>
            {ticker} · Next {result["forward_days"]} Trading Days &nbsp;|&nbsp;
            Confidence: <b style='color:#e8eaf0'>{confidence:.1%}</b>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Metrics Row ───────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        latest = candles[-1]
        prev   = candles[-2]
        day_chg = (latest["close"] - prev["close"]) / prev["close"] * 100

        with c1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='color:#e8eaf0'>${latest["close"]:.2f}</div>
              <div class='metric-label'>Latest Close</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            col = "#00e676" if day_chg >= 0 else "#ff5252"
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='color:{col}'>{day_chg:+.2f}%</div>
              <div class='metric-label'>Day Change</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='color:#5c8adb'>{confidence:.1%}</div>
              <div class='metric-label'>AI Confidence</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-value' style='color:#f39c12'>{result["prediction_time_ms"]:.0f}ms</div>
              <div class='metric-label'>Inference Time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── SHAP + Factors ────────────────────────────────────────────────────
        col_shap, col_factors = st.columns([1.4, 1])

        with col_shap:
            st.markdown("<div class='section-title'>Feature Impact (SHAP)</div>", unsafe_allow_html=True)
            st.plotly_chart(shap_waterfall_chart(result["top_factors"]), use_container_width=True)

        with col_factors:
            st.markdown("<div class='section-title'>Top Signals</div>", unsafe_allow_html=True)
            for f in result["top_factors"][:6]:
                cls = "factor-bull" if f["direction"] == "bullish" else "factor-bear"
                icon = "▲" if f["direction"] == "bullish" else "▼"
                color = "#00e676" if f["direction"] == "bullish" else "#ff5252"
                st.markdown(f"""
                <div class='factor-card {cls}'>
                  <span style='color:{color};font-weight:700'>{icon}</span>
                  &nbsp;<b>{f["display_name"]}</b><br>
                  <span style='color:#7b8fa8;font-size:0.88em'>
                    Value: {f["value"]:.4f} &nbsp;|&nbsp;
                    SHAP: <span style='color:{color}'>{f["shap_value"]:+.4f}</span>
                  </span>
                </div>""", unsafe_allow_html=True)

        # ── Raw Feature Table ─────────────────────────────────────────────────
        with st.expander("🔬 All 27 Raw Feature Values"):
            feat_df = pd.DataFrame([
                {"Feature": METADATA_DISPLAY.get(k, k) if (METADATA_DISPLAY := result.get("raw_features", {})) else k,
                 "Value": round(v, 6)}
                for k, v in result.get("raw_features", {}).items()
            ])
            st.dataframe(feat_df, use_container_width=True, height=360)

else:
    # Placeholder when no prediction yet
    st.markdown("""
    <div style='background:#111827;border:1px dashed #1e2d45;border-radius:12px;
                padding:40px;text-align:center;margin-top:16px'>
      <div style='font-size:2.5em;margin-bottom:12px'>📊</div>
      <div style='color:#7b8fa8;font-size:1em'>
        Adjust the settings in the sidebar and click <b style='color:#5c8adb'>Analyse & Predict</b>
        to run the AI model on the chart data.
      </div>
    </div>""", unsafe_allow_html=True)
