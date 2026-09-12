import os
import sys
import subprocess
import streamlit as st

# -------------------------------------------------------------------
# Dynamically pull Kronos directly from GitHub URL
# -------------------------------------------------------------------
KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
REPO_DIR = "Kronos_Source"

if not os.path.exists(REPO_DIR):
    with st.spinner("Fetching Kronos AI repository from GitHub..."):
        subprocess.run(["git", "clone", KRONOS_REPO_URL, REPO_DIR], check=True)

# Add cloned repo path to Python path
if os.path.abspath(REPO_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(REPO_DIR))

# Import Kronos modules from the dynamically loaded repo
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError as e:
    st.error(f"Failed to load Kronos model modules: {e}")
    st.stop()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import torch

# -------------------------------------------------------------------
# Page Config
# -------------------------------------------------------------------
st.set_page_config(page_title="Kronos Stock AI", page_icon="📈", layout="wide")
st.title("📈 Kronos Stock Market AI Forecaster")

# -------------------------------------------------------------------
# Model Loader
# -------------------------------------------------------------------
MODEL_MAP = {
    "Kronos-mini (4.1M params)": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "Kronos-small (24.7M params)": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "Kronos-base (102.3M params)": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

@st.cache_resource(show_spinner="Loading Kronos AI weights...")
def load_predictor(model_choice_key, device_type):
    cfg = MODEL_MAP[model_choice_key]
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    model.to(device_type)
    return KronosPredictor(model, tokenizer, max_context=cfg["max_context"]), cfg["max_context"]

# Sidebar Options
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Model Variant", list(MODEL_MAP.keys()), index=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Compute Device: **{device.upper()}**")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m", "5m"], index=0)
period = st.sidebar.selectbox("Period", ["60d", "100d", "1y", "2y"], index=2)

pred_len = st.sidebar.slider("Forecast Steps Ahead", 5, 120, 30, 5)
lookback = st.sidebar.slider("Lookback Window", 50, 512, 300, 10)
sample_count = st.sidebar.slider("Monte Carlo Samples", 1, 20, 8, 1)

# Fetch Market Data
@st.cache_data(ttl=300)
def get_data(symbol, interval, period):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df = df.rename(columns={date_col: 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df['amount'] = df['close'] * df['volume']
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

df = get_data(ticker, interval, period)

if df is not None:
    st.subheader(f"Data: {ticker}")
    history = df.tail(lookback).reset_index(drop=True)

    if st.button("🚀 Run Forecast", type="primary"):
        predictor, max_ctx = load_predictor(model_choice, device)
        if len(history) > max_ctx:
            history = history.tail(max_ctx).reset_index(drop=True)

        last_time = history['timestamps'].iloc[-1]
        time_diff = history['timestamps'].diff().median()
        future_ts = pd.Series([last_time + (i + 1) * time_diff for i in range(pred_len)])

        x_df = history[['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_ts = history['timestamps']

        sample_predictions = []
        bar = st.progress(0)
        for i in range(sample_count):
            pred = predictor.predict(
                df=x_df, x_timestamp=x_ts, y_timestamp=future_ts,
                pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1
            )
            sample_predictions.append(pred['close'].values)
            bar.progress((i + 1) / sample_count)
        bar.empty()

        close_paths = np.column_stack(sample_predictions)
        mean_forecast = close_paths.mean(axis=1)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=history['timestamps'], open=history['open'], high=history['high'],
            low=history['low'], close=history['close'], name="Historical"
        ))
        for idx in range(close_paths.shape[1]):
            fig.add_trace(go.Scatter(
                x=future_ts, y=close_paths[:, idx], mode='lines',
                line=dict(width=1, color='rgba(255, 165, 0, 0.2)'), showlegend=False
            ))
        fig.add_trace(go.Scatter(
            x=future_ts, y=mean_forecast, mode='lines+markers',
            line=dict(color='orange', width=3), name="Forecast Mean"
        ))
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Enter a valid ticker symbol.")
