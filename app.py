import os
import sys
import subprocess
import streamlit as st

# -------------------------------------------------------------------
# 1. Dynamically Clone Kronos Foundation Model Repo
# -------------------------------------------------------------------
KRONOS_REPO_URL = "https://github.com/shiyu-coder/Kronos.git"
REPO_DIR = "Kronos_Source"

if not os.path.exists(REPO_DIR):
    with st.spinner("Cloning Kronos AI Foundation Engine from GitHub..."):
        subprocess.run(["git", "clone", KRONOS_REPO_URL, REPO_DIR], check=True)

if os.path.abspath(REPO_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(REPO_DIR))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError as e:
    st.error(f"Failed to load Kronos modules: {e}")
    st.stop()

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import torch
import json
from openai import OpenAI

# -------------------------------------------------------------------
# 2. Page Configuration & TradingView Theme Injection
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Kronos Terminal | TradingView AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom TradingView CSS Theme
st.markdown("""
<style>
    /* TradingView Dark Color Palette */
    :root {
        --tv-bg: #131722;
        --tv-panel: #1e222d;
        --tv-border: #2a2e39;
        --tv-text: #d1d4dc;
        --tv-text-muted: #787b86;
        --tv-blue: #2962ff;
        --tv-green: #089981;
        --tv-red: #f23645;
    }

    .stApp {
        background-color: var(--tv-bg);
        color: var(--tv-text);
        font-family: -apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif;
    }
    
    /* Header & Panel styling */
    header, .stHeader { background-color: var(--tv-bg) !important; }
    div[data-testid="stSidebar"] {
        background-color: var(--tv-panel);
        border-right: 1px solid var(--tv-border);
    }
    
    /* TradingView Card Containers */
    .tv-card {
        background-color: var(--tv-panel);
        border: 1px solid var(--tv-border);
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .tv-metric-title {
        color: var(--tv-text-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tv-metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 4px;
    }
    .tv-badge-up {
        color: var(--tv-green);
        font-weight: 600;
    }
    .tv-badge-down {
        color: var(--tv-red);
        font-weight: 600;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: var(--tv-blue) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1e53e5 !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--tv-panel);
        border-bottom: 1px solid var(--tv-border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--tv-text-muted);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--tv-blue) !important;
        border-bottom-color: var(--tv-blue) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 3. Model Loader (Cached)
# -------------------------------------------------------------------
MODEL_MAP = {
    "Kronos-mini (4.1M params | 2048 ctx)": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "Kronos-small (24.7M params | 512 ctx)": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "Kronos-base (102.3M params | 512 ctx)": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

@st.cache_resource(show_spinner="Loading Kronos Transformer Engine...")
def load_kronos(model_key, device_type):
    cfg = MODEL_MAP[model_key]
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
    model = Kronos.from_pretrained(cfg["model_id"])
    model.to(device_type)
    return KronosPredictor(model, tokenizer, max_context=cfg["max_context"]), cfg["max_context"]

# -------------------------------------------------------------------
# 4. Technical Indicators Engine
# -------------------------------------------------------------------
def compute_technical_indicators(df):
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['SMA_200'] = data['close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI 14)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['close'].ewm(span=12, adjust=False).mean()
    ema26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    std20 = data['close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (std20 * 2)
    data['BB_Lower'] = data['SMA_20'] - (std20 * 2)
    
    return data

# -------------------------------------------------------------------
# 5. Sidebar & Asset Selection
# -------------------------------------------------------------------
st.sidebar.markdown("### 🛠️ TradingView AI Terminal")

asset_category = st.sidebar.selectbox(
    "Asset Class",
    ["Stocks & Equities", "ETFs & Funds", "Bonds & Treasury Yields", "Crypto", "Commodities"]
)

default_tickers = {
    "Stocks & Equities": "AAPL",
    "ETFs & Funds": "SPY",
    "Bonds & Treasury Yields": "^TNX",
    "Crypto": "BTC-USD",
    "Commodities": "GC=F"
}

ticker_input = st.sidebar.text_input("Ticker Symbol", value=default_tickers[asset_category]).upper()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Kronos AI Config")
model_choice = st.sidebar.selectbox("Kronos Variant", list(MODEL_MAP.keys()), index=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Hardware Compute: **{device.upper()}**")

interval = st.sidebar.selectbox("Timeframe / Interval", ["1d", "1h", "15m", "5m"], index=0)
period = st.sidebar.selectbox("Historical Lookback Period", ["60d", "100d", "1y", "2y", "5y"], index=2)

pred_len = st.sidebar.slider("Forecast Horizon (Steps)", 5, 120, 30, 5)
lookback_window = st.sidebar.slider("Context Window", 50, 512, 300, 10)
sample_count = st.sidebar.slider("Monte Carlo Paths", 1, 25, 10, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 OpenAI Reasoning Agent")
openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Enter key to enable text-based market explanations.")

# -------------------------------------------------------------------
# 6. Fetch Asset Data & Full Fundamentals
# -------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_asset_full_data(symbol, interval, period):
    ticker_obj = yf.Ticker(symbol)
    
    # Fetch historical data
    df = ticker_obj.history(period=period, interval=interval)
    if df.empty:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
    
    if df.empty:
        return None, None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    date_col = 'Date' if 'Date' in df.columns else 'Datetime'
    df = df.rename(columns={
        date_col: 'timestamps',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    if 'volume' not in df.columns or df['volume'].isna().all():
        df['volume'] = 0.0
    df['amount'] = df['close'] * df['volume']
    
    # Fetch trading view style fundamentals info
    info = {}
    try:
        info = ticker_obj.info
    except Exception:
        info = {}
        
    return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']], info

df_raw, asset_info = fetch_asset_full_data(ticker_input, interval, period)

# -------------------------------------------------------------------
# 7. Main Dashboard Execution
# -------------------------------------------------------------------
if df_raw is not None and not df_raw.empty:
    df_tech = compute_technical_indicators(df_raw)
    latest = df_tech.iloc[-1]
    prev = df_tech.iloc[-2] if len(df_tech) > 1 else latest
    
    price_change = latest['close'] - prev['close']
    pct_change = (price_change / prev['close']) * 100
    badge_class = "tv-badge-up" if price_change >= 0 else "tv-badge-down"
    
    # TradingView Header Bar
    asset_name = asset_info.get('longName', asset_info.get('shortName', ticker_input))
    sector_str = asset_info.get('sector', asset_info.get('quoteType', 'N/A'))
    industry_str = asset_info.get('industry', 'N/A')
    
    st.markdown(f"""
    <div class="tv-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 26px; font-weight: 800; color: #ffffff;">{ticker_input}</span>
                <span style="font-size: 16px; color: var(--tv-text-muted); margin-left: 10px;">{asset_name}</span>
                <div style="font-size: 12px; color: var(--tv-text-muted); margin-top: 2px;">
                    {sector_str} • {industry_str} • Currency: {asset_info.get('currency', 'USD')}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 28px; font-weight: 800; color: #ffffff;">${latest['close']:,.2f}</div>
                <div class="{badge_class}" style="font-size: 15px;">
                    {price_change:+.2f} ({pct_change:+.2f}%)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TradingView Workspace Tabs
    tab_chart, tab_fundamentals, tab_technicals, tab_data = st.tabs([
        "📈 Trading Chart & Kronos AI",
        "🏢 TradingView Fundamentals",
        "📊 Technical Indicators",
        "📋 Historical & Export Data"
    ])

    # -------------------------------------------------------------------
    # TAB 1: Trading Chart & Kronos AI Prediction
    # -------------------------------------------------------------------
    with tab_chart:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("#### 🚀 Execute AI Forecast")
        with c2:
            run_btn = st.button("Generate Kronos Forecast Paths")

        # Initialize chart state
        actual_lookback = min(lookback_window, len(df_tech))
        history_df = df_tech.tail(actual_lookback).reset_index(drop=True)

        if run_btn:
            predictor, max_ctx = load_kronos(model_choice, device)
            if len(history_df) > max_ctx:
                history_df = history_df.tail(max_ctx).reset_index(drop=True)

            last_time = history_df['timestamps'].iloc[-1]
            time_diff = history_df['timestamps'].diff().median()
            future_ts = pd.Series([last_time + (i + 1) * time_diff for i in range(pred_len)])

            x_df = history_df[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_ts = history_df['timestamps']

            sample_preds = []
            progress_bar = st.progress(0)

            for i in range(sample_count):
                pred_sample = predictor.predict(
                    df=x_df, x_timestamp=x_ts, y_timestamp=future_ts,
                    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1
                )
                sample_preds.append(pred_sample['close'].values)
                progress_bar.progress((i + 1) / sample_count)
            progress_bar.empty()

            close_paths = np.column_stack(sample_preds)
            mean_forecast = close_paths.mean(axis=1)
            upper_bound = np.percentile(close_paths, 95, axis=1)
            lower_bound = np.percentile(close_paths, 5, axis=1)

            st.session_state['kronos_results'] = {
                'future_ts': future_ts,
                'close_paths': close_paths,
                'mean_forecast': mean_forecast,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'pred_change_pct': ((mean_forecast[-1] - history_df['close'].iloc[-1]) / history_df['close'].iloc[-1]) * 100
            }

        # Build Plotly Candlestick Chart (TradingView Dark Theme)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=history_df['timestamps'],
            open=history_df['open'], high=history_df['high'],
            low=history_df['low'], close=history_df['close'],
            name="OHLC",
            increasing_line_color='#089981', decreasing_line_color='#f23645',
            increasing_fillcolor='#089981', decreasing_fillcolor='#f23645'
        ), row=1, col=1)

        # SMAs
        fig.add_trace(go.Scatter(x=history_df['timestamps'], y=history_df['SMA_20'], line=dict(color='#2962ff', width=1), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=history_df['timestamps'], y=history_df['SMA_50'], line=dict(color='#ff9800', width=1), name="SMA 50"), row=1, col=1)

        # Volume Bar Chart
        colors = ['#089981' if row['close'] >= row['open'] else '#f23645' for _, row in history_df.iterrows()]
        fig.add_trace(go.Bar(
            x=history_df['timestamps'], y=history_df['volume'],
            marker_color=colors, opacity=0.5, name="Volume"
        ), row=2, col=1)

        # Overlay Forecast if available
        if 'kronos_results' in st.session_state:
            res = st.session_state['kronos_results']
            
            # Paths
            for idx in range(res['close_paths'].shape[1]):
                fig.add_trace(go.Scatter(
                    x=res['future_ts'], y=res['close_paths'][:, idx],
                    mode='lines', line=dict(width=1, color='rgba(255, 165, 0, 0.25)'),
                    showlegend=False, hoverinfo='skip'
                ), row=1, col=1)

            # Confidence Band
            fig.add_trace(go.Scatter(
                x=list(res['future_ts']) + list(res['future_ts'])[::-1],
                y=list(res['upper_bound']) + list(res['lower_bound'])[::-1],
                fill='toself', fillcolor='rgba(255, 165, 0, 0.12)',
                line=dict(color='rgba(255,255,255,0)'), name="90% Confidence Band", hoverinfo='skip'
            ), row=1, col=1)

            # Mean Forecast
            fig.add_trace(go.Scatter(
                x=res['future_ts'], y=res['mean_forecast'],
                mode='lines+markers', line=dict(color='#ff9800', width=3),
                name="Kronos Forecast Mean"
            ), row=1, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131722",
            plot_bgcolor="#1e222d",
            height=650,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # OpenAI AI Reasoning Explanation Module
        if 'kronos_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 🧠 AI Market Reasoning Explanation")
            
            res = st.session_state['kronos_results']
            
            if openai_api_key:
                if st.button("Generate OpenAI Executive Market Explanation"):
                    with st.spinner("Analyzing technical pattern & generating explanation..."):
                        try:
                            client = OpenAI(api_key=openai_api_key)
                            
                            prompt = f"""
                            You are a senior quantitative financial analyst at TradingView. 
                            Analyze why the Kronos Time-Series AI foundation model predicted the following trajectory for {ticker_input}:

                            Historical Technical Indicators:
                            - Current Close: ${latest['close']:.2f}
                            - 14-period RSI: {latest['RSI_14']:.2f}
                            - MACD: {latest['MACD']:.2f} (Signal: {latest['MACD_Signal']:.2f})
                            - SMA 20: ${latest['SMA_20']:.2f} | SMA 50: ${latest['SMA_50']:.2f}
                            
                            Kronos AI Prediction Metrics:
                            - Horizon: {pred_len} steps
                            - Forecast Target Mean: ${res['mean_forecast'][-1]:.2f}
                            - Predicted Return: {res['pred_change_pct']:+.2f}%
                            - 95th Percentile Bullish Target: ${res['upper_bound'][-1]:.2f}
                            - 5th Percentile Bearish Target: ${res['lower_bound'][-1]:.2f}

                            Provide a concise 3-bullet point executive market breakdown explaining the potential price drivers, chart pattern setups (momentum, mean reversion, trend continuation), and risk factors that justify this trajectory.
                            """

                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7
                            )
                            
                            st.markdown(f"""
                            <div class="tv-card">
                                <h4 style="color: var(--tv-blue); margin-top: 0;">Analyst Explanation</h4>
                                {response.choices[0].message.content}
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as err:
                            st.error(f"OpenAI API Error: {err}")
            else:
                st.info("💡 Tip: Enter an OpenAI API key in the sidebar to generate an automated executive explanation for this forecast.")

    # -------------------------------------------------------------------
    # TAB 2: TradingView Fundamentals & Asset Profile
    # -------------------------------------------------------------------
    with tab_fundamentals:
        st.markdown("### 🏢 Complete TradingView Asset Profile")
        
        # Profile Summary
        summary = asset_info.get('longBusinessSummary', 'No detailed business profile available for this ticker/symbol.')
        st.markdown(f"""
        <div class="tv-card">
            <h5 style="color: var(--tv-text-muted);">Business Profile</h5>
            <p style="font-size: 14px; line-height: 1.6; color: var(--tv-text);">{summary}</p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown(f"""
            <div class="tv-card">
                <div class="tv-metric-title">Market Capitalization</div>
                <div class="tv-metric-value">${asset_info.get('marketCap', 0):,}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Trailing P/E</div>
                <div class="tv-metric-value">{asset_info.get('trailingPE', 'N/A')}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Forward P/E</div>
                <div class="tv-metric-value">{asset_info.get('forwardPE', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="tv-card">
                <div class="tv-metric-title">52-Week Range</div>
                <div class="tv-metric-value">${asset_info.get('fiftyTwoWeekLow', 0):,.2f} - ${asset_info.get('fiftyTwoWeekHigh', 0):,.2f}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Price to Book (P/B)</div>
                <div class="tv-metric-value">{asset_info.get('priceToBook', 'N/A')}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">PEG Ratio</div>
                <div class="tv-metric-value">{asset_info.get('pegRatio', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown(f"""
            <div class="tv-card">
                <div class="tv-metric-title">Dividend Yield</div>
                <div class="tv-metric-value">{asset_info.get('dividendYield', 0)*100 if asset_info.get('dividendYield') else 'N/A'}%</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Profit Margins</div>
                <div class="tv-metric-value">{asset_info.get('profitMargins', 0)*100 if asset_info.get('profitMargins') else 'N/A'}%</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Beta (Volatily)</div>
                <div class="tv-metric-value">{asset_info.get('beta', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown(f"""
            <div class="tv-card">
                <div class="tv-metric-title">Analyst Consensus Target</div>
                <div class="tv-metric-value">${asset_info.get('targetMeanPrice', 0):,.2f}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Analyst Recommendation</div>
                <div class="tv-metric-value">{str(asset_info.get('recommendationKey', 'N/A')).upper()}</div>
            </div>
            <div class="tv-card">
                <div class="tv-metric-title">Enterprise Value</div>
                <div class="tv-metric-value">${asset_info.get('enterpriseValue', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------
    # TAB 3: Technical Indicators Overview
    # -------------------------------------------------------------------
    with tab_technicals:
        st.markdown("### 📊 TradingView Oscillators & Moving Averages")

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("RSI (14)", f"{latest['RSI_14']:.2f}")
        t2.metric("MACD Line", f"{latest['MACD']:.2f}")
        t3.metric("SMA 20", f"${latest['SMA_20']:.2f}")
        t4.metric("SMA 200", f"${latest['SMA_200']:.2f}")

        # Technical Oscillators Chart
        fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
        
        # RSI
        fig_tech.add_trace(go.Scatter(x=df_tech['timestamps'], y=df_tech['RSI_14'], line=dict(color='#ab47bc', width=1.5), name="RSI 14"), row=1, col=1)
        fig_tech.add_hline(y=70, line_dash="dash", line_color="#f23645", row=1, col=1)
        fig_tech.add_hline(y=30, line_dash="dash", line_color="#089981", row=1, col=1)

        # MACD
        fig_tech.add_trace(go.Scatter(x=df_tech['timestamps'], y=df_tech['MACD'], line=dict(color='#2962ff', width=1.5), name="MACD"), row=2, col=1)
        fig_tech.add_trace(go.Scatter(x=df_tech['timestamps'], y=df_tech['MACD_Signal'], line=dict(color='#ff9800', width=1.5), name="Signal"), row=2, col=1)

        fig_tech.update_layout(
            template="plotly_dark", paper_bgcolor="#131722", plot_bgcolor="#1e222d", height=450,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_tech, use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 4: Historical & Export Data
    # -------------------------------------------------------------------
    with tab_data:
        st.markdown("### 📋 Historical Market Data Table")
        st.dataframe(df_tech, use_container_width=True)
        
        csv_bytes = df_tech.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Market CSV",
            data=csv_bytes,
            file_name=f"{ticker_input}_market_data.csv",
            mime="text/csv"
        )
else:
    st.error("Invalid Ticker Symbol or no data returned. Please verify the symbol in the sidebar.")
