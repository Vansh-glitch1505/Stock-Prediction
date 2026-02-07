import streamlit as st
from predict import predict_stock
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock Signal Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .title-text {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .subtitle-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* Signal badges */
    .signal-badge {
        font-size: 2.5rem;
        font-weight: 700;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Info box */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">📈 Stock Signal Predictor</h1>
        <p class="subtitle-text">XGBoost • Technical Indicators • 5-Day Forecast</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# INFO SECTION
# =========================
with st.expander("ℹ️ How It Works", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Data Collection**")
        st.write("Fetches historical stock data and calculates technical indicators")
    
    with col2:
        st.markdown("**🤖 ML Prediction**")
        st.write("Uses XGBoost model to predict price movements")
    
    with col3:
        st.markdown("**📈 Signal Generation**")
        st.write("Generates BUY/SELL/HOLD signals based on predictions")

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    symbol = st.text_input(
        "🔍 Enter Stock Symbol",
        value="RELIANCE.NS",
        help="Example: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT",
        placeholder="Enter ticker symbol..."
    )
    
    predict_button = st.button(
        "🚀 Predict Next 5 Trading Days",
        use_container_width=True
    )

# =========================
# PREDICTION LOGIC
# =========================
if predict_button:
    if not symbol.strip():
        st.error("⚠️ Please enter a valid stock symbol")
    else:
        with st.spinner("🔄 Fetching data & running ML model..."):
            try:
                result = predict_stock(symbol.upper().strip())
                
                signal = result["signal"]
                recent = result["recent"]
                
                # =========================
                # SIGNAL DISPLAY
                # =========================
                st.markdown("<br>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    if signal == "BUY":
                        st.markdown(f"""
                            <div class="signal-badge signal-buy">
                                🟢 SIGNAL: {signal}
                            </div>
                        """, unsafe_allow_html=True)
                        st.success("✅ Model suggests a buying opportunity", icon="📈")
                    elif signal == "SELL":
                        st.markdown(f"""
                            <div class="signal-badge signal-sell">
                                🔴 SIGNAL: {signal}
                            </div>
                        """, unsafe_allow_html=True)
                        st.error("⚠️ Model suggests selling or avoiding", icon="📉")
                    else:
                        st.markdown(f"""
                            <div class="signal-badge signal-hold">
                                🟡 SIGNAL: {signal}
                            </div>
                        """, unsafe_allow_html=True)
                        st.warning("⏸️ Model suggests holding current position", icon="⏳")
                
                # =========================
                # METRICS OVERVIEW
                # =========================
                st.markdown("<br><h2 class='section-header'>📊 Key Metrics</h2>", unsafe_allow_html=True)
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    latest_price = recent["Close"].iloc[-1]
                    st.metric(
                        label="Latest Close",
                        value=f"₹{latest_price:.2f}",
                        delta=None
                    )
                
                with metric_col2:
                    price_change = recent["Close"].iloc[-1] - recent["Close"].iloc[-2]
                    price_change_pct = (price_change / recent["Close"].iloc[-2]) * 100
                    st.metric(
                        label="Daily Change",
                        value=f"₹{abs(price_change):.2f}",
                        delta=f"{price_change_pct:.2f}%"
                    )
                
                with metric_col3:
                    high_30d = recent["Close"].max()
                    st.metric(
                        label="30-Day High",
                        value=f"₹{high_30d:.2f}"
                    )
                
                with metric_col4:
                    low_30d = recent["Close"].min()
                    st.metric(
                        label="30-Day Low",
                        value=f"₹{low_30d:.2f}"
                    )
                
                # =========================
                # PRICE CHART
                # =========================
                st.markdown("<br><h2 class='section-header'>📈 Price Trend (Last 30 Days)</h2>", unsafe_allow_html=True)
                
                # Create chart data with date index
                chart_data = recent.set_index(recent.index)["Close"]
                st.line_chart(chart_data, use_container_width=True, height=400)
                
                # =========================
                # DATA TABLE
                # =========================
                st.markdown("<br><h2 class='section-header'>📋 Detailed Predictions</h2>", unsafe_allow_html=True)
                
                # Format the dataframe for better display
                display_df = recent.copy()
                if "Close" in display_df.columns:
                    display_df["Close"] = display_df["Close"].apply(lambda x: f"₹{x:.2f}")
                if "Predicted" in display_df.columns:
                    display_df["Predicted"] = display_df["Predicted"].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # =========================
                # DISCLAIMER
                # =========================
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="info-box">
                        <strong>⚠️ Disclaimer:</strong> This is an educational tool only. 
                        Predictions are based on historical data and may not reflect future performance. 
                        Always do your own research and consult with financial advisors before making investment decisions.
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure the stock symbol is correct and data is available")

# =========================
# FOOTER
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("🛠️ Built with Streamlit + XGBoost")

with footer_col2:
    st.caption("📚 For educational purposes only")

with footer_col3:
    st.caption("💡 Not financial advice")