import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Stock AI Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================
# STYLE (DIFFERENT LOOK)
# =========================
st.markdown("""
<style>

body {
    background-color: #0b1220;
}

.stApp {
    background: radial-gradient(circle at top, #1e293b, #0b1220);
}

/* Header */
.header {
    text-align:center;
    padding:30px;
    color:white;
}

.header h1 {
    font-size:48px;
    font-weight:800;
}

.header p {
    color:#94a3b8;
    font-size:18px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding:20px;
    border-radius:16px;
    text-align:center;
    backdrop-filter: blur(10px);
}

.card h2 {
    color:#38bdf8;
    font-size:26px;
}

.card span {
    color:#cbd5e1;
}

/* Section */
.section {
    margin-top:30px;
    font-size:22px;
    color:white;
    font-weight:600;
}

/* Table */
[data-testid="stDataFrame"] {
    border-radius:12px;
    overflow:hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL + DATA
# =========================
model = pickle.load(open("apple_stock_model.pkl", "rb"))
df = pd.read_csv("apple_stock_data.csv")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <h1>📊 AI Stock Forecast System</h1>
    <p>Predicting Apple Stock Prices using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================
# FORECAST LOGIC
# =========================
history = list(df["Close"])
future = []

for i in range(30):

    X_input = pd.DataFrame({
        "Lag_1": [history[-1]],
        "Lag_2": [history[-2]],
        "Lag_3": [history[-3]],
        "Lag_7": [history[-7]],
        "MA_7": [np.mean(history[-7:])],
        "MA_30": [np.mean(history[-30:])]
    })

    pred = model.predict(X_input)[0]
    future.append(pred)
    history.append(pred)

forecast_df = pd.DataFrame({
    "Day": range(1, 31),
    "Prediction": future
})

# =========================
# KPI SECTION (NEW STYLE)
# =========================
st.markdown("### 📌 Key Insights")

c1, c2, c3 = st.columns(3)

c1.markdown(f"""
<div class="card">
<h2>${df['Close'].iloc[-1]:.2f}</h2>
<span>Last Close Price</span>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
<h2>${forecast_df['Prediction'].max():.2f}</h2>
<span>Max Forecast</span>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
<h2>${forecast_df['Prediction'].min():.2f}</h2>
<span>Min Forecast</span>
</div>
""", unsafe_allow_html=True)

# =========================
# HISTORICAL
# =========================
st.markdown('<div class="section">📈 Market History</div>', unsafe_allow_html=True)

fig1 = px.line(df, y="Close")
fig1.update_layout(
    template="plotly_dark",
    height=450
)

st.plotly_chart(fig1, use_container_width=True)

# =========================
# FORECAST
# =========================
st.markdown('<div class="section">🚀 Future Prediction (30 Days)</div>', unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=forecast_df["Day"],
    y=forecast_df["Prediction"],
    mode="lines+markers",
    line=dict(color="#38bdf8", width=3)
))

fig2.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# TABLE
# =========================
st.markdown("### 📊 Forecast Data")

st.dataframe(forecast_df, use_container_width=True)

# =========================
# DOWNLOAD
# =========================
csv = forecast_df.to_csv(index=False)

st.download_button(
    "⬇ Download Forecast Data",
    csv,
    "forecast.csv",
    "text/csv"
)