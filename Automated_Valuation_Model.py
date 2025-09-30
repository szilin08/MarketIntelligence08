import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

st.title("🏡 Automated Valuation Model (AVM)")

# --- Load Data ---
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, sheet_name="Open Transaction Data")
    df.columns = [c.strip() for c in df.columns]

    # Map possible column names
    candidates = {
        "Transaction Date": ["Transaction Date", "Month, Year of Transaction Date", "Transaction Date Serial"],
        "Transaction Price": ["Transaction Price", "Price", "Amount", "Consideration"],
        "District": ["District", "DISTRICT", "Daerah"],
        "Mukim": ["Mukim", "MUKIM"],
        "Property Type": ["Property Type", "PROPERTY TYPE", "Type"],
        "Scheme Name/Area": ["Scheme Name/Area", "Scheme Name", "Scheme", "Area"]
    }

    col_map = {}
    for std, opts in candidates.items():
        for o in opts:
            if o in df.columns:
                col_map[std] = o
                break

    # Fallbacks
    if "Transaction Price" not in col_map:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col_map["Transaction Price"] = num_cols[-1]

    if "Transaction Date" not in col_map:
        st.error(f"No date column found. Available: {df.columns.tolist()}")
        st.stop()

    # Rename
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Format
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Transaction Price"] = pd.to_numeric(df["Transaction Price"], errors="coerce")

    for cat in ["District", "Mukim", "Property Type", "Scheme Name/Area"]:
        if cat in df.columns:
            df[cat] = df[cat].astype(str).str.strip().str.title()

    df = df.dropna(subset=["Transaction Date", "Transaction Price"])
    df["Year"] = df["Transaction Date"].dt.year
    df["Month"] = df["Transaction Date"].dt.month

    return df

# Path
data_path = r"C:\Users\steffiephang\OneDrive - LBS Bina Holdings Sdn Bhd\Desktop\Steffie\ADHD_Project\AVM 2\Open Transaction Data.xlsx"
df = load_data(data_path)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

district_sel = st.sidebar.multiselect("District", df["District"].unique()) if "District" in df.columns else []
mukim_sel = st.sidebar.multiselect("Mukim", df["Mukim"].unique()) if "Mukim" in df.columns else []
ptype_sel = st.sidebar.multiselect("Property Type", df["Property Type"].unique()) if "Property Type" in df.columns else []

# Apply filters
filtered_df = df.copy()
if district_sel:
    filtered_df = filtered_df[filtered_df["District"].isin(district_sel)]
if mukim_sel:
    filtered_df = filtered_df[filtered_df["Mukim"].isin(mukim_sel)]
if ptype_sel:
    filtered_df = filtered_df[filtered_df["Property Type"].isin(ptype_sel)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --- Aggregate by month (USE FILTERED DATA) ---
ts = filtered_df.groupby(pd.Grouper(key="Transaction Date", freq="M"))["Transaction Price"].mean().dropna()

# --- Historical Chart ---
st.subheader("📈 Historical Monthly Average Prices")
st.line_chart(ts)

# --- Holt-Winters Forecast ---
periods = st.slider("Forecast horizon (months)", 6, 24, 12)

model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()
forecast = fit.forecast(periods)

# Plot forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Actual"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast"))

# Confidence band
upper = forecast * 1.1
lower = forecast * 0.9
fig.add_trace(go.Scatter(x=forecast.index, y=upper, mode="lines", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=forecast.index, y=lower, mode="lines", line=dict(width=0), fill="tonexty", name="Confidence Band"))

fig.update_layout(title="Forecasted Property Prices", yaxis_title="Price (RM)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Simple Valuation ---
st.subheader("💰 Property Valuation")
latest_price = ts.iloc[-1]
valuation = forecast.iloc[-1]

st.metric(label="Last Observed Avg Price", value=f"RM {latest_price:,.0f}")
st.metric(label="Forecasted Value (Next Period)", value=f"RM {valuation:,.0f}")

# --- Backtesting ---
st.subheader("🔍 Backtesting Performance")
if len(ts) > 24:
    train = ts.iloc[:-12]
    test = ts.iloc[-12:]

    fit_bt = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    pred_bt = fit_bt.forecast(12)

    mape = mean_absolute_percentage_error(test, pred_bt)
    st.write(f"Backtest MAPE (last 12 months): **{mape:.2%}**")

    bt_fig = go.Figure()
    bt_fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines", name="Train"))
    bt_fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Test (Actual)"))
    bt_fig.add_trace(go.Scatter(x=pred_bt.index, y=pred_bt.values, mode="lines+markers", name="Predicted"))
    bt_fig.update_layout(title="Backtesting Forecast vs Actual", yaxis_title="Price (RM)", template="plotly_white")
    st.plotly_chart(bt_fig, use_container_width=True)
else:
    st.info("Not enough data for backtesting (need > 24 months).")
