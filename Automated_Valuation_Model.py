import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

# Define standardize_name function globally
def standardize_name(s):
    if pd.isna(s):
        return s
    return str(s).strip().title()

st.title("Automated Valuation Model")

uploaded_file = st.file_uploader("Upload your Open Transaction Data.xlsx for Valuation", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload the Excel file to start the valuation model.")
    st.stop()

# Load and prepare data for valuation
@st.cache_data  # Cache the data loading to speed up subsequent runs
def load_and_prepare_data(uploaded_file):
    raw = pd.read_excel(uploaded_file, sheet_name="Open Transaction Data")
    raw.columns = [c.strip() for c in raw.columns]
    def deduplicate_columns(columns):
        """Ensure dataframe columns are unique by appending suffixes."""
        seen = {}
        new_cols = []
        for col in columns:
            if col not in seen:
                seen[col] = 0
                new_cols.append(col)
            else:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
        return new_cols
    raw.columns = deduplicate_columns(raw.columns)

    col_map_guess = {}
    candidates = {
        "Property Type": ["Property Type", "Property type", "PROPERTY TYPE"],
        "District": ["District", "DISTRICT"],
        "Mukim": ["Mukim", "Mukim "],
        "Scheme Name/Area": ["Scheme Name/Area", "Scheme Name", "Scheme"],
        "Month, Year of Transaction Date": ["Month, Year of Transaction Date", "Transaction Date", "Transaction Date Serial"],
        "Transaction Price": ["Transaction Price", "TransactionPrice", "Transaction Price "]
    }
    for std, opts in candidates.items():
        for o in opts:
            if o in raw.columns:
                col_map_guess[std] = o
                break

    if "Transaction Price" not in col_map_guess:
        numeric_cols = raw.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col_map_guess["Transaction Price"] = numeric_cols[-1]

    df_val = raw.rename(columns={v: k for k, v in col_map_guess.items()})
    required = ["Property Type", "District", "Mukim", "Scheme Name/Area", "Month, Year of Transaction Date", "Transaction Price"]
    missing = [c for c in required if c not in df_val.columns]
    if missing:
        st.error(f"Required column(s) missing: {missing}. Detected columns: {raw.columns.tolist()}")
        st.stop()

    df_val["Transaction Date"] = pd.to_datetime(df_val["Month, Year of Transaction Date"], errors="coerce")
    df_val["Year"] = df_val["Transaction Date"].dt.year.fillna(0).astype(int)
    df_val["Month"] = df_val["Transaction Date"].dt.month.fillna(0).astype(int)
    df_val["Transaction Price"] = pd.to_numeric(df_val["Transaction Price"], errors="coerce")

    df_val["District"] = df_val["District"].astype(str).apply(standardize_name)
    df_val["Mukim"] = df_val["Mukim"].astype(str).apply(standardize_name)
    df_val["Scheme Name/Area"] = df_val["Scheme Name/Area"].astype(str).apply(lambda x: x.strip() if pd.notna(x) else x)
    df_val["Property Type"] = df_val["Property Type"].astype(str).apply(standardize_name)

    return df_val

df_val = load_and_prepare_data(uploaded_file)

# Filters for valuation
st.sidebar.header("Valuation Filters")
val_districts = st.sidebar.multiselect("District", options=sorted(df_val["District"].unique()), default=[])
val_mukims = st.sidebar.multiselect("Mukim", options=sorted(df_val["Mukim"].unique()), default=[])
val_ptype = st.sidebar.multiselect("Property Type", options=sorted(df_val["Property Type"].unique()), default=[])
val_year_range = st.sidebar.slider("Year range", int(df_val["Year"].min()), int(df_val["Year"].max()), (int(df_val["Year"].min()), int(df_val["Year"].max())))

# Apply valuation filters
@st.cache_data  # Cache the filtered data
def apply_filters(df_val, val_districts, val_mukims, val_ptype, val_year_range):
    val_filtered = df_val.copy()
    if val_districts:
        val_filtered = val_filtered[val_filtered["District"].isin(val_districts)]
    if val_mukims:
        val_filtered = val_filtered[val_filtered["Mukim"].isin(val_mukims)]
    if val_ptype:
        val_filtered = val_filtered[val_filtered["Property Type"].isin(val_ptype)]
    val_filtered = val_filtered[(val_filtered["Year"] >= val_year_range[0]) & (val_filtered["Year"] <= val_year_range[1])]
    return val_filtered

val_filtered = apply_filters(df_val, val_districts, val_mukims, val_ptype, val_year_range)

# Prepare data for Prophet model
if not val_filtered.empty:
    @st.cache_data  # Cache the trend data and model fitting
    def prepare_prophet_data(val_filtered):
        trend_data = val_filtered.groupby(["Year", "Month"])["Transaction Price"].mean().reset_index()
        trend_data = trend_data[trend_data["Year"] > 0].head(100)  # Limit to 100 rows for speed
        trend_data["Date"] = pd.to_datetime(trend_data[["Year", "Month"]].assign(day=1))
        prophet_df = trend_data.rename(columns={"Date": "ds", "Transaction Price": "y"})
        return prophet_df

    prophet_df = prepare_prophet_data(val_filtered)

    @st.cache_data  # Cache the model fitting
    def fit_prophet_model(prophet_df):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        return model

    model = fit_prophet_model(prophet_df)

    @st.cache_data  # Cache the forecast
    def generate_forecast(_model):  # Use _model to avoid hashing
        future = _model.make_future_dataframe(periods=12, freq='M')
        forecast = _model.predict(future)
        return forecast

    forecast = generate_forecast(model)

    # Plot forecast
    fig_forecast = px.line(forecast, x="ds", y="yhat", title="Price Trend Forecast (Next 12 Months)")
    fig_forecast.add_scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="markers", name="Actual")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Input for valuation
    st.subheader("Valuation Input")
    district = st.selectbox("Select District", options=[""] + sorted(val_filtered["District"].unique()))
    mukim = st.selectbox("Select Mukim", options=[""] + sorted(val_filtered[val_filtered["District"] == district]["Mukim"].unique()) if district else [""])
    scheme = st.selectbox("Select Scheme Name/Area", options=[""] + sorted(val_filtered[val_filtered["Mukim"] == mukim]["Scheme Name/Area"].unique()) if mukim else [""])
    ptype = st.selectbox("Select Property Type", options=[""] + sorted(val_filtered["Property Type"].unique()))
    year = st.slider("Select Year", int(val_filtered["Year"].min()), int(val_filtered["Year"].max()), int(val_filtered["Year"].max()))

    if district and mukim and scheme and ptype and year:
        # Simple valuation based on average price in the selected category
        val_data = val_filtered[(val_filtered["District"] == district) & 
                              (val_filtered["Mukim"] == mukim) & 
                              (val_filtered["Scheme Name/Area"] == scheme) & 
                              (val_filtered["Property Type"] == ptype) & 
                              (val_filtered["Year"] == year)]
        if not val_data.empty:
            estimated_value = val_data["Transaction Price"].mean()
            st.success(f"Estimated Property Value: RM {estimated_value:,.0f}")
        else:
            st.warning("No data available for the selected criteria. Try adjusting filters.")