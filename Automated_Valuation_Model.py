# avm.py (Streamlit version)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
MIN_HISTORY = 36  # 3 years
MIN_TRANSACTIONS = 180
FORECAST_MONTHS = 24
OUTLIER_THRESHOLD = 0.02


# ---------------- CLEANING ----------------
def clean_data(df):
    df['date'] = pd.to_datetime(df['Month, Year of Transaction Date'],
                                format='%b-%y', errors='coerce')
    df = df.dropna(subset=['date', 'Price Per Sq Ft',
                           'Property Type', 'District'])
    lower = df['Price Per Sq Ft'].quantile(OUTLIER_THRESHOLD)
    upper = df['Price Per Sq Ft'].quantile(1 - OUTLIER_THRESHOLD)
    return df[(df['Price Per Sq Ft'] > lower) & (df['Price Per Sq Ft'] < upper)]


def get_valid_groups(df):
    grouped = df.groupby(['Property Type', 'District']).agg(
        transaction_count=('Price Per Sq Ft', 'size'),
        date_range=('date', lambda x: x.nunique())
    )
    return grouped[(grouped['date_range'] >= MIN_HISTORY) &
                   (grouped['transaction_count'] >= MIN_TRANSACTIONS)].reset_index()


# ---------------- FORECASTING ----------------
def process_group(df, prop_type, district):
    try:
        subset = df[(df['Property Type'] == prop_type) &
                    (df['District'] == district)]
        ts_data = subset.groupby(pd.Grouper(key='date', freq='ME'))[
            'Price Per Sq Ft'].mean().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data = ts_data.dropna()

        if len(ts_data) < MIN_HISTORY:
            return None

        train = ts_data.iloc[:-12]
        test = ts_data.iloc[-12:]

        # ARIMA model
        arima_model = ARIMA(train['y'], order=(2, 1, 2)).fit()
        arima_pred = arima_model.forecast(steps=12)
        arima_mape = mean_absolute_percentage_error(
            test['y'], arima_pred) * 100

        # Prophet model
        prophet_model = Prophet(yearly_seasonality=True).fit(train)
        prophet_pred = prophet_model.predict(
            prophet_model.make_future_dataframe(periods=12, freq='M')
        ).iloc[-12:]['yhat']
        prophet_mape = mean_absolute_percentage_error(
            test['y'], prophet_pred) * 100

        # Use model if within accuracy threshold
        if arima_mape <= 15 or prophet_mape <= 25:
            full_model = ARIMA(ts_data['y'], order=(2, 1, 2)).fit()
            forecast = full_model.get_forecast(steps=FORECAST_MONTHS)

            # Fit Prophet on full data
            prophet_model_full = Prophet(yearly_seasonality=True).fit(ts_data)
            future_prophet = prophet_model_full.make_future_dataframe(
                periods=FORECAST_MONTHS, freq='M')
            forecast_prophet = prophet_model_full.predict(
                future_prophet).iloc[-FORECAST_MONTHS:]

            return {
                'Property Type': prop_type,
                'District': district,
                'ARIMA_MAPE': arima_mape,
                'Prophet_MAPE': prophet_mape,
                'ARIMA_Forecast': forecast.predicted_mean,
                'ARIMA_Lower_CI': forecast.conf_int().iloc[:, 0],
                'ARIMA_Upper_CI': forecast.conf_int().iloc[:, 1],
                'Prophet_Forecast': forecast_prophet['yhat'].values,
                'Prophet_Lower_CI': forecast_prophet['yhat_lower'].values,
                'Prophet_Upper_CI': forecast_prophet['yhat_upper'].values,
                'Dates': pd.date_range(
                    start=ts_data['ds'].max() + pd.DateOffset(months=1),
                    periods=FORECAST_MONTHS,
                    freq='ME'
                ),
            }
        return None
    except Exception:
        return None


# ---------------- STREAMLIT APP ----------------
def run():
    st.header("🏠 Automated Valuation Model")

    uploaded_file = st.file_uploader("Upload transaction CSV", type="csv")
    if uploaded_file is None:
        st.info("Please upload your transaction dataset (CSV).")
        return

    df = pd.read_csv(
        uploaded_file,
        encoding='latin-1',
        converters={'Price Per Sq Ft': lambda x: float(
            str(x).replace('RM', '').replace(',', '').strip())},
        low_memory=False
    )
    df = clean_data(df)
    valid_groups = get_valid_groups(df)

    if valid_groups.empty:
        st.warning("No valid property groups found with enough history.")
        return

    # Dropdowns
    prop_type = st.selectbox("Property Type", valid_groups['Property Type'].unique())
    district = st.selectbox("District", valid_groups['District'].unique())

    st.write("Generating forecast...")
    result = process_group(df, prop_type, district)

    if not result:
        st.error("Not enough data for reliable forecast.")
        return

    forecast_df = pd.DataFrame({
        'Date': result['Dates'],
        'ARIMA_Forecast': result['ARIMA_Forecast'],
        'ARIMA_Lower_CI': result['ARIMA_Lower_CI'],
        'ARIMA_Upper_CI': result['ARIMA_Upper_CI'],
        'Prophet_Forecast': result['Prophet_Forecast'],
        'Prophet_Lower_CI': result['Prophet_Lower_CI'],
        'Prophet_Upper_CI': result['Prophet_Upper_CI'],
    })

    # Plot results
    actual_df = df[(df['Property Type'] == prop_type) &
                   (df['District'] == district)].groupby('date')[
        'Price Per Sq Ft'].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df['date'], y=actual_df['Price Per Sq Ft'],
                             mode='lines', name='Actual', line=dict(color='#4A90E2')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['ARIMA_Forecast'],
                             mode='lines', name='ARIMA Forecast', line=dict(color='#F5A623')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Prophet_Forecast'],
                             mode='lines', name='Prophet Forecast', line=dict(color='#2E7D32')))
    fig.update_layout(title=f"Forecast for {prop_type} in {district}",
                      xaxis_title="Date",
                      yaxis_title="Price per Sq Ft (RM)",
                      template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # Accuracy report
    st.subheader("Model Accuracy")
    st.write(pd.DataFrame({
        "ARIMA_MAPE%": [result['ARIMA_MAPE']],
        "Prophet_MAPE%": [result['Prophet_MAPE']]
    }))


if __name__ == "__main__":
    run()
