import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px

# ---------------- CONFIG ----------------
MIN_HISTORY = 36  # 3 years
MIN_TRANSACTIONS = 180
FORECAST_MONTHS = 24
OUTLIER_THRESHOLD = 0.02

# ---------------- CLEANING ----------------
def clean_data(df):
    df['date'] = pd.to_datetime(df['Month, Year of Transaction Date'], format='%b-%y', errors='coerce')
    df = df.dropna(subset=['date', 'Price Per Sq Ft', 'Property Type', 'District'])
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

# ---------------- LOAD DATA ----------------
df = pd.read_csv(
    r'C:\Users\steffiephang\OneDrive - LBS Bina Holdings Sdn Bhd\Desktop\Steffie\ADHD_Project\AVM\Open Transaction Data 2021-2024.csv',
    encoding='latin-1',
    parse_dates=['Month, Year of Transaction Date'],
    converters={'Price Per Sq Ft': lambda x: float(x.replace('RM', '').replace(',', '').strip())},
    low_memory=False
)
df = clean_data(df)
valid_groups = get_valid_groups(df)

# ---------------- FORECASTING ----------------
def process_group(prop_type, district):
    try:
        subset = df[(df['Property Type'] == prop_type) & (df['District'] == district)]
        ts_data = subset.groupby(pd.Grouper(key='date', freq='ME'))['Price Per Sq Ft'].mean().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data = ts_data.dropna()

        if len(ts_data) < MIN_HISTORY:
            return None

        train = ts_data.iloc[:-12]
        test = ts_data.iloc[-12:]

        # ARIMA model
        arima_model = ARIMA(train['y'], order=(2,1,2)).fit()
        arima_pred = arima_model.forecast(steps=12)
        arima_mape = mean_absolute_percentage_error(test['y'], arima_pred) * 100

        # Prophet model
        prophet_model = Prophet(yearly_seasonality=True).fit(train)
        prophet_pred = prophet_model.predict(
            prophet_model.make_future_dataframe(periods=12, freq='M')
        ).iloc[-12:]['yhat']
        prophet_mape = mean_absolute_percentage_error(test['y'], prophet_pred) * 100

        # Use model if within accuracy threshold
        if arima_mape <= 15 or prophet_mape <= 25:
            full_model = ARIMA(ts_data['y'], order=(2,1,2)).fit()
            forecast = full_model.get_forecast(steps=FORECAST_MONTHS)

            # Prepare historical test vs predicted for output (ARIMA)
            historical_comparison = pd.DataFrame({
                'Date': test['ds'],
                'Actual': test['y'],
                'Predicted': arima_pred.values,
                'Property Type': prop_type,
                'District': district
            })

            # Fit Prophet on full data
            prophet_model_full = Prophet(yearly_seasonality=True).fit(ts_data)
            future_prophet = prophet_model_full.make_future_dataframe(periods=FORECAST_MONTHS, freq='M')
            forecast_prophet = prophet_model_full.predict(future_prophet).iloc[-FORECAST_MONTHS:]

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
                'Historical_Comparison': historical_comparison
            }
        return None
    except Exception as e:
        return None

# ---------------- RUN FORECASTS ----------------
results = []
for _, row in tqdm(valid_groups.iterrows(), total=len(valid_groups)):
    result = process_group(row['Property Type'], row['District'])
    if result:
        results.append(result)

# ---------------- SAVE RESULTS ----------------
if results:
    forecast_df = pd.DataFrame({
        'Date': [d for res in results for d in res['Dates']],
        'Property Type': [res['Property Type'] for res in results for _ in res['Dates']],
        'District': [res['District'] for res in results for _ in res['Dates']],
        'ARIMA_Forecast': [v for res in results for v in res['ARIMA_Forecast']],
        'ARIMA_Lower_CI': [v for res in results for v in res['ARIMA_Lower_CI']],
        'ARIMA_Upper_CI': [v for res in results for v in res['ARIMA_Upper_CI']],
        'Prophet_Forecast': [v for res in results for v in res['Prophet_Forecast']],
        'Prophet_Lower_CI': [v for res in results for v in res['Prophet_Lower_CI']],
        'Prophet_Upper_CI': [v for res in results for v in res['Prophet_Upper_CI']],
    })

    accuracy_df = pd.DataFrame([{
        'Property Type': res['Property Type'],
        'District': res['District'],
        'ARIMA_MAPE%': res['ARIMA_MAPE'],
        'Prophet_MAPE%': res['Prophet_MAPE']
    } for res in results])

    # Combine all historical actual vs predicted test data
    historical_test_df = pd.concat([
        res['Historical_Comparison'] for res in results if 'Historical_Comparison' in res
    ], ignore_index=True)

    # Save all files
    forecast_df.to_csv("C:\\Users\\steffiephang\\OneDrive - LBS Bina Holdings Sdn Bhd\\Desktop\\Steffie\\ADHD_Project\\AVM\\reliable_property_forecasts.csv", index=False)
    accuracy_df.to_csv("C:\\Users\\steffiephang\\OneDrive - LBS Bina Holdings Sdn Bhd\\Desktop\\Steffie\\ADHD_Project\\AVM\\model_accuracy_report.csv", index=False)
    historical_test_df.to_csv("C:\\Users\\steffiephang\\OneDrive - LBS Bina Holdings Sdn Bhd\\Desktop\\Steffie\\ADHD_Project\\AVM\\historical_test_vs_predicted.csv", index=False)

    print(f"\u2705 Generated {len(results)} forecasts")
    print(f"[Chart] Avg ARIMA MAPE: {accuracy_df['ARIMA_MAPE%'].mean():.1f}%")
    print(f"[Chart] Avg Prophet MAPE: {accuracy_df['Prophet_MAPE%'].mean():.1f}%")
    print(f"[Info] Historical test actual vs predicted saved: {len(historical_test_df)} rows")
else:
    print("\u274c No reliable forecasts generated.")

# ---------------- DASHBOARD ----------------
df_forecast = forecast_df.copy()
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
property_types = sorted(df_forecast['Property Type'].unique())
districts = sorted(df_forecast['District'].unique())
df_accuracy = accuracy_df.copy()

app = Dash(__name__)
app.title = "AVM Forecast Dashboard"
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('\ud83d\udcc8 Forecast Page | ', href='/'),
        dcc.Link('\ud83d\udcca Model Validation Page', href='/validation')
    ], style={'textAlign': 'center', 'padding': '10px', 'fontSize': '18px'}),
    html.Div(id='page-content')
])

forecast_layout = html.Div([
    html.H2("Property Forecast Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Property Type"),
        dcc.Dropdown(
            options=[{'label': p, 'value': p} for p in property_types],
            value=property_types[0],
            id='property-type-dropdown'
        )
    ], style={'width': '45%', 'display': 'inline-block'}),
    html.Div([
        html.Label("District"),
        dcc.Dropdown(
            options=[{'label': d, 'value': d} for d in districts],
            value=districts[0],
            id='district-dropdown'
        )
    ], style={'width': '45%', 'display': 'inline-block', 'float': 'right'}),
    dcc.Graph(id='forecast-graph'),
])

validation_layout = html.Div([
    html.H2("Model Validation (MAPE %)", style={'textAlign': 'center'}),

    # MAPE Table
    dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in df_accuracy.columns],
        data=df_accuracy.to_dict('records'),
        style_cell={'textAlign': 'center'},
        style_table={'overflowX': 'auto'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        page_size=15,
        sort_action='native',
        filter_action='native'
    ),

    html.Br(),

    # Dropdowns
    html.Div([
        html.Label("Select Property Type:"),
        dcc.Dropdown(
            id='property-type-dropdown',
            options=[{'label': i, 'value': i} for i in df['Property Type'].unique()],
            value=df['Property Type'].unique()[0]
        ),
        html.Br(),
        html.Label("Select District:"),
        dcc.Dropdown(
            id='district-dropdown',
            options=[{'label': i, 'value': i} for i in df['District'].unique()],
            value=df['District'].unique()[0]
        )
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Br(),

    # Historical test graph
    dcc.Graph(id='historical-test-graph'),

    html.Br(),

    # Forecast graph
    dcc.Graph(id='forecast-graph')
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    return validation_layout if pathname == '/validation' else forecast_layout

# Callback for Forecast Page
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('property-type-dropdown', 'value'),
     Input('district-dropdown', 'value')]
)
def update_forecast_graph(selected_type, selected_district):
    actual_df = df[
        (df['Property Type'] == selected_type) &
        (df['District'] == selected_district)
    ].groupby('date')['Price Per Sq Ft'].mean().reset_index()
    actual_df.columns = ['Date', 'Actual']

    forecast_filtered = df_forecast[
        (df_forecast['Property Type'] == selected_type) &
        (df_forecast['District'] == selected_district)
    ]

    if actual_df.empty or forecast_filtered.empty:
        return px.line(title="No data available for this selection.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_df['Date'], y=actual_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#4A90E2')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Forecast'],
        mode='lines',
        name='ARIMA Forecast',
        line=dict(color='#F5A623')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Upper_CI'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Lower_CI'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(245, 166, 35, 0.2)',
        name='ARIMA CI'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Forecast'],
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='#2E7D32')
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Upper_CI'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Lower_CI'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(46, 125, 50, 0.2)',
        name='Prophet CI'
    ))
    fig.update_layout(
        title=f'Forecasted Prices: {selected_type} in {selected_district}',
        xaxis_title='Date',
        yaxis_title='Price Per Sq Ft (RM)',
        template='plotly_white',
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickformat='%b %Y'),
        yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickprefix='RM '),
        legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5),
        height=450,
        margin=dict(l=40, r=40, t=60, b=60)
    )
    return fig

# Callback for Validation Page
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('historical-test-graph', 'figure')],
    [Input('property-type-dropdown', 'value'),
     Input('district-dropdown', 'value')]
)
def update_validation_graphs(selected_type, selected_district):
    # Forecast graph
    actual_df = df[
        (df['Property Type'] == selected_type) &
        (df['District'] == selected_district)
    ].groupby('date')['Price Per Sq Ft'].mean().reset_index()
    actual_df.columns = ['Date', 'Actual']

    forecast_filtered = df_forecast[
        (df_forecast['Property Type'] == selected_type) &
        (df_forecast['District'] == selected_district)
    ]

    if actual_df.empty or forecast_filtered.empty:
        forecast_fig = px.line(title="No data available for this selection.")
    else:
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=actual_df['Date'], y=actual_df['Actual'],
            mode='lines',
            name='Actual',
            line=dict(color='#4A90E2')
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Forecast'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='#F5A623')
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Upper_CI'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['ARIMA_Lower_CI'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(245, 166, 35, 0.2)',
            name='ARIMA CI'
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Forecast'],
            mode='lines',
            name='Prophet Forecast',
            line=dict(color='#2E7D32')
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Upper_CI'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        forecast_fig.add_trace(go.Scatter(
            x=forecast_filtered['Date'], y=forecast_filtered['Prophet_Lower_CI'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(46, 125, 50, 0.2)',
            name='Prophet CI'
        ))
        forecast_fig.update_layout(
            title=f'Forecasted Prices: {selected_type} in {selected_district}',
            xaxis_title='Date',
            yaxis_title='Price Per Sq Ft (RM)',
            template='plotly_white',
            xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickformat='%b %Y'),
            yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickprefix='RM '),
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5),
            height=450,
            margin=dict(l=40, r=40, t=60, b=60)
        )

    # Historical test graph
    df_test = pd.read_csv(
        r"C:\Users\steffiephang\OneDrive - LBS Bina Holdings Sdn Bhd\Desktop\Steffie\ADHD_Project\AVM\historical_test_vs_predicted.csv"
    )
    test_filtered = df_test[
        (df_test['Property Type'] == selected_type) &
        (df_test['District'] == selected_district)
    ]

    if test_filtered.empty:
        historical_fig = px.line(title="No historical test data available.")
    else:
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Scatter(
            x=test_filtered['Date'], y=test_filtered['Actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#50C878')
        ))
        historical_fig.add_trace(go.Scatter(
            x=test_filtered['Date'], y=test_filtered['Predicted'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#D81B60')
        ))
        historical_fig.update_layout(
            title=f'Historical Test: Actual vs Predicted ({selected_type} in {selected_district})',
            xaxis_title='Date',
            yaxis_title='Price Per Sq Ft (RM)',
            template='plotly_white',
            xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickformat='%b %Y'),
            yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickprefix='RM '),
            legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5),
            height=450,
            margin=dict(l=40, r=40, t=60, b=60)
        )

    return forecast_fig, historical_fig

if __name__ == '__main__':
    port = 2223
    print(f"\U0001F310 Serving at http://127.0.0.1:{port}")
    app.run(debug=True, use_reloader=False, port=port)
