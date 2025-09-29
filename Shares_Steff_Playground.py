import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px

# --- Fetch Data Function ---
def fetch_data(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)

# --- Main App ---
def main():
    st.set_page_config(layout="wide")  # Makes the whole app wide
    st.title("📈 Competitor Stock Monitoring – LBS Bina as Base")

    # Base company (LBS Bina)
    base_ticker = "5789.KL"

    # Competitors list
    competitors = {
        "S P Setia": "8664.KL",
        "Sime Darby Property": "5288.KL",
        "Eco World": "8206.KL",
        "UEM Sunrise": "5148.KL",
        "IOI Properties": "5249.KL",
        "Mah Sing": "8583.KL",
        "IJM Corporation": "3336.KL",
        "Sunway": "5211.KL",
        "Gamuda": "5398.KL",
        "OSK Holdings": "5053.KL",
        "UOA Development": "5200.KL",
    }

    # Date inputs
    start = st.date_input("Start date", value=date(2020, 1, 1))
    end = st.date_input("End date", value=date.today())

    # Select competitors
    selected_competitors = st.multiselect(
        "Select competitors to compare against LBS Bina",
        list(competitors.keys())
    )

    if st.button("Get Historical Data"):
        try:
            # Fetch base company data
            df_base = fetch_data(base_ticker, start, end + pd.Timedelta(days=1))
            df_base["Company"] = "LBS Bina"

            # Store all dataframes
            dfs = [df_base]

            # Fetch competitors
            for comp in selected_competitors:
                ticker = competitors[comp]
                df = fetch_data(ticker, start, end + pd.Timedelta(days=1))
                df["Company"] = comp
                dfs.append(df)

            # Combine into single dataframe
            df_all = pd.concat(dfs, keys=[d["Company"].iloc[0] for d in dfs]).reset_index()

            # --- Side by Side Charts ---
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Closing Price Comparison")
                fig_close = px.line(
                    df_all,
                    x="Date", y="Close", color="Company",
                    title="Closing Price",
                    width=800, height=500
                )
                st.plotly_chart(fig_close, use_container_width=True)

            with col2:
                st.subheader("Volume Comparison")
                fig_vol = px.line(
                    df_all,
                    x="Date", y="Volume", color="Company",
                    title="Trading Volume",
                    width=800, height=500
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            # --- Historical Data at Bottom ---
            st.subheader("📊 Historical Data Table")
            st.dataframe(df_all)

            # Download CSV
            csv = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Combined CSV",
                csv,
                file_name="competitor_comparison.csv",
                mime="text/csv"
            )

            # --- Automated Analysis & Explanations ---
            st.subheader("🤖 Automated Analysis & Explanations")

            # Closing Price Insights
            st.markdown("### Closing Price Explanation")
            first_closes = df_all.groupby('Company')['Close'].first()
            last_closes = df_all.groupby('Company')['Close'].last()
            pct_changes = ((last_closes - first_closes) / first_closes * 100).sort_values(ascending=False)

            st.write("This section analyzes the daily closing prices for LBS Bina and selected competitors over the chosen period. Key insights:")
            st.write("- **Trends**: Upward price movements indicate growth, while downward or flat trends suggest declines or stability.")
            st.write("- **Comparisons**: Use LBS Bina as a baseline to see how competitors perform relative to it.")
            st.write("Percentage Change Over the Period (from start to end date):")
            for company, pct in pct_changes.items():
                direction = "gain" if pct > 0 else "loss"
                st.write(f"- **{company}**: {pct:.2f}% {direction}.")
            
            top_gainer = pct_changes.idxmax()
            top_gain = pct_changes.max()
            bottom_performer = pct_changes.idxmin()
            bottom_performance = pct_changes.min()
            st.write(f"The top performer is **{top_gainer}** with a {top_gain:.2f}% gain, suggesting strong market confidence or positive developments.")
            st.write(f"The weakest performer is **{bottom_performer}** with a {bottom_performance:.2f}% change, which might indicate challenges or market headwinds.")
            
            if 'LBS Bina' in pct_changes:
                base_pct = pct_changes['LBS Bina']
                avg_pct = pct_changes.mean()
                performance_note = "outperformed" if base_pct > avg_pct else "underperformed" if base_pct < avg_pct else "matched"
                st.write(f"LBS Bina's performance ({base_pct:.2f}%) {performance_note} the group average ({avg_pct:.2f}%), providing context on its competitive standing.")

            # Volume Insights
            st.markdown("### Volume Explanation")
            avg_volumes = df_all.groupby('Company')['Volume'].mean().sort_values(ascending=False)
            max_volumes = df_all.groupby('Company')['Volume'].max()

            st.write("This section examines trading volumes over time. Volume spikes often signal high interest, news events, or market reactions. Steady volumes suggest consistent trading activity.")
            st.write("Average Trading Volumes:")
            for company, vol in avg_volumes.items():
                st.write(f"- **{company}**: {vol:,.0f} shares per day on average.")
            
            highest_avg_vol = avg_volumes.idxmax()
            highest_max_vol = max_volumes.idxmax()
            st.write(f"The stock with the highest average volume is **{highest_avg_vol}**, implying greater liquidity and investor attention.")
            st.write(f"The largest single-day volume spike was for **{highest_max_vol}** at {max_volumes[highest_max_vol]:,.0f} shares, potentially tied to a major event.")

            # Volatility Insights
            st.markdown("### Stock Volatility (Annualized)")
            df_all['Daily_Return'] = df_all.groupby('Company')['Close'].pct_change()
            volatilities = (df_all.groupby('Company')['Daily_Return'].std() * (252 ** 0.5)).sort_values(ascending=False)

            st.write("Volatility measures how much a stock’s price fluctuates, indicating risk. Higher values mean larger price swings (higher risk/reward). Calculated from daily returns, annualized.")
            st.write("Annualized Volatility:")
            for company, vol in volatilities.items():
                st.write(f"- **{company}**: {vol:.2%} volatility.")
            
            highest_vol = volatilities.idxmax()
            st.write(f"**{highest_vol}** has the highest volatility, making it a riskier but potentially more rewarding investment compared to steadier stocks.")
            
            # Volatility Chart
            volatility_df = pd.DataFrame({
                'Company': volatilities.index,
                'Volatility': volatilities.values
            })
            fig_volatility = px.bar(
                volatility_df,
                x='Company',
                y='Volatility',
                title="Annualized Volatility Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.2%'
            )
            fig_volatility.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_volatility, use_container_width=True)

            # Moving Average Trends
            st.markdown("### Moving Average Trends (50-Day)")
            df_all['MA50'] = df_all.groupby('Company')['Close'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop=True)
            df_all['Above_MA50'] = df_all['Close'] > df_all['MA50']
            ma_trends = df_all.groupby('Company')['Above_MA50'].mean() * 100

            st.write("This section shows the percentage of days each stock’s closing price was above its 50-day moving average, indicating bullish (above) or bearish (below) trends. A higher percentage suggests stronger upward momentum.")
            st.write("Percentage of Days Above 50-Day Moving Average:")
            for company, pct in ma_trends.sort_values(ascending=False).items():
                st.write(f"- **{company}**: {pct:.2f}% of days.")
            
            strongest_trend = ma_trends.idxmax()
            weakest_trend = ma_trends.idxmin()
            st.write(f"**{strongest_trend}** spent the most time above its 50-day moving average, indicating the strongest bullish trend.")
            st.write(f"**{weakest_trend}** spent the least time above its moving average, suggesting weaker momentum or a bearish trend.")
            if 'LBS Bina' in ma_trends:
                base_ma = ma_trends['LBS Bina']
                avg_ma = ma_trends.mean()
                trend_note = "stronger" if base_ma > avg_ma else "weaker" if base_ma < avg_ma else "similar"
                st.write(f"LBS Bina ({base_ma:.2f}%) had a {trend_note} trend compared to the group average ({avg_ma:.2f}%).")
            
            # Moving Average Trends Chart
            ma_trends_df = pd.DataFrame({
                'Company': ma_trends.index,
                'Percentage': ma_trends.values
            })
            fig_ma_trends = px.bar(
                ma_trends_df,
                x='Company',
                y='Percentage',
                title="Percentage of Days Above 50-Day Moving Average",
                width=800,
                height=500,
                color='Company',
                text_auto='.2f'
            )
            fig_ma_trends.update_layout(yaxis_title="Percentage of Days (%)")
            st.plotly_chart(fig_ma_trends, use_container_width=True)

            # Maximum Drawdown
            st.markdown("### Maximum Drawdown")
            df_pivot = df_all.pivot(index='Date', columns='Company', values='Close')
            rolling_max = df_pivot.cummax()
            drawdowns = (df_pivot - rolling_max) / rolling_max
            max_drawdowns = (-drawdowns.min() * 100).sort_values(ascending=False)

            st.write("This section measures the largest percentage drop from a peak price to a trough for each stock, showing downside risk. Larger drawdowns indicate higher risk of significant losses.")
            st.write("Maximum Drawdown Over the Period:")
            for company, drawdown in max_drawdowns.items():
                st.write(f"- **{company}**: {drawdown:.2f}% maximum loss.")
            
            highest_drawdown = max_drawdowns.idxmax()
            lowest_drawdown = max_drawdowns.idxmin()
            st.write(f"**{highest_drawdown}** had the largest drawdown, indicating the highest risk of significant price drops.")
            st.write(f"**{lowest_drawdown}** had the smallest drawdown, suggesting greater price stability.")
            if 'LBS Bina' in max_drawdowns:
                base_drawdown = max_drawdowns['LBS Bina']
                avg_drawdown = max_drawdowns.mean()
                risk_note = "riskier" if base_drawdown > avg_drawdown else "safer" if base_drawdown < avg_drawdown else "similar"
                st.write(f"LBS Bina’s drawdown ({base_drawdown:.2f}%) was {risk_note} compared to the group average ({avg_drawdown:.2f}%).")
            
            # Maximum Drawdown Chart
            drawdown_df = pd.DataFrame({
                'Company': max_drawdowns.index,
                'Drawdown': max_drawdowns.values
            })
            fig_drawdown = px.bar(
                drawdown_df,
                x='Company',
                y='Drawdown',
                title="Maximum Drawdown Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.2f'
            )
            fig_drawdown.update_layout(yaxis_title="Maximum Drawdown (%)")
            st.plotly_chart(fig_drawdown, use_container_width=True)

            # Average Daily Returns
            st.markdown("### Average Daily Returns")
            avg_daily_returns = (df_all.groupby('Company')['Daily_Return'].mean() * 100).sort_values(ascending=False)

            st.write("This section shows the average daily percentage return for each stock, indicating typical daily performance. Positive values suggest consistent daily gains, while negative values indicate losses.")
            st.write("Average Daily Returns:")
            for company, ret in avg_daily_returns.items():
                direction = "gain" if ret > 0 else "loss"
                st.write(f"- **{company}**: {ret:.4f}% {direction} per day.")
            
            highest_daily = avg_daily_returns.idxmax()
            lowest_daily = avg_daily_returns.idxmin()
            st.write(f"**{highest_daily}** had the highest average daily return, showing consistent daily gains.")
            st.write(f"**{lowest_daily}** had the lowest average daily return, indicating weaker daily performance.")
            if 'LBS Bina' in avg_daily_returns:
                base_daily = avg_daily_returns['LBS Bina']
                avg_daily = avg_daily_returns.mean()
                daily_note = "stronger" if base_daily > avg_daily else "weaker" if base_daily < avg_daily else "similar"
                st.write(f"LBS Bina’s average daily return ({base_daily:.4f}%) was {daily_note} compared to the group average ({avg_daily:.4f}%).")
            
            # Average Daily Returns Chart
            returns_df = pd.DataFrame({
                'Company': avg_daily_returns.index,
                'Average Daily Return': avg_daily_returns.values
            })
            fig_returns = px.bar(
                returns_df,
                x='Company',
                y='Average Daily Return',
                title="Average Daily Returns Comparison",
                width=800,
                height=500,
                color='Company',
                text_auto='.4f'
            )
            fig_returns.update_layout(yaxis_title="Average Daily Return (%)")
            st.plotly_chart(fig_returns, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching data: {e}")


if __name__ == "__main__":
    main()