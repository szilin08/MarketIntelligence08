import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from geopy.geocoders import Nominatim
import time
import os
from prophet import Prophet  # For time-series forecasting
import io

# Set page config at the top (once per app)
st.set_page_config(layout="wide", page_title="MY Property Market Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Property Market Dashboard", "Automated Valuation Model"])

# Define standardize_name function globally
def standardize_name(s):
    if pd.isna(s):
        return s
    return str(s).strip().title()

if page == "Property Market Dashboard":
    # -------------------------
    # Helpers & cached funcs
    # -------------------------
    @st.cache_data
    def load_geojson_safe(path):
        """Try to load a geojson safely (force GeoJSON driver if needed)."""
        try:
            return gpd.read_file(f"GeoJSON:{path}")
        except Exception:
            return gpd.read_file(path, driver="GeoJSON")

    @st.cache_data
    def geocode_places(place_list):
        """Geocode a list of place names with Nominatim (cached)."""
        geolocator = Nominatim(user_agent="property_map_streamlit")
        results = {}
        for p in place_list:
            key = str(p).strip()
            if not key or key in results:
                continue
            try:
                loc = geolocator.geocode(f"{key}, Malaysia", timeout=10)
                if loc:
                    results[key] = (loc.latitude, loc.longitude)
                else:
                    results[key] = (None, None)
            except Exception:
                results[key] = (None, None)
            time.sleep(1)  # polite
        return results

    def standardize_name(s):
        if pd.isna(s):
            return s
        return str(s).strip().title()

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

    # -------------------------
    # Config: local files (update paths if different)
    # -------------------------
    DISTRICT_GEO = r"C:\Users\steffiephang\OneDrive - LBS Bina Holdings Sdn Bhd\Desktop\Steffie\ADHD_Project\AVM 2\gadm41_MYS_2.json"

    # -------------------------
    # UI & Upload
    # -------------------------
    st.title("Malaysian Property Market Intelligence Dashboard (Drillable Map)")

    uploaded_file = st.file_uploader("Upload your Open Transaction Data.xlsx", type=["xlsx"])

    # session state for navigation
    if "drill_stack" not in st.session_state:
        st.session_state.drill_stack = []  # list of tuples (level, area_name)

    # Back control
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅ Back"):
            if st.session_state.drill_stack:
                st.session_state.drill_stack.pop()
    with col2:
        st.write("Click a region on the map to drill down. Use Back to go up.")

    if uploaded_file is None:
        st.info("Please upload the Excel file to start analyzing.")
        st.stop()

    # -------------------------
    # Load & prepare dataframe
    # -------------------------
    raw = pd.read_excel(uploaded_file, sheet_name="Open Transaction Data")

    # ✅ fix duplicate column names
    raw.columns = [c.strip() for c in raw.columns]
    raw.columns = deduplicate_columns(raw.columns)

    # attempt to map common header variants to expected names
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

    # if price not found, pick last numeric column as fallback
    if "Transaction Price" not in col_map_guess:
        numeric_cols = raw.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col_map_guess["Transaction Price"] = numeric_cols[-1]

    df = raw.rename(columns={v: k for k, v in col_map_guess.items()})
    required = ["Property Type", "District", "Mukim", "Scheme Name/Area", "Month, Year of Transaction Date", "Transaction Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Required column(s) missing: {missing}. Detected columns: {raw.columns.tolist()}")
        st.stop()

    # clean and derived columns
    df["Transaction Date"] = pd.to_datetime(df["Month, Year of Transaction Date"], errors="coerce")
    df["Year"] = df["Transaction Date"].dt.year.fillna(0).astype(int)
    df["Month"] = df["Transaction Date"].dt.month.fillna(0).astype(int)
    df["Transaction Price"] = pd.to_numeric(df["Transaction Price"], errors="coerce")

    df["District"] = df["District"].astype(str).apply(standardize_name)
    df["Mukim"] = df["Mukim"].astype(str).apply(standardize_name)
    df["Scheme Name/Area"] = df["Scheme Name/Area"].astype(str).apply(lambda x: x.strip() if pd.notna(x) else x)
    df["Property Type"] = df["Property Type"].astype(str).apply(standardize_name)

    # sidebar filters
    st.sidebar.header("Filters")
    states_sel = st.sidebar.multiselect("State (if present)", options=sorted(df.get("State", df["District"].unique())), default=[])
    districts_sel = st.sidebar.multiselect("District", options=sorted(df["District"].unique()), default=[])
    mukims_sel = st.sidebar.multiselect("Mukim", options=sorted(df["Mukim"].unique()), default=[])
    ptype_sel = st.sidebar.multiselect("Property Type", options=sorted(df["Property Type"].unique()), default=[])
    min_year = int(df["Year"].replace(0, pd.NA).min(skipna=True))
    max_year = int(df["Year"].max(skipna=True))
    year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))
    min_price = int(df["Transaction Price"].min(skipna=True))
    max_price = int(df["Transaction Price"].max(skipna=True))
    price_range = st.sidebar.slider("Price range", min_price, max_price, (min_price, max_price))

    # apply filters
    filtered = df.copy()
    if districts_sel:
        filtered = filtered[filtered["District"].isin(districts_sel)]
    if mukims_sel:
        filtered = filtered[filtered["Mukim"].isin(mukims_sel)]
    if ptype_sel:
        filtered = filtered[filtered["Property Type"].isin(ptype_sel)]
    filtered = filtered[(filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])]
    filtered = filtered[(filtered["Transaction Price"] >= price_range[0]) & (filtered["Transaction Price"] <= price_range[1])]

    # -------------------------
    # Determine display level
    # -------------------------
    # Levels: District (choropleth) -> Mukim (bubble fallback) -> Scheme (bubble)
    if not st.session_state.drill_stack:
        display_level = "District"
    else:
        last_level, last_area = st.session_state.drill_stack[-1]
        if last_level == "District":
            display_level = "Mukim"
        elif last_level == "Mukim":
            display_level = "Scheme"
        elif last_level == "Scheme Name/Area":
            display_level = "Scheme"
        else:
            display_level = "District"

    st.header(f"Map view: {display_level}")

    # -------------------------
    # District choropleth
    # -------------------------
    if display_level == "District":
        if not os.path.exists(DISTRICT_GEO):
            st.error(f"District GeoJSON not found at {DISTRICT_GEO}. Please place the file there.")
            st.stop()

        gdf = load_geojson_safe(DISTRICT_GEO)
        if "NAME_2" not in gdf.columns:
            st.warning(f"Expected 'NAME_2' in {DISTRICT_GEO} but found columns: {gdf.columns.tolist()}")
        gdf["NAME_2"] = gdf["NAME_2"].astype(str).apply(standardize_name)

        agg = filtered.groupby("District")["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "Value"})
        agg["District"] = agg["District"].astype(str).apply(standardize_name)

        merged = gdf.merge(agg, left_on="NAME_2", right_on="District", how="left")
        merged["Value"] = merged["Value"].fillna(0)

        fig = px.choropleth_mapbox(
            merged,
            geojson=merged.geometry.__geo_interface__,
            locations=merged.index,
            color="Value",
            color_continuous_scale="YlOrRd",
            mapbox_style="open-street-map",
            center={"lat": 4.2105, "lon": 101.9758},
            zoom=5,
            hover_data={"NAME_2": True, "Value": True},
            labels={"Value": "Avg Price (RM)"}
        )
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
        st.plotly_chart(fig, use_container_width=True)

        if clicked:
            pt = clicked[0]
            idx = pt.get("pointIndex")
            if idx is not None:
                chosen = merged.iloc[idx]["NAME_2"]
                st.session_state.drill_stack.append(("District", chosen))
                st.experimental_rerun()

    # -------------------------
    # Mukim bubble map (geocode fallback)
    # -------------------------
    elif display_level == "Mukim":
        parent = None
        for lvl, area in st.session_state.drill_stack[::-1]:
            if lvl == "District":
                parent = area
                break
        if parent is None:
            st.error("Parent district not found in drill stack. Use Back and click a district first.")
            st.stop()

        st.subheader(f"Mukim-level (bubble map) for: {parent}")
        df_sub = filtered[filtered["District"] == parent].copy()
        if df_sub.empty:
            st.info("No transactions for this district under current filters.")
        else:
            mukim_agg = df_sub.groupby("Mukim")["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "AvgPrice"})
            mukim_agg["Mukim"] = mukim_agg["Mukim"].astype(str).apply(standardize_name)

            mukims_to_geocode = mukim_agg["Mukim"].dropna().unique().tolist()[:200]
            coords = geocode_places(mukims_to_geocode)
            mukim_agg["lat"] = mukim_agg["Mukim"].map(lambda x: coords.get(x, (None, None))[0])
            mukim_agg["lon"] = mukim_agg["Mukim"].map(lambda x: coords.get(x, (None, None))[1])
            points = mukim_agg.dropna(subset=["lat", "lon"])

            if points.empty:
                st.info("No geocoded mukim coordinates found.")
            else:
                fig = px.scatter_mapbox(
                    points,
                    lat="lat", lon="lon",
                    size="AvgPrice",
                    color="AvgPrice",
                    hover_name="Mukim",
                    hover_data={"AvgPrice": ":.0f"},
                    mapbox_style="open-street-map",
                    center={"lat": points["lat"].mean(), "lon": points["lon"].mean()},
                    zoom=9,
                    title=f"Avg Transaction Price by Mukim in {parent}"
                )
                clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
                st.plotly_chart(fig, use_container_width=True)

                if clicked:
                    idx = clicked[0].get("pointIndex")
                    if idx is not None:
                        chosen_mukim = points.iloc[idx]["Mukim"]
                        st.session_state.drill_stack.append(("Mukim", chosen_mukim))
                        st.experimental_rerun()

    # -------------------------
    # Scheme bubble map & details
    # -------------------------
    elif display_level == "Scheme":
        parent_mukim = None
        parent_district = None
        for lvl, area in st.session_state.drill_stack[::-1]:
            if lvl == "Mukim" and parent_mukim is None:
                parent_mukim = area
            if lvl == "District" and parent_district is None:
                parent_district = area

        ctx = f"{parent_mukim if parent_mukim else parent_district}"
        st.subheader(f"Scheme-level for: {ctx}")

        if parent_mukim:
            df_sub = filtered[filtered["Mukim"] == parent_mukim].copy()
        elif parent_district:
            df_sub = filtered[filtered["District"] == parent_district].copy()
        else:
            df_sub = filtered.copy()

        if df_sub.empty:
            st.info("No transactions in this scope.")
        else:
            scheme_agg = df_sub.groupby("Scheme Name/Area").agg(
                Transaction_Count=("Transaction Price", "count"),
                Avg_Price=("Transaction Price", "mean")
            ).reset_index()
            scheme_agg["Scheme Name/Area"] = scheme_agg["Scheme Name/Area"].astype(str)

            schemes = scheme_agg["Scheme Name/Area"].dropna().unique().tolist()[:300]
            coords = geocode_places(schemes)
            scheme_agg["lat"] = scheme_agg["Scheme Name/Area"].map(lambda x: coords.get(x, (None, None))[0])
            scheme_agg["lon"] = scheme_agg["Scheme Name/Area"].map(lambda x: coords.get(x, (None, None))[1])
            points = scheme_agg.dropna(subset=["lat", "lon"]).copy()

            if not points.empty:
                fig = px.scatter_mapbox(
                    points,
                    lat="lat", lon="lon",
                    size="Avg_Price",
                    color="Transaction_Count",
                    hover_name="Scheme Name/Area",
                    hover_data={"Avg_Price": ":.0f", "Transaction_Count": True},
                    mapbox_style="open-street-map",
                    center={"lat": points["lat"].mean(), "lon": points["lon"].mean()},
                    zoom=11,
                    title=f"Schemes in {ctx}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geocoded scheme coordinates found.")

            chosen = st.selectbox("Pick a scheme to inspect", options=["--"] + scheme_agg["Scheme Name/Area"].fillna("N/A").tolist())
            if chosen and chosen != "--":
                st.write("Summary for scheme:")
                st.write(scheme_agg[scheme_agg["Scheme Name/Area"] == chosen].T)
                st.dataframe(df_sub[df_sub["Scheme Name/Area"] == chosen].sort_values("Transaction Date", ascending=False).reset_index(drop=True))
                if st.button(f"Drill into scheme: {chosen}"):
                    st.session_state.drill_stack.append(("Scheme Name/Area", chosen))
                    st.experimental_rerun()

    # -------------------------
    # Bottom analytics
    # -------------------------
    st.header("Analytics (current selection)")
    mask = pd.Series(True, index=filtered.index)
    for lvl, area in st.session_state.drill_stack:
        if lvl == "District":
            mask &= filtered["District"] == area
        elif lvl == "Mukim":
            mask &= filtered["Mukim"] == area
        elif lvl == "Scheme Name/Area":
            mask &= filtered["Scheme Name/Area"] == area

    current = filtered[mask].copy()

    if current.empty:
        st.write("No data for the current selection.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", len(current))
        c2.metric("Avg Price", f"RM {current['Transaction Price'].mean():,.0f}")
        c3.metric("Median Price", f"RM {current['Transaction Price'].median():,.0f}")
        c4.metric("Price Range", f"RM {int(current['Transaction Price'].min()):,} - RM {int(current['Transaction Price'].max()):,}")

        trend = current.groupby(["Year", "Month"])["Transaction Price"].mean().reset_index()
        if not trend.empty:
            trend["Date"] = pd.to_datetime(trend["Year"].astype(str) + "-" + trend["Month"].astype(str) + "-01", errors="coerce")
            fig_trend = px.line(trend.sort_values("Date"), x="Date", y="Transaction Price", title="Avg Price Trend")
            st.plotly_chart(fig_trend, use_container_width=True)

        type_avg = current.groupby("Property Type")["Transaction Price"].mean().reset_index().sort_values("Transaction Price", ascending=False)
        if not type_avg.empty:
            fig_type = px.bar(type_avg, x="Property Type", y="Transaction Price", title="Avg Price by Property Type")
            st.plotly_chart(fig_type, use_container_width=True)

    st.subheader("Raw data (current selection)")
    st.dataframe(current.reset_index(drop=True))

elif page == "Automated Valuation Model":
    st.title("Automated Valuation Model")
    uploaded_file = st.file_uploader("Upload your Open Transaction Data.xlsx for Valuation", type=["xlsx"])

    if uploaded_file is None:
        st.info("Please upload the Excel file to start the valuation model.")
        st.stop()

    # Load and prepare data for valuation
    raw = pd.read_excel(uploaded_file, sheet_name="Open Transaction Data")
    raw.columns = [c.strip() for c in raw.columns]
    raw.columns = deduplicate_columns(raw.columns)

    col_map_guess = {}
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

    # Filters for valuation
    st.sidebar.header("Valuation Filters")
    val_districts = st.sidebar.multiselect("District", options=sorted(df_val["District"].unique()), default=[])
    val_mukims = st.sidebar.multiselect("Mukim", options=sorted(df_val["Mukim"].unique()), default=[])
    val_ptype = st.sidebar.multiselect("Property Type", options=sorted(df_val["Property Type"].unique()), default=[])
    val_year_range = st.sidebar.slider("Year range", int(df_val["Year"].min()), int(df_val["Year"].max()), (int(df_val["Year"].min()), int(df_val["Year"].max())))

    # Apply valuation filters
    val_filtered = df_val.copy()
    if val_districts:
        val_filtered = val_filtered[val_filtered["District"].isin(val_districts)]
    if val_mukims:
        val_filtered = val_filtered[val_filtered["Mukim"].isin(val_mukims)]
    if val_ptype:
        val_filtered = val_filtered[val_filtered["Property Type"].isin(val_ptype)]
    val_filtered = val_filtered[(val_filtered["Year"] >= val_year_range[0]) & (val_filtered["Year"] <= val_year_range[1])]

    # Prepare data for Prophet model
    if not val_filtered.empty:
        trend_data = val_filtered.groupby(["Year", "Month"])["Transaction Price"].mean().reset_index()
        trend_data["Date"] = pd.to_datetime(trend_data[["Year", "Month"]].assign(day=1))
        prophet_df = trend_data.rename(columns={"Date": "ds", "Transaction Price": "y"})

        # Fit Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)

        # Forecast 12 months ahead
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

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