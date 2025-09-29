# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from geopy.geocoders import Nominatim
import requests
from io import BytesIO
import json
import time
import os

st.set_page_config(layout="wide", page_title="MY Property Market Dashboard")

# -------------------------
# Helpers & cached funcs
# -------------------------
@st.cache_data
def load_geojson_safe(path_or_url: str):
    """Load GeoJSON from local file path or remote URL into a GeoDataFrame and also return raw geojson dict."""
    try:
        if str(path_or_url).lower().startswith("http"):
            resp = requests.get(path_or_url, timeout=30)
            resp.raise_for_status()
            raw = resp.content
            # geopandas can read BytesIO of GeoJSON
            gdf = gpd.read_file(BytesIO(raw))
            geojson = json.loads(resp.content)
            return gdf, geojson
        else:
            # local path
            gdf = gpd.read_file(path_or_url)
            # also get raw geojson dict by reading file
            try:
                with open(path_or_url, "r", encoding="utf-8") as f:
                    geojson = json.load(f)
            except Exception:
                geojson = None
            return gdf, geojson
    except Exception as e:
        st.error(f"Failed to load GeoJSON from {path_or_url}: {e}")
        return gpd.GeoDataFrame(), None

@st.cache_data
def geocode_places(place_list):
    """Geocode a list of place names with Nominatim (cached). Returns dict place -> (lat, lon)."""
    geolocator = Nominatim(user_agent="property_map_streamlit_v1")
    results = {}
    for name in place_list:
        key = str(name).strip()
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
# Config (use your GitHub raw URL here)
# -------------------------
DISTRICT_GEO_URL = "https://raw.githubusercontent.com/szilin08/MarketIntelligence08/main/gadm41_MYS_2.json"

# -------------------------
# UI & Upload
# -------------------------
st.title("Malaysian Property Market Intelligence Dashboard (Drillable Map)")

uploaded_file = st.file_uploader("Upload your Open Transaction Data.xlsx", type=["xlsx"])

# navigation drill stack kept in session_state as list of tuples (level, area_name)
if "drill_stack" not in st.session_state:
    st.session_state.drill_stack = []

# Back control
col1, col2 = st.columns([1, 6])
with col1:
    if st.button("⬅ Back"):
        if st.session_state.drill_stack:
            st.session_state.drill_stack.pop()
with col2:
    st.write("Click a district on the map to drill down. Use Back to go up the drill path.")

if uploaded_file is None:
    st.info("Please upload the Excel file to start analyzing.")
    st.stop()

# -------------------------
# Load & prepare dataframe
# -------------------------
raw = pd.read_excel(uploaded_file, sheet_name="Open Transaction Data", engine="openpyxl")

# sanitize and deduplicate columns
raw.columns = [c.strip() for c in raw.columns]
raw.columns = deduplicate_columns(raw.columns)

# common header candidates mapping
col_map_guess = {}
candidates = {
    "Property Type": ["Property Type", "Property type", "PROPERTY TYPE"],
    "District": ["District", "DISTRICT", "District "],
    "Mukim": ["Mukim", "Mukim "],
    "Scheme Name/Area": ["Scheme Name/Area", "Scheme Name", "Scheme"],
    "Month, Year of Transaction Date": ["Month, Year of Transaction Date", "Transaction Date", "Transaction Date Serial"],
    "Transaction Price": ["Transaction Price", "TransactionPrice", "Transaction Price ", "Price"]
}
for std, opts in candidates.items():
    for o in opts:
        if o in raw.columns:
            col_map_guess[std] = o
            break

# fallback: last numeric column for price
if "Transaction Price" not in col_map_guess:
    numeric_cols = raw.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        col_map_guess["Transaction Price"] = numeric_cols[-1]

df = raw.rename(columns={v: k for k, v in col_map_guess.items()})

required = ["Property Type", "District", "Mukim", "Scheme Name/Area", "Month, Year of Transaction Date", "Transaction Price"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Required column(s) missing from uploaded sheet: {missing}. Detected columns: {raw.columns.tolist()}")
    st.stop()

# derived and cleaning
df["Transaction Date"] = pd.to_datetime(df["Month, Year of Transaction Date"], errors="coerce")
df["Year"] = df["Transaction Date"].dt.year.fillna(0).astype(int)
df["Month"] = df["Transaction Date"].dt.month.fillna(0).astype(int)
df["Transaction Price"] = pd.to_numeric(df["Transaction Price"], errors="coerce")

df["District"] = df["District"].astype(str).apply(standardize_name)
df["Mukim"] = df["Mukim"].astype(str).apply(standardize_name)
df["Scheme Name/Area"] = df["Scheme Name/Area"].astype(str).apply(lambda x: x.strip() if pd.notna(x) else x)
df["Property Type"] = df["Property Type"].astype(str).apply(standardize_name)

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
districts_sel = st.sidebar.multiselect("District", options=sorted(df["District"].dropna().unique()), default=[])
mukims_sel = st.sidebar.multiselect("Mukim", options=sorted(df["Mukim"].dropna().unique()), default=[])
ptype_sel = st.sidebar.multiselect("Property Type", options=sorted(df["Property Type"].dropna().unique()), default=[])
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
# Determine display level (District -> Mukim -> Scheme)
# -------------------------
if not st.session_state.drill_stack:
    display_level = "District"
else:
    last_level, _ = st.session_state.drill_stack[-1]
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
# 1) District choropleth (using GeoJSON from GitHub)
# -------------------------
if display_level == "District":
    # load geojson from GitHub raw url
    gdf, geojson_raw = load_geojson_safe(DISTRICT_GEO_URL)

    if gdf.empty:
        st.error("Failed to load district GeoJSON (check URL / network).")
        st.stop()

    # Try to find the district name property in geojson features.
    # GADM usually uses NAME_2 for district and NAME_1 for state.
    # We'll prepare the aggregated DataFrame to match feature properties.NAME_2
    agg = filtered.groupby("District")["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "Value"})
    agg["District"] = agg["District"].astype(str).apply(standardize_name)

    # Determine feature property key (prefer NAME_2)
    candidate_key = None
    if geojson_raw and "features" in geojson_raw and len(geojson_raw["features"]) > 0:
        props = geojson_raw["features"][0].get("properties", {})
        # prefer NAME_2 (GADM), fallback to any key that looks like name
        if "NAME_2" in props:
            candidate_key = "NAME_2"
        else:
            for k in props.keys():
                if "name" in k.lower() or "NAME" in k:
                    candidate_key = k
                    break

    if candidate_key is None:
        st.warning("Could not find a suitable district name property in GeoJSON properties. Using geo dataframe 'NAME_2' column if present.")
        if "NAME_2" in gdf.columns:
            candidate_key = "NAME_2"

    # if candidate_key present, standardize the gdf property values
    if candidate_key:
        # create a small DataFrame for plotting with correct 'District' values
        # Ensure the property values are standardized to match the 'agg' values
        gdf[candidate_key] = gdf[candidate_key].astype(str).apply(standardize_name)

    # If we have a geojson_raw, use plotly's featureidkey to join by property
    if geojson_raw and candidate_key:
        # prepare a DataFrame with column named exactly "District" to match locations arg
        plot_df = agg.copy()
        # If some districts in geojson do not exist in plot_df, that's ok (they'll be blank)
        fig = px.choropleth_mapbox(
            plot_df,
            geojson=geojson_raw,
            locations="District",
            color="Value",
            featureidkey=f"properties.{candidate_key}",
            color_continuous_scale="YlOrRd",
            mapbox_style="open-street-map",
            center={"lat": 4.2105, "lon": 101.9758},
            zoom=5,
            labels={"Value": "Avg Price (RM)"},
            hover_data={"District": True, "Value": ":,.0f"}
        )
    else:
        # fallback to merging using gdf: merge gdf and agg on NAME_2 (if present)
        if "NAME_2" in gdf.columns:
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
                labels={"Value": "Avg Price (RM)"},
                hover_data={"NAME_2": True, "Value": ":,.0f"}
            )
        else:
            st.error("Cannot determine district name field in GeoJSON. Inspect geojson properties/columns.")
            st.stop()

    # display and capture clicks using streamlit_plotly_events
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
    st.plotly_chart(fig, use_container_width=True)

    if clicked:
        data = clicked[0]
        # try robust extraction of clicked district name
        chosen = None
        # plotly returns 'location' for choropleth when featureidkey is used:
        if "location" in data and data["location"]:
            chosen = standardize_name(data["location"])
        # try properties if present
        elif "properties" in data and isinstance(data["properties"], dict):
            # try candidate_key inside properties
            if candidate_key and data["properties"].get(candidate_key):
                chosen = standardize_name(data["properties"].get(candidate_key))
        # as fallback, try pointIndex -> map via merged (if defined)
        elif "pointIndex" in data:
            idx = data.get("pointIndex")
            try:
                if 'merged' in locals() and idx is not None:
                    chosen = standardize_name(merged.iloc[int(idx)][candidate_key if candidate_key else "NAME_2"])
            except Exception:
                chosen = None

        if chosen:
            st.session_state.drill_stack.append(("District", chosen))
            st.experimental_rerun()
        else:
            st.warning("Could not determine which district was clicked. Try clicking a different area.")

# -------------------------
# 2) Mukim bubble map (geocode fallback) - inside selected district
# -------------------------
elif display_level == "Mukim":
    # find parent district from drill stack
    parent = None
    for lvl, area in reversed(st.session_state.drill_stack):
        if lvl == "District":
            parent = area
            break
    if parent is None:
        st.error("Parent district not found in drill stack. Click a district first.")
        st.stop()

    st.subheader(f"Mukim-level (bubble map) for: {parent}")
    df_sub = filtered[filtered["District"] == parent].copy()
    if df_sub.empty:
        st.info("No transactions for this district under current filters.")
    else:
        mukim_agg = df_sub.groupby("Mukim")["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "AvgPrice"})
        mukim_agg["Mukim"] = mukim_agg["Mukim"].astype(str).apply(standardize_name)

        # geocode mukim names (capped)
        mukims_to_geocode = mukim_agg["Mukim"].dropna().unique().tolist()[:200]
        coords = geocode_places(mukims_to_geocode)
        mukim_agg["lat"] = mukim_agg["Mukim"].map(lambda x: coords.get(x, (None, None))[0])
        mukim_agg["lon"] = mukim_agg["Mukim"].map(lambda x: coords.get(x, (None, None))[1])
        points = mukim_agg.dropna(subset=["lat", "lon"]).copy()

        if points.empty:
            st.info("No geocoded mukim coordinates found.")
        else:
            fig = px.scatter_mapbox(
                points,
                lat="lat", lon="lon",
                size="AvgPrice",
                color="AvgPrice",
                hover_name="Mukim",
                hover_data={"AvgPrice": ":,.0f"},
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
                    chosen_mukim = points.iloc[int(idx)]["Mukim"]
                    st.session_state.drill_stack.append(("Mukim", chosen_mukim))
                    st.experimental_rerun()

# -------------------------
# 3) Scheme bubble map & details (within Mukim or District)
# -------------------------
elif display_level == "Scheme":
    # get parent mukim or district
    parent_mukim = None
    parent_district = None
    for lvl, area in reversed(st.session_state.drill_stack):
        if lvl == "Mukim" and parent_mukim is None:
            parent_mukim = area
        if lvl == "District" and parent_district is None:
            parent_district = area

    ctx = parent_mukim if parent_mukim else parent_district
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

        # geocode schemes (capped)
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
# Bottom analytics (current selection)
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
