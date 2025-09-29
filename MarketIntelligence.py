# app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from geopy.geocoders import Nominatim
import time

st.set_page_config(layout="wide", page_title="MY Property Market Dashboard")

# -------------------------
# Helpers & cached funcs
# -------------------------
@st.cache_data
def load_geojson_safe(path):
    """Try to load a geojson safely (force GeoJSON driver if needed)."""
    try:
        return gpd.read_file(path)
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

# -------------------------
# Config: GitHub-hosted GeoJSON
# -------------------------
DISTRICT_GEO = "https://raw.githubusercontent.com/szilin08/MarketIntelligence08/main/gadm41_MYS_2.json"

# -------------------------
# UI & Upload
# -------------------------
st.title("Malaysian Property Market Intelligence Dashboard (Interactive Drilldown)")

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

# deduplicate column names
raw.columns = [c.strip() for c in raw.columns]
def deduplicate_columns(columns):
    seen, new_cols = {}, []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    return new_cols
raw.columns = deduplicate_columns(raw.columns)

# column mapping
candidates = {
    "Property Type": ["Property Type", "PROPERTY TYPE"],
    "District": ["District", "DISTRICT"],
    "Mukim": ["Mukim"],
    "Scheme Name/Area": ["Scheme Name/Area", "Scheme Name", "Scheme"],
    "Month, Year of Transaction Date": ["Month, Year of Transaction Date", "Transaction Date"],
    "Transaction Price": ["Transaction Price", "TransactionPrice"]
}
col_map_guess = {}
for std, opts in candidates.items():
    for o in opts:
        if o in raw.columns:
            col_map_guess[std] = o
            break
if "Transaction Price" not in col_map_guess:
    num_cols = raw.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        col_map_guess["Transaction Price"] = num_cols[-1]

df = raw.rename(columns={v: k for k, v in col_map_guess.items()})
required = ["Property Type", "District", "Mukim", "Scheme Name/Area", "Month, Year of Transaction Date", "Transaction Price"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Required column(s) missing: {missing}")
    st.stop()

# clean & derive
df["Transaction Date"] = pd.to_datetime(df["Month, Year of Transaction Date"], errors="coerce")
df["Year"] = df["Transaction Date"].dt.year.fillna(0).astype(int)
df["Month"] = df["Transaction Date"].dt.month.fillna(0).astype(int)
df["Transaction Price"] = pd.to_numeric(df["Transaction Price"], errors="coerce")
df["District"] = df["District"].astype(str).apply(standardize_name)
df["Mukim"] = df["Mukim"].astype(str).apply(standardize_name)
df["Scheme Name/Area"] = df["Scheme Name/Area"].astype(str).apply(standardize_name)
df["Property Type"] = df["Property Type"].astype(str).apply(standardize_name)

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters")
districts_sel = st.sidebar.multiselect("District", sorted(df["District"].unique()))
mukims_sel = st.sidebar.multiselect("Mukim", sorted(df["Mukim"].unique()))
ptype_sel = st.sidebar.multiselect("Property Type", sorted(df["Property Type"].unique()))

year_range = st.sidebar.slider("Year range", int(df["Year"].min()), int(df["Year"].max()),
                               (int(df["Year"].min()), int(df["Year"].max())))
price_range = st.sidebar.slider("Price range", int(df["Transaction Price"].min()),
                                int(df["Transaction Price"].max()),
                                (int(df["Transaction Price"].min()), int(df["Transaction Price"].max())))

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
# Decide level
# -------------------------
if not st.session_state.drill_stack:
    display_level = "District"
else:
    last_level, _ = st.session_state.drill_stack[-1]
    if last_level == "District":
        display_level = "Mukim"
    elif last_level == "Mukim":
        display_level = "Scheme"
    else:
        display_level = "District"

st.header(f"Map view: {display_level}")

# -------------------------
# District Map
# -------------------------
if display_level == "District":
    gdf = load_geojson_safe(DISTRICT_GEO)
    gdf["NAME_2"] = gdf["NAME_2"].astype(str).apply(standardize_name)

    agg = filtered.groupby("District")["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "Value"})
    agg["District"] = agg["District"].astype(str).apply(standardize_name)
    merged = gdf.merge(agg, left_on="NAME_2", right_on="District", how="left").fillna(0)

    fig = px.choropleth_mapbox(
        merged,
        geojson=merged.geometry.__geo_interface__,
        locations=merged.index,
        color="Value",
        mapbox_style="open-street-map",
        center={"lat": 4.2, "lon": 101.9}, zoom=5,
        hover_data={"NAME_2": True, "Value": True}
    )
    event = st.plotly_chart(fig, use_container_width=True, on_event="plotly_click")
    if event:
        clicked_district = event["points"][0]["hovertext"]
        st.session_state.drill_stack.append(("District", clicked_district))
        st.experimental_rerun()

# -------------------------
# Analytics (shared)
# -------------------------
st.subheader("Analytics (current selection)")
mask = pd.Series(True, index=filtered.index)
for lvl, area in st.session_state.drill_stack:
    mask &= filtered[lvl] == area if lvl in filtered.columns else True
current = filtered[mask]

if current.empty:
    st.write("No data here.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", len(current))
    c2.metric("Avg Price", f"RM {current['Transaction Price'].mean():,.0f}")
    c3.metric("Median", f"RM {current['Transaction Price'].median():,.0f}")
