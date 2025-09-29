import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
from streamlit_plotly_events import plotly_events
from geopy.geocoders import Nominatim
import time
import os

st.set_page_config(layout="wide", page_title="MY Property Market Dashboard")

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
# Levels: District (choropleth) -> Scheme (bubble fallback)
if not st.session_state.drill_stack:
    display_level = "District"
else:
    last_level, last_area = st.session_state.drill_stack[-1]
    if last_level == "District":
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
# Scheme bubble map (geocode fallback) with year bubbles
# -------------------------
elif display_level == "Scheme":
    parent = None
    for lvl, area in st.session_state.drill_stack[::-1]:
        if lvl == "District":
            parent = area
            break
    if parent is None:
        st.error("Parent district not found in drill stack. Use Back and click a district first.")
        st.stop()

    st.subheader(f"Scheme-level (bubble map) for: {parent}")
    df_sub = filtered[filtered["District"] == parent].copy()
    if df_sub.empty:
        st.info("No transactions for this district under current filters.")
    else:
        scheme_year_agg = df_sub.groupby(["Scheme Name/Area", "Year"])["Transaction Price"].mean().reset_index().rename(columns={"Transaction Price": "AvgPrice"})
        scheme_year_agg["Scheme Name/Area"] = scheme_year_agg["Scheme Name/Area"].astype(str).apply(standardize_name)
        scheme_year_agg = scheme_year_agg[scheme_year_agg["Year"] > 0]

        schemes = scheme_year_agg["Scheme Name/Area"].dropna().unique().tolist()[:200]
        coords = geocode_places(schemes)
        scheme_year_agg["lat"] = scheme_year_agg["Scheme Name/Area"].map(lambda x: coords.get(x, (None, None))[0])
        scheme_year_agg["lon"] = scheme_year_agg["Scheme Name/Area"].map(lambda x: coords.get(x, (None, None))[1])
        points = scheme_year_agg.dropna(subset=["lat", "lon"])

        if points.empty:
            st.info("No geocoded scheme coordinates found.")
        else:
            fig = px.scatter_mapbox(
                points,
                lat="lat", lon="lon",
                size="AvgPrice",
                color="Year",
                hover_name="Scheme Name/Area",
                hover_data={"AvgPrice": ":.0f", "Year": True},
                mapbox_style="open-street-map",
                center={"lat": points["lat"].mean(), "lon": points["lon"].mean()},
                zoom=9,
                title=f"Avg Transaction Price by Scheme and Year in {parent}"
            )
            clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)
            st.plotly_chart(fig, use_container_width=True)

            if clicked:
                idx = clicked[0].get("pointIndex")
                if idx is not None:
                    chosen = points.iloc[idx]["Scheme Name/Area"]
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
