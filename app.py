import os
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from shapely.geometry import Point

from predict import predict_price


# ===============================
# 1. BASIC PAGE CONFIG & CSS
# ===============================

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.markdown(
    """
<style>
/* Global font */
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.block-container {
    max-width: 900px !important;
    padding-top: 1.5rem !important;
    background: #f8fbff;
}

/* Headings and text */
h1, h2, h3, h4, h5, h6, .card-title, label, .stTextInput label, .stNumberInput label, .stSelectbox label {
    font-weight: 800 !important;
    color: #111827 !important;
}
.st-b7 {
    background-color: rgb(250 250 250);
}
.st-bc {
    color: rgb(0 0 0);
}

li {
    background-color: white !important;
}
@supports (scrollbar-color:transparent transparent) {
    * {
        scrollbar-width: thin;
        scrollbar-color: rgb(172 177 195 / 86%) #fffdfd;
    }
}

@supports (scrollbar-color: transparent transparent) {
    ul[role="listbox"]:hover {
        scrollbar-color: rgba(250, 250, 250, 0.4) transparent;
    }
}



div[data-testid="stSelectbox"][aria-label="Hướng nhà"] span {
    color: blue;
}


.stMarkdown p, .stCaption, .stInfo, .stWarning, .stSuccess, .stError {
    color: #1f2937 !important;
}

/* Hide number input steppers so fields look like pure text numeric inputs */
.stNumberInput button {
    display: none !important;
}

.stNumberInput input, .stTextInput input {
    font-weight: 700 !important;
    color: #0f172a !important;
    border-radius: 10px !important;
    background: #ffffff !important;
}

/* Basic card styling */
.card {
    background: linear-gradient(180deg, #ffffff 0%, #f4f9ff 100%);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border: 1px solid #dbeafe;
    box-shadow: 0 6px 24px rgba(59, 130, 246, 0.15);
    margin-bottom: 1.1rem;
}

.card-title {
    font-weight: 800;
    font-size: 1.05rem;
    margin-bottom: 0.9rem;
    color: #0f172a !important;
}
.st-emotion-cache-13k62yr {
    position: absolute;
    background: rgb(250 250 250);
    color: rgb(250, 250, 250);
    inset: 0px;
    color-scheme: dark;
    overflow: hidden;
}
.stButton>button {
    border-radius: 999px !important;
    padding: 1rem 2.5rem !important;
    background: #0489ff !important;
    color: #fff !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    max-width: 400px !important;
    margin: 0 auto !important;
    display: block !important;
}

div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
.st-emotion-cache-gx6i9d p {
   color: rgb(250 250 250)!important;
}
.st-dt {
    background-color: rgb(255 255 255);
}
.st-emotion-cache-1kncm28 {
    
    background-color: rgb(255 255 255);
    
}

div[data-testid="stSelectbox"] div {
    font-size: 16px;
    color: #ffe5e5;
    color-scheme: light;
    font-weight: 700 !important;
}


div[data-testid="stSelectbox"] div {
    font-size: 16px;
    color: #000000;
    color-scheme: light;
}

.st-bo {
    background-color: rgb(255 255 255);
}


.st-de {
    background-color: rgb(255 255 255);
}


.st-emotion-cache-gquqoo {
    background: rgb(255 255 255);
}

.st-emotion-cache-13k62yr {
    color: rgb(0 0 0);
}
 ul{
    background: rgb(255 255 255);
}
 p{
    color: rgb(0 0 0);
    font-weight: 600 !important;
}
.st-emotion-cache-15ckyqm {
    color: rgb(0 0 0);
}

.st-emotion-cache-8jg6dq {
    padding: 0.2em 0.4em;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    margin: 0px;
    border-radius: 0.25rem;
    background: rgb(255 255 255) !important;
    color: rgb(0 0 0) !important;
    font-family: "Source Code Pro", monospace;
    font-size: 0.75em;
    font-weight: 600 !important;
}

.st-cc {
    background: rgb(255 255 255)!important;
}

.st-dr {
    background: rgb(255 255 255)!important;
}

button[data-testid="stBaseButton-secondary"] {
    background: rgb(255 255 255) !important;
    color: rgb(0 0 0) !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;

    border: 2px solid #000000 !important;
    border-radius: 10px !important;   /* vuông bo nhẹ */

    padding: 0.9rem 1.6rem !important;
    max-width: 300px !important;
    margin: 1.2rem auto !important;   /* căn giữa */

    transition: 
        background-color 0.2s ease,
        color 0.2s ease,
        border-color 0.2s ease,
        transform 0.15s ease;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background: rgb(61 136 205) !important;
    color: rgb(255 255 255) !important;
    border-color: rgb(0 0 0) !important;
    transform: translateY(-1px);
    border: 2px solid rgb(255 255 255) !important;
}


.st-emotion-cache-gx6i9d p {
    color: rgb(0 0 0) !important;
}

@supports (scrollbar-color: auto) {
    * {
        scrollbar-color: rgba(250, 250, 250, 0.4)!important;
    }

    *:hover {
        scrollbar-color: rgba(250, 250, 250, 0.4) transparent;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ===============================
# 2. CONSTANTS / OPTIONS
# ===============================

HOUSE_DIR_OPTIONS = [
    'khác',
    'Đông - Bắc',
    'Bắc',
    'Đông - Nam',
    'Nam', 'Tây',
    'Tây - Bắc',
    'Tây - Nam',
    'Đông'
]

BALCONY_DIR_OPTIONS = [
    'khác',
    'Tây - Nam',
    'Nam',
    'Tây - Bắc',
    'Đông - Bắc',
    'Bắc',
    'Đông - Nam',
    'Đông',
    'Tây'
]

LEGAL_OPTIONS = [
    'Có sổ riêng',
    'khác',
    'Hợp đồng mua bán'
]

FURNITURE_OPTIONS = [
'cơ bản',
'đầy đủ',
'không nội thất',
'cao cấp',
'thô',
'đẹp',
'khác'
]


# ===============================
# 3. LOAD GADM (PROVINCE / DISTRICT)
# ===============================

@st.cache_resource
def load_gadm_2():
    shp_path = os.path.join("notebook", "maps", "vietnam", "gadm41_VNM_2.shp")
    gdf = gpd.read_file(shp_path)[["NAME_1", "NAME_2", "geometry"]]
    gdf = gdf.to_crs(epsg=4326)
    return gdf


gadm_gdf = load_gadm_2()
ALL_PROVINCES = sorted(gadm_gdf["NAME_1"].unique().tolist())


@st.cache_data
def get_districts_by_province():
    mapping = {}
    for _, row in gadm_gdf[["NAME_1", "NAME_2"]].drop_duplicates().iterrows():
        prov = row["NAME_1"]
        dist = row["NAME_2"]
        mapping.setdefault(prov, set()).add(dist)
    # convert to sorted lists
    return {prov: sorted(list(dists)) for prov, dists in mapping.items()}


PROV_DIST_MAP = get_districts_by_province()


# ===============================
# 4. GEOCODER
# ===============================

@st.cache_resource
def get_geocoder():
    return Nominatim(user_agent="house_price_streamlit_app")


GEOCODER = get_geocoder()


def normalize_text_ascii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text_nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text_nfd if unicodedata.category(ch) != "Mn")


def geocode_address(address: str):
    """
    Return (lat, lon, full_address) or (None, None, None).
    Uses hierarchical fallback: full address -> ward+district+city -> district+city
    """
    if not address:
        return None, None, None
    
    # Try full address first
    try:
        loc = GEOCODER.geocode(address, timeout=5)
        if loc:
            return loc.latitude, loc.longitude, loc.address
    except Exception:
        pass
    
    # Parse address to extract components
    address_parts = [part.strip() for part in address.split(',')]
    
    if len(address_parts) >= 3:
        # Try without street name (ward + district + city)
        fallback_1 = ', '.join(address_parts[1:])
        try:
            loc = GEOCODER.geocode(fallback_1, timeout=5)
            if loc:
                return loc.latitude, loc.longitude, loc.address
        except Exception:
            pass
    
    if len(address_parts) >= 2:
        # Try without street and ward (district + city)
        fallback_2 = ', '.join(address_parts[2:]) if len(address_parts) > 2 else ', '.join(address_parts[-2:])
        try:
            loc = GEOCODER.geocode(fallback_2, timeout=5)
            if loc:
                return loc.latitude, loc.longitude, loc.address
        except Exception:
            pass
    
    # Final fallback: try just the last part (usually city/province)
    if address_parts:
        try:
            loc = GEOCODER.geocode(address_parts[-1], timeout=5)
            if loc:
                return loc.latitude, loc.longitude, loc.address
        except Exception:
            pass
    
    return None, None, None


def province_district_from_point(lat: float, lon: float):
    """Use shapefile to map lat/lon to NAME_1, NAME_2."""
    if lat is None or lon is None:
        return None, None
    point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    joined = gpd.sjoin(point_gdf, gadm_gdf, how="left", predicate="within")
    if joined.empty:
        return None, None
    return joined.iloc[0]["NAME_1"], joined.iloc[0]["NAME_2"]


def parse_province_district_from_text(address: str):
    """
    Fallback: try to guess province/district by matching cleaned address text
    to shapefile NAME_1 / NAME_2 strings.
    """
    if not address:
        return None, None

    addr_norm = normalize_text_ascii(address).lower()

    # Try province first
    prov_match = None
    for prov in ALL_PROVINCES:
        if normalize_text_ascii(prov).lower() in addr_norm:
            prov_match = prov
            break

    # Try district within that province
    dist_match = None
    if prov_match:
        for dist in PROV_DIST_MAP[prov_match]:
            if normalize_text_ascii(dist).lower() in addr_norm:
                dist_match = dist
                break

    return prov_match, dist_match


def coerce_project_value(text: str) -> bool:
    """
    Project backend rule:
    - If user types something that clearly means True (yes/true/1) -> True
    - If empty or invalid -> False
    """
    if not text:
        return False
    s = text.strip().lower()
    if s in {"true", "yes", "y", "1", "project", "du an", "dự án"}:
        return True
    # Any other weird / unknown text counts as False as per requirement
    return False


# ===============================
# 5. UI
# ===============================

st.title("House Price Prediction")

st.markdown(
    "Enter the property information below. The model will use your inputs and "
    "nearby amenities to estimate a suitable price."
)

# -------- Address & location card --------
st.markdown("<div class='card-title'>Location</div>", unsafe_allow_html=True)

# Initialize session state for geocoding results
if 'geocoded_address' not in st.session_state:
    st.session_state.geocoded_address = None
if 'geocoded_lat' not in st.session_state:
    st.session_state.geocoded_lat = None
if 'geocoded_lon' not in st.session_state:
    st.session_state.geocoded_lon = None
if 'geocoded_NAME_1' not in st.session_state:
    st.session_state.geocoded_NAME_1 = None
if 'geocoded_NAME_2' not in st.session_state:
    st.session_state.geocoded_NAME_2 = None

# Add a button to trigger geocoding manually
col_addr_btn1, col_addr_btn2 = st.columns([3, 1])
with col_addr_btn1:
    address = st.text_input(
        "Full address",
        placeholder="e.g. 123 Dinh Cong Street, Dinh Cong Ward, Hoang Mai District, Hanoi",
    )
with col_addr_btn2:
    geocode_btn = st.button(" Geocode", key="geocode_button")

# Only geocode when button is clicked
auto_lat, auto_lon, full_addr = None, None, None
auto_NAME_1, auto_NAME_2 = None, None

if geocode_btn and address:
    with st.spinner("Geocoding address..."):
        auto_lat, auto_lon, full_addr = geocode_address(address)
        if auto_lat is not None and auto_lon is not None:
            # Store in session state
            st.session_state.geocoded_address = address
            st.session_state.geocoded_lat = auto_lat
            st.session_state.geocoded_lon = auto_lon
            
            # First, try spatial join to shapefile
            prov, dist = province_district_from_point(auto_lat, auto_lon)
            auto_NAME_1, auto_NAME_2 = prov, dist
            
            # If that fails, fall back to text parsing
            if auto_NAME_1 is None or auto_NAME_2 is None:
                text_prov, text_dist = parse_province_district_from_text(address)
                auto_NAME_1 = auto_NAME_1 or text_prov
                auto_NAME_2 = auto_NAME_2 or text_dist
            
            # Store province/district in session state
            st.session_state.geocoded_NAME_1 = auto_NAME_1
            st.session_state.geocoded_NAME_2 = auto_NAME_2
        else:
            # Geocoding failed – try text parsing only
            text_prov, text_dist = parse_province_district_from_text(address)
            auto_NAME_1, auto_NAME_2 = text_prov, text_dist
            st.session_state.geocoded_NAME_1 = auto_NAME_1
            st.session_state.geocoded_NAME_2 = auto_NAME_2
            st.warning("Could not geocode this address; province/district inferred from text if possible.")

# Use stored values from session state if available
if st.session_state.geocoded_address == address:
    auto_lat = st.session_state.geocoded_lat
    auto_lon = st.session_state.geocoded_lon
    auto_NAME_1 = st.session_state.geocoded_NAME_1
    auto_NAME_2 = st.session_state.geocoded_NAME_2
    if auto_lat and auto_lon:
        st.info(f"Using geocoded coordinates: {auto_lat:.6f}, {auto_lon:.6f}")

# Lat / lon inputs (optional, under address). Default to auto values if any.
col_lat, col_lon = st.columns(2)
with col_lat:
    lat = st.number_input(
        "Latitude (optional)",
        value=float(auto_lat) if auto_lat is not None else 0.0,
        format="%.6f",
    )
with col_lon:
    lon = st.number_input(
        "Longitude (optional)",
        value=float(auto_lon) if auto_lon is not None else 0.0,
        format="%.6f",
    )

if auto_lat is None and auto_lon is None and (lat == 0.0 and lon == 0.0):
    st.caption("If left at 0, the model will not be able to compute spatial features.")

# Province / district dropdowns
default_prov_index = 0
if auto_NAME_1 and auto_NAME_1 in ALL_PROVINCES:
    default_prov_index = ALL_PROVINCES.index(auto_NAME_1)

NAME_1 = st.selectbox("Province / City (NAME_1)", ALL_PROVINCES, index=default_prov_index)

district_options = PROV_DIST_MAP.get(NAME_1, [])
default_dist_index = 0
if auto_NAME_2 and auto_NAME_2 in district_options:
    default_dist_index = district_options.index(auto_NAME_2)

NAME_2 = st.selectbox("District / County (NAME_2)", district_options, index=default_dist_index if district_options else 0)

st.markdown("</div>", unsafe_allow_html=True)


# -------- Property details card --------
st.markdown("<div class='card-title'>Property details</div>", unsafe_allow_html=True)

area = st.number_input("Area (m²)", min_value=1.0, value=60.0, step=1.0)

col_bed, col_bath = st.columns(2)
with col_bed:
    n_bed = st.number_input("Bedrooms", min_value=0, value=2, step=1)
with col_bath:
    n_bath = st.number_input("Bathrooms", min_value=0, value=2, step=1)

house_dir = st.selectbox("House direction", HOUSE_DIR_OPTIONS, index=0)
balcony_dir = st.selectbox("Balcony direction", BALCONY_DIR_OPTIONS, index=0)
legal_c = st.selectbox("Legal status", LEGAL_OPTIONS, index=0)
furniture_c = st.selectbox("Furniture", FURNITURE_OPTIONS, index=0)

project_text = st.text_input("Project (text; empty or invalid → treated as False)", value="")

st.markdown("</div>", unsafe_allow_html=True)


# -------- Predict button --------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(" Predict price", use_container_width=True, type="primary")

if predict_button:
    # Validate lat/lon: if both zero and no auto geocode, show error
    if (auto_lat is None and auto_lon is None) and (lat == 0.0 and lon == 0.0):
        st.error("Please provide a valid address so that latitude and longitude can be determined.")
    else:
        # Prefer explicit user-entered lat/lon if non-zero; otherwise use auto
        final_lat = lat if not np.isclose(lat, 0.0) else (auto_lat or 0.0)
        final_lon = lon if not np.isclose(lon, 0.0) else (auto_lon or 0.0)
        
        if np.isclose(final_lat, 0.0) or np.isclose(final_lon, 0.0):
            st.error("Latitude/longitude are required for spatial features; please check your address.")
        else:
            project_bool = coerce_project_value(project_text)
            
            input_df = pd.DataFrame(
                [
                    {
                        "area": area,
                        "lat": final_lat,
                        "lon": final_lon,
                        "NAME_1": NAME_1,
                        "NAME_2": NAME_2,
                        "house_dir": house_dir,
                        "balcony_dir": balcony_dir,
                        "legal_c": legal_c,
                        "furniture_c": furniture_c,
                        "n_Bedrooms": int(n_bed),
                        "n_Bathrooms": int(n_bath),
                        "project": project_bool,
                    }
                ]
            )
            
            result = predict_price(input_df)
            price_million = result["predicted_price"].iloc[0]
            price_billion = price_million / 1000  # Convert million to billion
            
            # Display predicted price prominently
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'> Predicted Price: {price_billion:.2f} tỷ VND</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Display input summary
            st.markdown("###  INFO SUMMARY")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Address:** {address}")
                st.write(f"**Area:** {area} m²")
                st.write(f"**Bedrooms:** {n_bed}")
                st.write(f"**Bathrooms:** {n_bath}")
                st.write(f"**House Direction:** {house_dir}")
                st.write(f"**Balcony Direction:** {balcony_dir}")
            with col_b:
                st.write(f"**Province:** {NAME_1}")
                st.write(f"**District:** {NAME_2}")
                st.write(f"**Legal Status:** {legal_c}")
                st.write(f"**Furniture:** {furniture_c}")
                st.write(f"**Project:** {project_bool}")
                st.write(f"**Location:** Latitude {final_lat:.6f}, Longitude {final_lon:.6f}")