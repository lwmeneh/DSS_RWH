# ============================================================
# GeoPond-DSS  ·  v2.0
# Geospatial Pond Siting and Preliminary Design
# Decision Support System
#
# ICAR Research Complex for NEH Region, Umiam, Meghalaya
#
# Web-GIS DSS: field capture → land-use verification →
# hydrology → multi-criteria site suitability → pond design →
# geology / seepage screening → cost estimate → report
# ============================================================

from pathlib import Path
import base64
import hashlib
import html
import io
import json
import math
import re
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.transform import xy
from rasterio.warp import reproject, Resampling

import streamlit as st

import folium
from folium.plugins import (
    Draw,
    Fullscreen,
    LocateControl,
    MeasureControl,
    MiniMap,
    MousePosition,
)

try:  # folium >= 0.14
    from folium.plugins import Geocoder
except Exception:  # pragma: no cover
    Geocoder = None

from streamlit_folium import st_folium

from shapely.geometry import shape, mapping, Point, box
from shapely.ops import unary_union
from shapely import affinity

from pyproj import Transformer

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoPond-DSS | ICAR NEH",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS & CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "cache"
ASSETS_DIR = BASE_DIR / "assets"
ICAR_LOGO = ASSETS_DIR / "icar_logo.png"

WORKING_CRS = "EPSG:32646"
MAP_CENTER = [25.88, 91.88]
MAP_ZOOM = 12
APP_VERSION = "2.0"

# --- Rasters -------------------------------------------------
SLOPE_RASTER = CACHE_DIR / "03_Slope.tif"
RAINFALL_RASTER = CACHE_DIR / "04_Rainfall.tif"
ELEVATION_RASTER = CACHE_DIR / "05_Elevation.tif"
DRAINAGE_DENSITY_RASTER = CACHE_DIR / "06_DrainageDensity.tif"
LINEAMENT_DENSITY_RASTER = CACHE_DIR / "08_LineamentDensity.tif"
TWI_RASTER = CACHE_DIR / "10_TWI.tif"
FLOW_ACCUMULATION_RASTER = CACHE_DIR / "flow_accumulation.tif"
CN_RASTER = CACHE_DIR / "CN_II.tif"
MEAN_RUNOFF_RASTER = CACHE_DIR / "mean_runoff_mm.tif"
DEPENDABLE_RUNOFF_RASTER = CACHE_DIR / "dependable_runoff_mm.tif"
RUNOFF_COEFFICIENT_RASTER = CACHE_DIR / "runoff_coefficient.tif"

# --- Vectors -------------------------------------------------
BOUNDARY_GPKG = CACHE_DIR / "boundary_utm46.gpkg"
STREAMS_GPKG = CACHE_DIR / "streams_utm46.gpkg"
LULC_2022_GPKG = CACHE_DIR / "lulc_2021_10k_clean_utm46.gpkg"

# LULC 2015: either a column in the 2022 layer, or a separate file in data/cache.
# Put your exact file name first in this list if it is not detected automatically.
LULC_2015_FILES = [
    CACHE_DIR / "RiBhoi_LULC_2015_16_UTM46N.gpkg",      # district subset (lighter)
    CACHE_DIR / "Meghalaya_LULC_2015_16_UTM46N.gpkg",   # state-wide NRSC 2015-16
    CACHE_DIR / "lulc_2015_10k_clean_utm46.gpkg",
    CACHE_DIR / "lulc_2015_utm46.gpkg",
    CACHE_DIR / "LULC_2015.gpkg",
    CACHE_DIR / "LULC_2015.shp",
    CACHE_DIR / "LULC_2015.geojson",
]
LULC_2015_COLUMNS = ["LULC_2015", "LULC2015", "LU_2015", "LULC_15", "Class_2015", "LULC15"]
LULC_GENERIC_COLUMNS = ["DESCR_2", "DESCR2", "LULC", "LULC_CLASS", "Class", "Class_Name", "LU_Class", "Landuse",
                        "Land_Use", "LANDUSE", "Description", "Legend", "Level_II", "Level2"]
PROJECTED_LULC_2025_SHP = CACHE_DIR / "2025_Projected_LULC.shp"
SOIL_GPKG = CACHE_DIR / "Soil_Meghalaya_50k.gpkg"
SOIL_SHP = CACHE_DIR / "Soil_Meghalaya_50k.shp"
GEOMORPHOLOGY_GPKG = CACHE_DIR / "Geomorphology.gpkg"
LITHOLOGY_GPKG = CACHE_DIR / "Lithology.gpkg"
SHEAR_ZONE_GPKG = CACHE_DIR / "Shear_Zone.gpkg"
SOIL_C_HORIZON_GPKG = CACHE_DIR / "Soil_C_Horizon.gpkg"

ESRI_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_TOPO = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
)
ESRI_LABELS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)

LAND_USE_OPTIONS = [
    "Crop land",
    "Fallow land",
    "Plantation",
    "Shifting Cultivation",
    "Scrub land",
    "Barren land",
    "Forest Open",
    "Forest Moderately Dense",
    "Forest Dense",
    "Built up",
    "Waterbody",
    "River/Stream",
]

LULC_COLORS = {
    "Crop land": "#f4d03f",
    "Fallow land": "#e8c872",
    "Plantation": "#a3d977",
    "Shifting Cultivation": "#e59866",
    "Scrub land": "#d7bde2",
    "Barren land": "#b0a8a0",
    "Forest Open": "#58d68d",
    "Forest Moderately Dense": "#28b463",
    "Forest Dense": "#196f3d",
    "Built up": "#e74c3c",
    "Waterbody": "#3498db",
    "River/Stream": "#5dade2",
}

# Land uses where a farm pond is normally not appropriate
RESTRICTED_LULC = {"Forest Dense", "Built up", "Waterbody", "River/Stream"}

# Agricultural classes, and classes that agricultural land is often mapped as
# in a single-date image (fallow, jhum regrowth, bush fallow).
AGRICULTURAL_LULC = {"Crop land", "Fallow land", "Plantation", "Shifting Cultivation"}
TRANSITIONAL_LULC = {"Scrub land", "Forest Open", "Fallow land"}

# Representative SCS curve numbers (AMC II, HSG C; after USDA TR-55 / NEH-4).
# Only the DIFFERENCE between classes is used, to shift the mapped CN raster
# to the land use actually observed in the field.
CN_BY_LULC = {
    "Crop land": 85, "Fallow land": 86, "Plantation": 76, "Shifting Cultivation": 86,
    "Scrub land": 73, "Barren land": 88, "Forest Open": 77, "Forest Moderately Dense": 73, "Forest Dense": 70,
    "Built up": 90, "Waterbody": 98, "River/Stream": 98,
}

# Weights of the composite suitability index (sum = 1.0)
SUITABILITY_WEIGHTS = {
    "Slope": 0.20,
    "Wetness (TWI)": 0.15,
    "Water availability": 0.20,
    "Seepage": 0.20,
    "Terrain / landform": 0.15,
    "Shear-zone distance": 0.10,
}


# ============================================================
# 3. UI THEME (CSS)
# ============================================================

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --ink:#0f2a25; --ink-2:#3d5a53; --muted:#7a8f89;
  --line:#dbe7e3; --card:#ffffff; --bg:#f3f7f6;
  --brand:#067a5f; --brand-2:#0aa37b; --brand-dark:#04382f;
  --water:#1478b8; --amber:#e39b16; --red:#d64045; --green:#18a058;
}
html, body, [class*="css"]{font-family:Inter,"Segoe UI",Arial,sans-serif;}
#MainMenu, footer{visibility:hidden;}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(1200px 500px at 85% -10%, #d9efe8 0%, transparent 60%),
             radial-gradient(900px 400px at -10% 0%, #e2eef8 0%, transparent 55%),
             var(--bg);
}
[data-testid="stHeader"]{background:rgba(255,255,255,.55);backdrop-filter:blur(10px);}
.block-container{max-width:1820px;padding:.4rem .9rem 3rem .9rem;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#05332b 0%,#0a4a3e 60%,#0d5546 100%);
}
[data-testid="stSidebar"] *{color:#e7f3ef;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] *{color:#0f2a25 !important;}
[data-testid="stSidebar"] [data-testid="stExpander"]{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;
}
[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.35);}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.20);border-color:#fff;}
[data-testid="stSidebar"] .stButton>button p{color:#ffffff !important;}
.sb-brand{display:flex;gap:10px;align-items:center;padding:4px 2px 12px;border-bottom:1px solid rgba(255,255,255,.14);margin-bottom:10px;}
.sb-brand b{font-size:18px;letter-spacing:.01em;}
.sb-brand span{display:block;font-size:11px;opacity:.7;}
.sb-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;}

/* ---------- header ---------- */
.gp-header{
  position:relative;overflow:hidden;
  background:linear-gradient(115deg,#032d26 0%,#05634c 45%,#0a9a72 100%);
  padding:20px 26px;color:#fff;border-radius:18px;
  box-shadow:0 10px 30px rgba(3,45,38,.25);
}
.gp-header:after{
  content:"";position:absolute;right:-60px;top:-60px;width:320px;height:320px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,255,255,.16),transparent 65%);
}
.gp-header-row{display:flex;align-items:center;gap:20px;position:relative;z-index:1;flex-wrap:wrap;}
.gp-header-row>div:nth-child(2){flex:1 1 380px;min-width:0;}
.gp-logo img{width:84px;height:84px;object-fit:contain;background:#fff;border-radius:14px;padding:6px;}
.gp-org{font-size:10.5px;font-weight:800;letter-spacing:.14em;opacity:.78;}
.gp-institute{font-size:15px;font-weight:600;margin-top:2px;opacity:.95;}
.gp-title{font-size:32px;font-weight:800;line-height:1.05;margin-top:6px;letter-spacing:-.01em;}
.gp-title small{font-size:13px;font-weight:600;background:rgba(255,255,255,.18);padding:3px 9px;border-radius:30px;margin-left:8px;vertical-align:middle;}
.gp-subtitle{font-size:14px;margin-top:4px;opacity:.9;}
.gp-chips{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;}
.gp-chip{padding:4px 10px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.18);border-radius:40px;font-size:11px;font-weight:600;}
.gp-head-stats{margin-left:auto;display:grid;grid-template-columns:repeat(3,auto);gap:8px;position:relative;z-index:1;}
.gp-head-stat{background:rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:8px 14px;text-align:right;}
.gp-head-stat span{display:block;font-size:10px;letter-spacing:.08em;opacity:.7;}
.gp-head-stat b{font-size:17px;}

/* ---------- stepper ---------- */
.gp-steps{display:flex;gap:0;margin:12px 0 14px;background:#fff;border:1px solid var(--line);border-radius:14px;padding:6px;overflow-x:auto;box-shadow:0 2px 10px rgba(15,42,37,.05);}
.gp-step{flex:1;min-width:150px;display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:10px;}
.gp-step .n{width:28px;height:28px;flex:0 0 28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;background:#e8f0ee;color:var(--muted);}
.gp-step .t{font-size:12.5px;font-weight:700;color:var(--muted);line-height:1.2;}
.gp-step .t small{display:block;font-weight:500;font-size:10.5px;}
.gp-step.done .n{background:var(--brand);color:#fff;}
.gp-step.done .t{color:var(--ink);}
.gp-step.active{background:linear-gradient(90deg,#e6f6f0,#f5fbf9);}
.gp-step.active .n{background:var(--water);color:#fff;box-shadow:0 0 0 4px rgba(20,120,184,.15);}
.gp-step.active .t{color:var(--water);}

/* ---------- section titles ---------- */
.gp-section{display:flex;align-items:center;gap:10px;margin:18px 0 10px;}
.gp-section .ic{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--brand),var(--brand-2));display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 4px 12px rgba(6,122,95,.25);}
.gp-section h3{margin:0;font-size:19px;font-weight:800;color:var(--ink);}
.gp-section p{margin:0;font-size:12px;color:var(--muted);}

/* ---------- cards ---------- */
.gp-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 16px;margin-bottom:10px;box-shadow:0 3px 14px rgba(15,42,37,.05);}
.gp-card-title{font-size:13px;font-weight:800;color:var(--brand-dark);margin-bottom:9px;padding-bottom:7px;border-bottom:1px solid #edf3f1;letter-spacing:.01em;}
.gp-hint{background:linear-gradient(120deg,#e8f3fb,#f6fbfd);color:#22556c;border:1px solid #d1e6f1;border-left:4px solid var(--water);border-radius:12px;padding:10px 13px;font-size:12.5px;line-height:1.6;}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.kv div{background:#f4f8f7;border-radius:10px;padding:8px 10px;}
.kv span{display:block;font-size:10.5px;color:var(--muted);}
.kv b{display:block;font-size:14px;color:var(--ink);margin-top:2px;}
.kv4{grid-template-columns:repeat(4,minmax(0,1fr));}
.mono{font-family:"JetBrains Mono",monospace;}

/* ---------- verdict ---------- */
.verdict{border-radius:18px;padding:18px 20px;color:#fff;box-shadow:0 10px 28px rgba(0,0,0,.14);height:100%;}
.verdict .k{font-size:10.5px;letter-spacing:.14em;font-weight:800;opacity:.8;}
.verdict .v{font-size:30px;font-weight:800;margin-top:2px;line-height:1.1;}
.verdict .s{font-size:13px;opacity:.92;margin-top:6px;line-height:1.5;}
.verdict ul{margin:10px 0 0 0;padding-left:18px;font-size:12.5px;line-height:1.65;}
.v-high{background:linear-gradient(130deg,#065f46,#10a46e);}
.v-good{background:linear-gradient(130deg,#0b5f7a,#1b9bc4);}
.v-marg{background:linear-gradient(130deg,#8a5a00,#e39b16);}
.v-poor{background:linear-gradient(130deg,#7c1d22,#d64045);}

.badge{display:inline-block;padding:3px 10px;border-radius:30px;font-size:11px;font-weight:800;letter-spacing:.04em;}
.b-low{background:#dcf5e8;color:#0d6b3c;} .b-mod{background:#fff1d6;color:#8a5a00;}
.b-high{background:#fde0e1;color:#9b1c22;} .b-unk{background:#eceff1;color:#546e7a;}

/* ---------- streamlit widgets ---------- */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:12px 14px;box-shadow:0 2px 10px rgba(15,42,37,.05);transition:transform .15s ease,box-shadow .15s ease;}
[data-testid="stMetric"]:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(15,42,37,.10);}
[data-testid="stMetricLabel"]{color:var(--muted);font-weight:600;}
[data-testid="stMetricValue"]{color:var(--ink);font-weight:800;font-size:1.45rem;}
.stButton>button,.stDownloadButton>button{width:100%;min-height:46px;border-radius:11px;font-size:14px;font-weight:700;}
.stButton>button[kind="primary"]{color:#fff;border:0;background:linear-gradient(90deg,#067a5f,#0aa37b);box-shadow:0 6px 18px rgba(6,122,95,.28);}
.stButton>button[kind="primary"]:hover{filter:brightness(1.06);}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"]{gap:4px;padding:5px;background:#fff;border:1px solid var(--line);border-radius:13px;overflow-x:auto;}
.stTabs [data-baseweb="tab"]{border-radius:9px;height:40px;font-weight:650;padding:0 14px;white-space:nowrap;}
.stTabs [aria-selected="true"]{background:linear-gradient(90deg,#e3f4ee,#eff9f6);color:var(--brand) !important;}
iframe{border-radius:14px;}

.legend-bar{height:10px;border-radius:6px;background:linear-gradient(90deg,#d73027,#fc8d59,#fee08b,#91cf60,#1a9850);}
.legend-row{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);margin-top:3px;}

@media only screen and (max-width:1250px){
  .gp-head-stats{margin-left:0;}
  .gp-title{font-size:27px;}
}
@media only screen and (max-width:900px){
  .block-container{padding:.2rem .3rem 2rem;}
  .gp-header{padding:14px 12px;border-radius:14px;}
  .gp-header-row{flex-direction:column;text-align:center;gap:8px;}
  .gp-head-stats{margin:6px auto 0;}
  .gp-chips{justify-content:center;}
  .gp-title{font-size:25px;}
  .kv4{grid-template-columns:repeat(2,minmax(0,1fr));}
  .gp-step{min-width:120px;}
}
"""

# Inject as ONE <style> block with no blank lines: Streamlit's Markdown parser
# ends an HTML block at a blank line and would print the rest as text.
_css_compact = "\n".join(line for line in APP_CSS.splitlines() if line.strip())
st.markdown(f"<style>\n{_css_compact}\n</style>", unsafe_allow_html=True)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def image_to_base64(path):
    try:
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None


def safe_text(value, default="Unknown"):
    if value is None:
        return default
    if isinstance(value, float) and not np.isfinite(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def norm_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def first_matching_column(columns, candidates):
    lookup = {norm_name(c): c for c in columns}
    for candidate in candidates:
        key = norm_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def fmt(value, pattern="{:,.0f}", default="—"):
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return default
        return pattern.format(value)
    except Exception:
        return default


def esc(value):
    return html.escape(safe_text(value, ""))


def section(icon, title, subtitle=""):
    st.markdown(
        f'<div class="gp-section"><div class="ic">{icon}</div>'
        f'<div><h3>{title}</h3><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(level):
    level = safe_text(level).upper()
    cls = {"LOW": "b-low", "MODERATE": "b-mod", "HIGH": "b-high"}.get(level, "b-unk")
    return f'<span class="badge {cls}">{level}</span>'


# ============================================================
# 5. REQUIRED GIS FILE CHECK
# ============================================================

REQUIRED_FILES = [
    SLOPE_RASTER, RAINFALL_RASTER, ELEVATION_RASTER, DRAINAGE_DENSITY_RASTER,
    LINEAMENT_DENSITY_RASTER, TWI_RASTER, FLOW_ACCUMULATION_RASTER, CN_RASTER,
    MEAN_RUNOFF_RASTER, DEPENDABLE_RUNOFF_RASTER, RUNOFF_COEFFICIENT_RASTER,
    BOUNDARY_GPKG, STREAMS_GPKG,
]

missing_files = [str(p.relative_to(BASE_DIR)) for p in REQUIRED_FILES if not p.exists()]

if missing_files:
    st.error("GeoPond-DSS GIS database is incomplete. The following layers are missing:")
    st.code("\n".join(missing_files))
    st.stop()


# ============================================================
# 6. VECTOR LOADING (cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vector_utm(path_string, fallback_path_string=None):
    candidates = [Path(path_string)]
    if fallback_path_string:
        candidates.append(Path(fallback_path_string))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.empty:
            return gdf
        if gdf.crs is None:
            return None
        try:
            invalid = ~gdf.geometry.is_valid
            if invalid.any():
                gdf = gdf.copy()
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
        except Exception:
            pass
        if gdf.crs.to_epsg() != 32646:
            gdf = gdf.to_crs(WORKING_CRS)
        return gdf
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def simplified_wgs(_gdf, key, tolerance=15.0):
    """Simplify in metres, then project to WGS84 for light-weight web display."""
    if _gdf is None or _gdf.empty:
        return None
    g = _gdf[["geometry"]].copy()
    g["geometry"] = g.geometry.simplify(tolerance, preserve_topology=True)
    return g.to_crs("EPSG:4326")


boundary_utm = load_vector_utm(str(BOUNDARY_GPKG))
streams_utm = load_vector_utm(str(STREAMS_GPKG))
lulc_2022_utm = load_vector_utm(str(LULC_2022_GPKG))
projected_lulc_2025_utm = load_vector_utm(str(PROJECTED_LULC_2025_SHP))
soil_gdf = load_vector_utm(str(SOIL_GPKG), str(SOIL_SHP))
geomorph_gdf = load_vector_utm(str(GEOMORPHOLOGY_GPKG))
lithology_gdf = load_vector_utm(str(LITHOLOGY_GPKG))
shear_gdf = load_vector_utm(str(SHEAR_ZONE_GPKG))
c_horizon_gdf = load_vector_utm(str(SOIL_C_HORIZON_GPKG))

boundary_wgs = (
    boundary_utm.to_crs("EPSG:4326")
    if boundary_utm is not None and not boundary_utm.empty
    else None
)


def _ok(g):
    return g is not None and not g.empty


def _match_column(columns, preferred):
    lookup = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in columns}
    for cand in preferred:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in lookup:
            return lookup[key]
    return None


@st.cache_resource(show_spinner=False)
def load_vector_utm_clipped(path_string, bounds_utm):
    """Read only features inside a UTM bounding box (state-wide layers → district)."""
    try:
        clip_box = gpd.GeoSeries([box(*bounds_utm)], crs=WORKING_CRS)
        gdf = gpd.read_file(path_string, bbox=clip_box)  # reprojected to the file CRS by geopandas
        if gdf.empty or gdf.crs is None:
            return None
        try:
            invalid = ~gdf.geometry.is_valid
            if invalid.any():
                gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
        except Exception:
            pass
        if gdf.crs.to_epsg() != 32646:
            gdf = gdf.to_crs(WORKING_CRS)
        return gdf
    except Exception:
        return load_vector_utm(path_string)


def _district_bounds(margin=3000.0):
    if not _ok(boundary_utm):
        return None
    minx, miny, maxx, maxy = boundary_utm.total_bounds
    return (float(minx - margin), float(miny - margin), float(maxx + margin), float(maxy + margin))


def _find_2015_source():
    """Locate LULC 2015: a column in the 2022 layer, else a separate file in data/cache."""
    if _ok(lulc_2022_utm):
        col = _match_column(lulc_2022_utm.columns, LULC_2015_COLUMNS)
        if col:
            return lulc_2022_utm, col, f"{LULC_2022_GPKG.name} → {col}"
    files = [p for p in LULC_2015_FILES if p.exists()]
    for pattern in ("*2015*.gpkg", "*2015*.shp", "*2015*.geojson"):
        files += [p for p in sorted(CACHE_DIR.glob(pattern)) if p not in files]
    bounds = _district_bounds()
    for path in files:
        gdf = (load_vector_utm_clipped(str(path), bounds) if bounds
               else load_vector_utm(str(path)))
        if not _ok(gdf):
            continue
        col = (_match_column(gdf.columns, LULC_2015_COLUMNS)
               or _match_column(gdf.columns, LULC_GENERIC_COLUMNS))
        if col:
            return gdf, col, f"{path.name} → {col}"
    return None, None, "not found"


# year → {"gdf", "column", "source"}
LULC_SOURCES = {}
if _ok(lulc_2022_utm):
    _c22 = _match_column(lulc_2022_utm.columns, ["LULC_2022", "LULC2022"] + LULC_GENERIC_COLUMNS)
    if _c22:
        LULC_SOURCES[2022] = {"gdf": lulc_2022_utm, "column": _c22,
                              "source": f"{LULC_2022_GPKG.name} → {_c22}"}
_g15, _c15, _s15 = _find_2015_source()
if _g15 is not None:
    LULC_SOURCES[2015] = {"gdf": _g15, "column": _c15, "source": _s15}


LAYER_STATUS = {
    "District boundary": _ok(boundary_utm),
    "Drainage network": _ok(streams_utm),
    "LULC 2015": 2015 in LULC_SOURCES,
    "LULC 2022": 2022 in LULC_SOURCES,
    "Projected LULC 2025": _ok(projected_lulc_2025_utm),
    "Soil (1:50k)": _ok(soil_gdf),
    "Geomorphology": _ok(geomorph_gdf),
    "Lithology": _ok(lithology_gdf),
    "Shear zones": _ok(shear_gdf),
    "Soil C-horizon": _ok(c_horizon_gdf),
}
RASTER_COUNT = 11


# ============================================================
# 7. SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "field_geometry": None,
    "last_drawing": None,
    "assessment": None,
    "current_lulc": None,
    "map_center": None,
    "map_nonce": 0,
}
for _k, _v in SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_assessment():
    st.session_state["field_geometry"] = None
    st.session_state["last_drawing"] = None
    st.session_state["assessment"] = None
    st.session_state["current_lulc"] = None
    st.session_state["map_nonce"] += 1  # forces a fresh map (clears drawings)


# ============================================================
# 8. GEOMETRY HELPERS
# ============================================================

def geometry_gdf(geometry_geojson, crs="EPSG:4326"):
    return gpd.GeoDataFrame({"id": [1]}, geometry=[shape(geometry_geojson)], crs=crs)


def to_utm_geometry(geometry_geojson):
    return geometry_gdf(geometry_geojson).to_crs(WORKING_CRS).geometry.iloc[0]


def field_area_ha(geometry_geojson):
    return float(to_utm_geometry(geometry_geojson).area) / 10000.0


def field_perimeter_m(geometry_geojson):
    return float(to_utm_geometry(geometry_geojson).length)


def field_centroid_wgs(geometry_geojson):
    c = shape(geometry_geojson).centroid
    return c.y, c.x


def as_plain_geojson(geom):
    return json.loads(json.dumps(mapping(geom)))


def get_last_polygon(map_output):
    if not map_output:
        return None
    drawings = map_output.get("all_drawings") or []
    polygon = None
    for drawing in drawings:
        if not drawing:
            continue
        geom = drawing.get("geometry", {})
        if geom.get("type") in ("Polygon", "MultiPolygon"):
            polygon = geom
    return polygon


def parse_uploaded_boundary(uploaded):
    """Read a GeoJSON / KML / zipped Shapefile / GPKG field boundary → GeoJSON (WGS84)."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / uploaded.name
        path.write_bytes(uploaded.getvalue())
        if suffix == ".zip":
            gdf = gpd.read_file(f"zip://{path}")
        elif suffix == ".kml":
            try:
                gdf = gpd.read_file(path, driver="KML")
            except Exception:
                gdf = gpd.read_file(path)
        else:
            gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("The file contains no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("No polygon features found in the file.")
    geom = unary_union(gdf.to_crs("EPSG:4326").geometry.tolist())
    if not geom.is_valid:
        geom = geom.buffer(0)
    return as_plain_geojson(geom)


# ============================================================
# 9. RASTER HELPERS
# ============================================================

def raster_mean(raster_path, field_geojson):
    polygon = geometry_gdf(field_geojson)
    with rasterio.open(raster_path) as src:
        geom = polygon.to_crs(src.crs).geometry.iloc[0]
        values = np.array([], dtype=float)
        try:
            clipped, _ = rio_mask(
                src, [mapping(geom)], crop=True, indexes=1, filled=False, all_touched=True
            )
            values = clipped.compressed() if np.ma.isMaskedArray(clipped) else clipped.reshape(-1)
            values = values.astype(float)
            values = values[np.isfinite(values)]
            if src.nodata is not None:
                values = values[values != src.nodata]
        except Exception:
            pass

        if len(values) == 0:
            centroid = geom.centroid
            try:
                sample = next(src.sample([(centroid.x, centroid.y)], indexes=1, masked=True))
                sample = np.ma.filled(np.ma.asarray(sample, dtype=float), np.nan).reshape(-1)
                values = sample[np.isfinite(sample)]
            except Exception:
                pass

        return float(np.mean(values)) if len(values) else np.nan


def sample_raster(raster_path, x_utm, y_utm):
    """Sample a raster at a point given in WORKING_CRS."""
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs(WORKING_CRS, src.crs, always_xy=True)
        rx, ry = transformer.transform(x_utm, y_utm)
        try:
            value = next(src.sample([(rx, ry)], indexes=1, masked=True))[0]
            if np.ma.is_masked(value):
                return np.nan
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except Exception:
            return np.nan


def field_reference_grid(field_geojson):
    polygon = geometry_gdf(field_geojson)
    with rasterio.open(ELEVATION_RASTER) as src:
        geom = polygon.to_crs(src.crs).geometry.iloc[0]
        data, transform = rio_mask(
            src, [mapping(geom)], crop=True, indexes=1, filled=False, all_touched=True
        )
        if np.ma.isMaskedArray(data):
            arr = data.filled(np.nan).astype(float)
            valid = ~np.ma.getmaskarray(data)
        else:
            arr = data.astype(float)
            valid = np.isfinite(arr)
        if src.nodata is not None:
            valid &= arr != src.nodata
        return arr, valid, transform, src.crs


def raster_to_reference(raster_path, reference_shape, reference_transform, reference_crs):
    destination = np.full(reference_shape, np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        source = src.read(1).astype(np.float64)
        if src.nodata is not None:
            source[source == src.nodata] = np.nan
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination


def normalize_array(array, valid_mask):
    values = array[valid_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(array, dtype=float)
    low, high = np.percentile(values, 5), np.percentile(values, 95)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array, dtype=float)
    return np.clip((array - low) / (high - low), 0, 1)


def suitability_rgba(surface, inside_mask):
    """Map a suitability surface to an RGBA image (red → yellow → green)."""
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([
        [215, 48, 39], [252, 141, 89], [254, 224, 139], [145, 207, 96], [26, 152, 80],
    ], dtype=float)
    vals = surface[inside_mask & np.isfinite(surface)]
    rgba = np.zeros(surface.shape + (4,), dtype=np.uint8)
    if vals.size == 0:
        return rgba
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    norm = np.zeros_like(surface) if hi <= lo else (surface - lo) / (hi - lo)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0, 1)
    for ch in range(3):
        rgba[..., ch] = np.interp(norm, stops, colors[:, ch]).astype(np.uint8)
    rgba[..., 3] = np.where(inside_mask & np.isfinite(surface), 200, 0).astype(np.uint8)
    return rgba


# ============================================================
# 10. VECTOR LOOKUP HELPERS
# ============================================================

def point_record(gdf, x, y):
    """Return the feature covering a point (coordinates in WORKING_CRS)."""
    if gdf is None or gdf.empty:
        return None
    point = Point(x, y)
    try:
        indices = list(gdf.sindex.intersection(point.bounds))
        candidate = gdf.iloc[indices] if indices else gdf.iloc[0:0]
    except Exception:
        candidate = gdf
    if candidate.empty:
        return None
    try:
        hit = candidate[candidate.geometry.covers(point)]
    except Exception:
        hit = candidate[candidate.intersects(point)]
    return None if hit.empty else hit.iloc[0]


def nearest_distance_to_layer(gdf, x, y):
    if gdf is None or gdf.empty:
        return None
    distances = gdf.geometry.distance(Point(x, y))
    return None if distances.empty else float(distances.min())


def extract_record_fields(record, mapping_dict):
    if record is None:
        return {key: "Unknown" for key in mapping_dict}
    output = {}
    columns = list(record.index)
    for output_name, candidates in mapping_dict.items():
        column = first_matching_column(columns, candidates)
        output[output_name] = safe_text(record[column]) if column else "Unknown"
    return output


def local_subset(gdf, area_geom_utm):
    """Features of a UTM layer intersecting an area (UTM)."""
    if gdf is None or gdf.empty:
        return None
    try:
        ids = list(gdf.sindex.intersection(area_geom_utm.bounds))
        sub = gdf.iloc[ids] if ids else gdf.iloc[0:0]
    except Exception:
        sub = gdf
    sub = sub[sub.intersects(area_geom_utm)]
    return sub if not sub.empty else None


GEOMORPH_FIELDS = ["legend_sho", "description", "descriptio", "geomorphology", "landform"]


# ============================================================
# 11. LULC
# ============================================================

def canonical_lulc_name(value):
    """Map heterogeneous LULC legends (2015 / 2022 / NRSC Level-II) to one class list."""
    text = safe_text(value)
    key = norm_name(text)
    exact = {
        "cropland": "Crop land", "agriculture": "Crop land", "agriculturalland": "Crop land",
        "plantation": "Plantation",
        "shiftingcultivation": "Shifting Cultivation", "jhum": "Shifting Cultivation",
        "scrub": "Scrub land", "scrubland": "Scrub land",
        "forestopen": "Forest Open", "openforest": "Forest Open",
        "forestmoderatelydense": "Forest Moderately Dense",
        "moderatelydenseforest": "Forest Moderately Dense",
        "forestdense": "Forest Dense", "denseforest": "Forest Dense",
        "builtup": "Built up", "settlement": "Built up",
        "waterbody": "Waterbody", "waterbodies": "Waterbody", "riverstream": "River/Stream",
        # NRSC LULC 2015-16 Level-II (forest types, not density)
        "scrubforest": "Forest Open",
        "deciduous": "Forest Moderately Dense",
        "evergreensemievergreen": "Forest Dense", "evergreen": "Forest Dense",
        "forestplantation": "Plantation",
        "currentshiftingcultivation": "Shifting Cultivation",
        "rural": "Built up", "urban": "Built up",
        "fallow": "Fallow land",
        "grassgrazingland": "Scrub land",
    }
    if key in exact:
        return exact[key]
    # keyword fallbacks for differently worded legends
    rules = [
        (("shifting", "jhum", "abandoned"), "Shifting Cultivation"),
        (("fallow",), "Fallow land"),
        (("crop", "agri", "kharif", "rabi", "paddy", "cultivat", "terrace"), "Crop land"),
        (("plantation", "orchard", "horticult", "arecanut", "teagarden"), "Plantation"),
        (("scrub", "shrub", "bush"), "Scrub land"),
        (("barren", "rocky", "sandy", "mining", "quarry", "wasteland"), "Barren land"),
        (("river", "stream", "drain", "canal"), "River/Stream"),
        (("wetland", "swamp", "marsh", "mangrove"), "Waterbody"),
        (("water", "pond", "lake", "reservoir", "tank"), "Waterbody"),
        (("built", "settlement", "urban", "rural", "village", "habitation", "industrial"), "Built up"),
    ]
    for words, cls in rules:
        if any(w in key for w in words):
            return cls
    if "forest" in key:
        if "dense" in key and "moderate" not in key:
            return "Forest Dense"
        if "moderate" in key:
            return "Forest Moderately Dense"
        return "Forest Open"
    return text


def _empty_lulc(year):
    return {"year": year, "available": False, "composition": [], "dominant": "Unknown",
            "confidence": 0.0, "coverage": 0.0, "agri_percent": 0.0,
            "classification": "Unavailable", "agri_geom": None}


def mapped_lulc_summary(field_geojson, year):
    """Area composition and core-dominant class of one LULC year inside the field."""
    src = LULC_SOURCES.get(year)
    if src is None:
        return _empty_lulc(year)
    gdf, column = src["gdf"], src["column"]

    field_geom = to_utm_geometry(field_geojson)
    if not field_geom.is_valid:
        field_geom = field_geom.buffer(0)
    field_area = field_geom.area
    if field_area <= 0:
        return _empty_lulc(year)

    core_buffer = min(5.0, max(1.0, 0.02 * math.sqrt(field_area)))
    field_core = field_geom.buffer(-core_buffer)
    if field_core.is_empty or field_core.area < 0.35 * field_area:
        field_core = field_geom

    candidate = local_subset(gdf, field_geom)
    if candidate is None:
        out = _empty_lulc(year)
        out["available"] = True
        out["classification"] = "No mapped polygons in field"
        return out
    candidate = candidate.copy()
    candidate["_class"] = candidate[column].apply(canonical_lulc_name)
    candidate["_clip"] = candidate.geometry.intersection(field_geom)
    candidate["_full_area"] = candidate["_clip"].area
    candidate["_core_area"] = candidate.geometry.intersection(field_core).area

    grouped = candidate.groupby("_class", dropna=False)["_full_area"].sum().reset_index()
    grouped = grouped[grouped["_full_area"] > 0.5].sort_values("_full_area", ascending=False)
    coverage = min(100.0, float(grouped["_full_area"].sum()) / field_area * 100)

    composition = [
        {"LULC": r["_class"], "Area (ha)": round(r["_full_area"] / 10000, 3),
         "Percent of field": round(r["_full_area"] / field_area * 100, 1)}
        for _, r in grouped.iterrows()
    ]

    core_grouped = (candidate.groupby("_class")["_core_area"].sum().reset_index()
                    .sort_values("_core_area", ascending=False))
    core_total = core_grouped["_core_area"].sum() if not core_grouped.empty else 0
    if core_grouped.empty or core_total <= 0:
        dominant, confidence = "Unknown", 0.0
    else:
        dominant = core_grouped.iloc[0]["_class"]
        confidence = core_grouped.iloc[0]["_core_area"] / core_total * 100

    agri = candidate[candidate["_class"].isin(AGRICULTURAL_LULC)]
    agri_geom = unary_union(agri["_clip"].tolist()) if not agri.empty else None
    agri_percent = (agri_geom.area / field_area * 100) if agri_geom is not None else 0.0

    if coverage < 70:
        classification = "Incomplete GIS coverage"
    elif confidence >= 70:
        classification = f"{year}: {dominant}"
    else:
        classification = f"{year}: mixed, mainly {dominant}"

    return {"year": year, "available": True, "composition": composition, "dominant": dominant,
            "confidence": float(confidence), "coverage": float(coverage),
            "agri_percent": float(min(agri_percent, 100.0)),
            "classification": classification, "agri_geom": agri_geom}


def reconcile_lulc(field_geojson, s15, s22):
    """
    Combine LULC 2015 and 2022 into one suggested planning land use.

    Single-date maps often record fallow cropland and jhum regrowth as scrub /
    open forest. If the field was agricultural in 2015 and is transitional in
    2022, the agricultural class is suggested (to be confirmed in the field).
    """
    d15 = s15["dominant"] if s15["available"] else "Unknown"
    d22 = s22["dominant"] if s22["available"] else "Unknown"

    field_area = to_utm_geometry(field_geojson).area
    geoms = [g for g in (s15.get("agri_geom"), s22.get("agri_geom")) if g is not None]
    agri_any = (unary_union(geoms).area / field_area * 100) if geoms and field_area > 0 else 0.0
    agri_any = float(min(agri_any, 100.0))

    if d15 == "Unknown" and d22 == "Unknown":
        suggested, status, level = None, "No LULC evidence for this field", "unk"
        note = "Select the observed land use from the field visit."
    elif d15 == "Unknown":
        suggested, status, level = d22, "2022 only (2015 not available)", "mod"
        note = "Only one LULC year is available; confirm the land use in the field."
    elif d22 == "Unknown":
        suggested, status, level = d15, "2015 only (2022 not available)", "mod"
        note = "Only one LULC year is available; confirm the land use in the field."
    elif d15 == d22:
        suggested, status, level = d22, f"Stable: {d22} in 2015 and 2022", "low"
        note = "Both years agree."
    elif d15 in AGRICULTURAL_LULC and d22 in TRANSITIONAL_LULC:
        suggested, status, level = d15, f"{d15} (2015) → {d22} (2022)", "high"
        note = (f"Likely fallow or jhum-cycle land mapped as {d22.lower()} in 2022. "
                f"Suggested planning land use: {d15} — confirm with the farmer.")
    elif d22 in AGRICULTURAL_LULC:
        suggested, status, level = d22, f"{d15} (2015) → {d22} (2022)", "mod"
        note = "Brought under agriculture after 2015."
    else:
        suggested, status, level = d22, f"{d15} (2015) → {d22} (2022)", "mod"
        note = "Land use changed between the two maps; confirm in the field."

    # Strong agricultural footprint in either year overrides a non-agricultural default
    if suggested not in AGRICULTURAL_LULC and agri_any >= 50:
        agri_cls = d15 if d15 in AGRICULTURAL_LULC else d22 if d22 in AGRICULTURAL_LULC else "Crop land"
        note += f" {agri_any:.0f}% of the field is agricultural in at least one year; suggesting {agri_cls}."
        suggested = agri_cls

    return {"d15": d15, "d22": d22, "suggested": suggested, "status": status,
            "level": level, "note": note, "agri_any": agri_any,
            "agri_15": s15["agri_percent"], "agri_22": s22["agri_percent"]}


def table_cn_for_composition(composition):
    """Area-weighted table CN of a mapped composition (classes not in table ignored)."""
    num = den = 0.0
    for row in composition:
        cn = CN_BY_LULC.get(row["LULC"])
        if cn is not None:
            num += cn * row["Percent of field"]
            den += row["Percent of field"]
    return num / den if den > 0 else None


def scs_runoff(p_mm, cn):
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    return (p_mm - ia) ** 2 / (p_mm - ia + s) if p_mm > ia else 0.0


def runoff_lulc_correction(cn_map, c_map, mapped_composition, observed_class):
    """
    Shift the mapped CN by the table difference (observed − mapped land use) and
    rescale runoff. The representative event depth is calibrated so SCS-CN with
    the mapped CN reproduces the mapped runoff coefficient.
    """
    out = {"applied": False, "factor": 1.0, "cn_map": cn_map, "cn_adj": cn_map,
           "c_map": c_map, "c_adj": c_map, "event_p": np.nan, "note": ""}
    cn_tab_map = table_cn_for_composition(mapped_composition)
    cn_tab_obs = CN_BY_LULC.get(observed_class)
    if cn_tab_map is None or cn_tab_obs is None:
        out["note"] = "No mapped LULC to compare with — runoff used as mapped."
        return out
    if not (np.isfinite(cn_map) and np.isfinite(c_map)) or cn_map <= 0 or c_map <= 0:
        out["note"] = "CN or runoff coefficient unavailable — runoff used as mapped."
        return out
    delta = cn_tab_obs - cn_tab_map
    if abs(delta) < 0.5:
        out["note"] = "Observed land use matches the mapped land use — no correction needed."
        return out

    cn_adj = float(np.clip(cn_map + delta, 30, 98))

    def ratio(p, cn):
        return scs_runoff(p, cn) / p

    lo, hi = 1.0, 3000.0
    if ratio(lo, cn_map) < c_map < ratio(hi, cn_map):
        for _ in range(80):
            mid = (lo + hi) / 2
            if ratio(mid, cn_map) < c_map:
                lo = mid
            else:
                hi = mid
        p_event = (lo + hi) / 2
    else:
        p_event = 75.0
    c_model = ratio(p_event, cn_map)
    factor = ratio(p_event, cn_adj) / c_model if c_model > 0 else 1.0
    factor = float(np.clip(factor, 0.25, 4.0))

    out.update({"applied": True, "factor": factor, "cn_adj": cn_adj,
                "c_adj": float(min(c_map * factor, 0.98)), "event_p": float(p_event),
                "note": (f"CN shifted {delta:+.1f} (mapped mix ≈ {cn_tab_map:.0f} → "
                         f"{observed_class} {cn_tab_obs}); runoff × {factor:.2f}.")})
    return out


# ============================================================
# 12. MULTI-CRITERIA SITE SELECTION
# ============================================================

def find_pond_site(field_geojson, boundary_clearance=10, n_candidates=5, min_separation=15.0):
    """
    Weighted overlay inside the field (elevation, TWI, slope, drainage density,
    flow accumulation) with exclusion / penalty rules (field boundary, streams,
    shear zones, scarps / ridges). Returns the best site, alternative candidates
    and the suitability surface for mapping.
    """
    elevation, elevation_valid, transform, raster_crs = field_reference_grid(field_geojson)
    shape_ref = elevation.shape

    slope = raster_to_reference(SLOPE_RASTER, shape_ref, transform, raster_crs)
    twi = raster_to_reference(TWI_RASTER, shape_ref, transform, raster_crs)
    drainage = raster_to_reference(DRAINAGE_DENSITY_RASTER, shape_ref, transform, raster_crs)
    flow = raster_to_reference(FLOW_ACCUMULATION_RASTER, shape_ref, transform, raster_crs)

    valid = (
        elevation_valid & np.isfinite(elevation) & np.isfinite(slope)
        & np.isfinite(twi) & np.isfinite(drainage)
    )
    if np.count_nonzero(valid) == 0:
        return None

    lowest_idx = np.unravel_index(np.argmin(np.where(valid, elevation, np.inf)), shape_ref)
    lowest_x, lowest_y = xy(transform, lowest_idx[0], lowest_idx[1], offset="center")

    elevation_norm = normalize_array(elevation, valid)
    slope_norm = normalize_array(slope, valid)
    twi_norm = normalize_array(twi, valid)
    drainage_norm = normalize_array(drainage, valid)

    score = (
        0.45 * (1 - elevation_norm)
        + 0.25 * twi_norm
        + 0.15 * (1 - slope_norm)
        + 0.10 * drainage_norm
    )
    flow_valid = valid & np.isfinite(flow)
    if np.count_nonzero(flow_valid):
        flow_norm = normalize_array(np.log1p(np.maximum(flow, 0)), flow_valid)
        score = score + 0.05 * np.where(flow_valid, flow_norm, 0)

    suitability_surface = np.where(valid, score, np.nan)

    preferred = valid & (slope <= 8)
    if np.count_nonzero(preferred) == 0:
        preferred = valid & (slope <= 15)
    if np.count_nonzero(preferred) == 0:
        preferred = valid
    ranked = np.where(preferred, score, -np.inf)

    # --- geometry in raster CRS and UTM ---------------------------------
    field_geom = geometry_gdf(field_geojson).to_crs(raster_crs).geometry.iloc[0]
    field_utm = to_utm_geometry(field_geojson)
    search_area_utm = field_utm.buffer(1000)
    to_utm = Transformer.from_crs(raster_crs, WORKING_CRS, always_xy=True)
    to_wgs = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    local_streams = local_subset(streams_utm, search_area_utm)
    stream_union = unary_union(local_streams.geometry.tolist()) if local_streams is not None else None

    local_shear = local_subset(shear_gdf, search_area_utm)
    shear_union = unary_union(local_shear.geometry.tolist()) if local_shear is not None else None
    shear_far = None
    if shear_union is None and _ok(shear_gdf):
        c = field_utm.centroid
        shear_far = nearest_distance_to_layer(shear_gdf, c.x, c.y)

    local_geomorph = local_subset(geomorph_gdf, field_utm.buffer(50))

    order = np.argsort(ranked.ravel())[::-1]
    candidates = []
    for flat_index in order[:2000]:
        r, c = np.unravel_index(flat_index, shape_ref)
        if not np.isfinite(ranked[r, c]):
            continue
        px, py = xy(transform, r, c, offset="center")
        point = Point(px, py)
        boundary_distance = point.distance(field_geom.boundary)
        if boundary_distance < boundary_clearance or not field_geom.covers(point):
            continue

        ux, uy = to_utm.transform(px, py)
        upoint = Point(ux, uy)

        stream_distance = None
        if stream_union is not None:
            stream_distance = upoint.distance(stream_union)
            if stream_distance < 10:
                continue

        shear_distance = upoint.distance(shear_union) if shear_union is not None else shear_far

        adjusted = float(score[r, c])
        penalties = []
        if shear_distance is not None:
            if shear_distance < 50:
                adjusted -= 0.35
                penalties.append("Shear zone < 50 m")
            elif shear_distance < 100:
                adjusted -= 0.20
                penalties.append("Shear zone < 100 m")
            elif shear_distance < 250:
                adjusted -= 0.08
                penalties.append("Shear zone < 250 m")

        landform = extract_record_fields(
            point_record(local_geomorph, ux, uy), {"landform": GEOMORPH_FIELDS}
        )["landform"].lower()
        if "scarp" in landform or "ridge" in landform:
            adjusted -= 0.25
            penalties.append("Scarp / ridge landform")

        candidates.append({
            "x": float(px), "y": float(py), "x_utm": float(ux), "y_utm": float(uy),
            "elevation": float(elevation[r, c]),
            "slope": float(slope[r, c]),
            "twi": float(twi[r, c]),
            "drainage_density": float(drainage[r, c]),
            "flow_accumulation": float(flow[r, c]) if np.isfinite(flow[r, c]) else np.nan,
            "site_score": float(score[r, c]),
            "adjusted_score": adjusted,
            "boundary_distance": float(boundary_distance),
            "stream_distance": float(stream_distance) if stream_distance is not None else None,
            "shear_distance": float(shear_distance) if shear_distance is not None else None,
            "penalties": penalties,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda d: d["adjusted_score"], reverse=True)
    picks = []
    for cand in candidates:
        if all(math.hypot(cand["x"] - p["x"], cand["y"] - p["y"]) >= min_separation for p in picks):
            lon, lat = to_wgs.transform(cand["x"], cand["y"])
            cand["longitude"], cand["latitude"] = float(lon), float(lat)
            cand["rank"] = len(picks) + 1
            picks.append(cand)
        if len(picks) >= n_candidates:
            break

    lowest_lon, lowest_lat = to_wgs.transform(lowest_x, lowest_y)

    # --- suitability image (for map overlay) ----------------------------
    inside = geometry_mask([mapping(field_geom)], out_shape=shape_ref, transform=transform,
                           invert=True, all_touched=True)
    rgba = suitability_rgba(suitability_surface, inside)
    h, w = shape_ref
    corners = [xy(transform, rr, cc, offset=o) for rr, cc, o in
               [(0, 0, "ul"), (0, w - 1, "ur"), (h - 1, 0, "ll"), (h - 1, w - 1, "lr")]]
    lons, lats = zip(*[to_wgs.transform(cx, cy) for cx, cy in corners])
    image_bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

    field_values = suitability_surface[inside & np.isfinite(suitability_surface)]

    return {
        "recommended": picks[0],
        "alternatives": picks,
        "lowest": {
            "longitude": float(lowest_lon), "latitude": float(lowest_lat),
            "elevation": float(elevation[lowest_idx]),
        },
        "overlay": {"rgba": rgba, "bounds": image_bounds},
        "field_scores": field_values.astype(float).tolist(),
        "field_stats": {
            "cells": int(np.count_nonzero(valid)),
            "elev_min": float(np.nanmin(np.where(valid, elevation, np.nan))),
            "elev_max": float(np.nanmax(np.where(valid, elevation, np.nan))),
            "slope_mean": float(np.nanmean(np.where(valid, slope, np.nan))),
            "twi_mean": float(np.nanmean(np.where(valid, twi, np.nan))),
            "cell_size": float(abs(transform.a)),
        },
    }


# ============================================================
# 13. POND DESIGN
# ============================================================

def frustum_volume(depth, area_bottom, area_top):
    return depth / 3 * (area_bottom + area_top + math.sqrt(area_bottom * area_top))


def calculate_pond_geometry(bottom_breadth, water_depth, freeboard, side_slope, length_breadth_ratio):
    bottom_length = bottom_breadth * length_breadth_ratio
    water_top_breadth = bottom_breadth + 2 * side_slope * water_depth
    water_top_length = bottom_length + 2 * side_slope * water_depth
    bottom_area = bottom_length * bottom_breadth
    top_water_area = water_top_length * water_top_breadth
    volume = frustum_volume(water_depth, bottom_area, top_water_area)

    excavation_depth = water_depth + freeboard
    top_breadth = bottom_breadth + 2 * side_slope * excavation_depth
    top_length = bottom_length + 2 * side_slope * excavation_depth
    footprint = top_length * top_breadth
    excavation_volume = frustum_volume(excavation_depth, bottom_area, footprint)

    # Lining up to full excavation depth (water + freeboard)
    side_slant = math.sqrt(excavation_depth ** 2 + (side_slope * excavation_depth) ** 2)
    long_side_area = (bottom_length + top_length) / 2 * side_slant
    short_side_area = (bottom_breadth + top_breadth) / 2 * side_slant
    lining_area = bottom_area + 2 * long_side_area + 2 * short_side_area

    return {
        "bottom_length": float(bottom_length),
        "bottom_breadth": float(bottom_breadth),
        "water_top_length": float(water_top_length),
        "water_top_breadth": float(water_top_breadth),
        "top_length": float(top_length),
        "top_breadth": float(top_breadth),
        "water_depth": float(water_depth),
        "freeboard": float(freeboard),
        "side_slope": float(side_slope),
        "excavation_depth": float(excavation_depth),
        "volume": float(volume),
        "water_spread_area": float(top_water_area),
        "excavation_volume": float(excavation_volume),
        "footprint": float(footprint),
        "lining_area": float(lining_area),
        "liner_purchase_area": float(lining_area * 1.10),
    }


def design_farm_pond(available_water_m3, field_area_ha_value, storage_fraction,
                     maximum_land_fraction, water_depth, freeboard, side_slope,
                     length_breadth_ratio):
    field_area_m2 = field_area_ha_value * 10000
    requested_storage = max(0.0, available_water_m3 * storage_fraction)
    maximum_footprint = field_area_m2 * maximum_land_fraction

    def geom(b):
        return calculate_pond_geometry(b, water_depth, freeboard, side_slope, length_breadth_ratio)

    low, high = 0.5, 120.0
    for _ in range(80):
        mid = (low + high) / 2
        if geom(mid)["footprint"] > maximum_footprint:
            high = mid
        else:
            low = mid

    maximum_geom = geom(low)
    target_storage = min(requested_storage, maximum_geom["volume"])
    limiting_factor = "Land (max. footprint)" if requested_storage > maximum_geom["volume"] else "Runoff yield"

    lower, upper = 0.5, max(low, 0.5)
    for _ in range(80):
        mid = (lower + upper) / 2
        if geom(mid)["volume"] > target_storage:
            upper = mid
        else:
            lower = mid

    result = geom(lower)
    result["requested_storage"] = float(requested_storage)
    result["target_storage"] = float(target_storage)
    result["max_storage_by_land"] = float(maximum_geom["volume"])
    result["limiting_factor"] = limiting_factor
    result["occupied_percent"] = result["footprint"] / field_area_m2 * 100 if field_area_m2 else 0
    # Solver floor (0.5 m bottom breadth) can exceed a near-zero runoff target
    result["insufficient_runoff"] = requested_storage < 50.0
    return result


# ============================================================
# 14. GEOLOGICAL / SEEPAGE SCREENING
# ============================================================

def classify_soil_texture(texture):
    text = safe_text(texture).lower()
    if text == "unknown":
        return "Unknown", 1
    if "clay" in text and "sandy" not in text:
        return "Low to moderate seepage tendency", 0
    if any(term in text for term in ["fine", "silt loam", "silty clay"]):
        return "Moderate seepage tendency", 1
    if "loam" in text and "sandy" not in text:
        return "Moderate to high seepage tendency", 2
    if "sandy" in text:
        return "High seepage tendency", 3
    return "Field verification required", 1


def terrain_constraint(landform, slope):
    text = safe_text(landform).lower()
    if "scarp" in text or "ridge" in text or slope >= 24:
        return "SEVERE", 3
    if "highly dissected" in text or "structural hills" in text or slope >= 14:
        return "HIGH", 2
    if slope >= 8:
        return "MODERATE", 1
    return "LOW", 0


def shear_zone_class(distance):
    if distance is None:
        return "Unknown", 0
    if distance < 50:
        return "VERY HIGH CONCERN", 3
    if distance < 100:
        return "HIGH CONCERN", 2
    if distance < 250:
        return "MODERATE CONCERN", 1
    return "LOWER MAPPED CONCERN", 0


def build_screening(soil_texture, landform, lithology, shear_distance, lineament_density, slope):
    soil_text, soil_score = classify_soil_texture(soil_texture)
    terrain_text, terrain_score = terrain_constraint(landform, slope)
    shear_text, shear_score = shear_zone_class(shear_distance)

    lith_text = safe_text(lithology).lower()
    lithology_score = 2 if any(t in lith_text for t in ["sandstone", "limestone", "gravel", "alluv"]) else 1

    lineament_score = 0
    if lineament_density is not None and np.isfinite(lineament_density):
        if lineament_density >= 0.55:
            lineament_score = 2
        elif lineament_density >= 0.19:
            lineament_score = 1

    seepage_score = soil_score + shear_score + lithology_score + lineament_score
    seepage = "HIGH" if seepage_score >= 6 else "MODERATE" if seepage_score >= 3 else "LOW"

    lining = {
        "HIGH": "High seepage-risk screening. Field permeability testing and engineered "
                "lining (geomembrane / LDPE / HDPE) are likely required.",
        "MODERATE": "Moderate seepage-risk screening. Assess field permeability and available "
                    "clay material. Compacted clay or geomembrane lining may be required.",
        "LOW": "Lower GIS-screened seepage concern. Field permeability verification is still "
               "required before deciding that lining is unnecessary.",
    }[seepage]

    return {
        "soil_screen": soil_text,
        "terrain_constraint": terrain_text,
        "shear_concern": shear_text,
        "seepage_risk": seepage,
        "seepage_score": seepage_score,
        "components": {
            "Soil texture": soil_score, "Shear zone": shear_score,
            "Lithology": lithology_score, "Lineament density": lineament_score,
        },
        "lining_recommendation": lining,
    }


# ============================================================
# 15. COMPOSITE SUITABILITY INDEX
# ============================================================

def suitability_index(slope, twi, dependable_runoff_mm, seepage_risk, terrain_level,
                      shear_distance, current_lulc):
    """Indicative 0–100 index combining terrain, hydrology and geology screening."""
    sub = {
        "Slope": float(np.interp(slope, [0, 3, 8, 15, 24], [100, 100, 70, 35, 0])),
        "Wetness (TWI)": float(np.interp(twi if np.isfinite(twi) else 6, [3, 6, 9, 12], [15, 45, 80, 100])),
        "Water availability": float(np.interp(
            dependable_runoff_mm if np.isfinite(dependable_runoff_mm) else 0,
            [0, 150, 400, 800, 1200], [0, 30, 65, 90, 100])),
        "Seepage": {"LOW": 100.0, "MODERATE": 60.0, "HIGH": 25.0}.get(seepage_risk, 50.0),
        "Terrain / landform": {"LOW": 100.0, "MODERATE": 70.0, "HIGH": 40.0, "SEVERE": 10.0}.get(terrain_level, 50.0),
        "Shear-zone distance": 70.0 if shear_distance is None else float(
            np.interp(shear_distance, [0, 50, 100, 250, 500], [0, 20, 50, 85, 100])),
    }
    index = sum(SUITABILITY_WEIGHTS[k] * v for k, v in sub.items())

    flags = []
    if current_lulc in RESTRICTED_LULC:
        index = min(index, 35.0)
        flags.append(f"Observed land use '{current_lulc}' is normally excluded from farm-pond siting.")
    if terrain_level == "SEVERE":
        index = min(index, 40.0)
        flags.append("Severe terrain constraint (scarp / ridge or slope ≥ 24°).")

    if index >= 75:
        label, css, short = "Highly Suitable", "v-high", "S1"
    elif index >= 60:
        label, css, short = "Suitable", "v-good", "S2"
    elif index >= 45:
        label, css, short = "Marginally Suitable", "v-marg", "S3"
    else:
        label, css, short = "Not Recommended", "v-poor", "N"
    return {"index": float(index), "label": label, "css": css, "class": short,
            "components": sub, "flags": flags}


def cost_estimate(pond, excavation_rate, liner_rate, seepage_risk, include_liner, misc_percent):
    excavation_cost = pond["excavation_volume"] * excavation_rate
    liner_needed = include_liner == "Always" or (include_liner == "If seepage ≥ MODERATE" and seepage_risk in {"MODERATE", "HIGH"})
    liner_cost = pond["liner_purchase_area"] * liner_rate if liner_needed else 0.0
    subtotal = excavation_cost + liner_cost
    misc = subtotal * misc_percent / 100
    total = subtotal + misc
    per_m3 = total / pond["target_storage"] if pond["target_storage"] > 0 else np.nan
    return {"excavation": excavation_cost, "liner": liner_cost, "liner_included": liner_needed,
            "misc": misc, "total": total, "per_m3": per_m3}


# ============================================================
# 16. CHARTS (Plotly)
# ============================================================

CHART_FONT = dict(family="Inter, Segoe UI, Arial", color="#0f2a25", size=12)


def _layout(fig, height=320, **kw):
    # Defaults first, caller overrides second — never pass a key twice.
    settings = dict(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=CHART_FONT,
    )
    settings.update(kw)
    fig.update_layout(**settings)
    return fig


def chart_gauge(value, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": " / 100", "font": {"size": 30}},
        title={"text": f"<b>{label}</b>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#0f2a25", "thickness": 0.22},
            "steps": [
                {"range": [0, 45], "color": "#f8d4d5"},
                {"range": [45, 60], "color": "#fde9c4"},
                {"range": [60, 75], "color": "#cfe8f5"},
                {"range": [75, 100], "color": "#c9f0dc"},
            ],
            "threshold": {"line": {"color": "#d64045", "width": 3}, "value": 45},
        },
    ))
    return _layout(fig, 260, margin=dict(l=20, r=20, t=50, b=0))


def chart_radar(components):
    cats = list(components.keys())
    vals = [components[c] for c in cats]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1], theta=cats + cats[:1], fill="toself",
        fillcolor="rgba(10,163,123,.25)", line=dict(color="#067a5f", width=2), name="Site",
    ))
    fig.update_polars(radialaxis=dict(range=[0, 100], showline=False, gridcolor="#dbe7e3"),
                      angularaxis=dict(gridcolor="#dbe7e3"), bgcolor="rgba(0,0,0,0)")
    return _layout(fig, 320, showlegend=False, title="Criteria profile (0–100)")


def chart_water_balance(a, pond):
    labels = ["Rainfall on field", "Mean runoff", "75% dependable runoff", "Pond storage target"]
    rain_vol = a["rainfall"] * a["field_area"] * 10 if np.isfinite(a["rainfall"]) else 0
    vals = [rain_vol, a["mean_volume"], a["dependable_volume"], pond["target_storage"]]
    vals = [v if np.isfinite(v) else 0 for v in vals]
    colors = ["#9ecae1", "#4292c6", "#1478b8", "#067a5f"]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:,.0f} m³" for v in vals], textposition="auto",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="m³ per year", gridcolor="#e8efed")
    return _layout(fig, 300, title="Field water balance")


def chart_cross_section(p):
    B, D = p["bottom_breadth"], p["excavation_depth"]
    T, fb = p["top_breadth"], p["freeboard"]
    WT = p["water_top_breadth"]
    margin = max(4.0, T * 0.15)
    fig = go.Figure()
    fig.add_trace(go.Scatter(  # ground & cut
        x=[-T / 2 - margin, -T / 2, -B / 2, B / 2, T / 2, T / 2 + margin],
        y=[0, 0, -D, -D, 0, 0], mode="lines", line=dict(color="#7b5b3a", width=3),
        name="Excavation profile", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(  # earth fill below
        x=[-T / 2 - margin, -T / 2, -B / 2, B / 2, T / 2, T / 2 + margin, T / 2 + margin, -T / 2 - margin],
        y=[0, 0, -D, -D, 0, 0, -D - 0.6, -D - 0.6], fill="toself",
        fillcolor="rgba(160,120,80,.25)", line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(  # water
        x=[-WT / 2, -B / 2, B / 2, WT / 2, -WT / 2], y=[-fb, -D, -D, -fb, -fb],
        fill="toself", fillcolor="rgba(20,120,184,.45)", line=dict(color="#1478b8", width=1.5),
        name="Water (FSL)", hoverinfo="skip",
    ))
    ann = [
        (0, -D - 0.3, f"Bottom width {B:.1f} m"),
        (0, 0.35, f"Top width {T:.1f} m"),
        (0, -fb - (D - fb) / 2, f"Water depth {p['water_depth']:.2f} m"),
        (T / 2 + margin * 0.5, -fb / 2, f"Freeboard {fb:.2f} m"),
        (-T / 2 - margin * 0.35, -D / 2, f"Side slope {p['side_slope']:.2f}H : 1V"),
    ]
    for x0, y0, t in ann:
        fig.add_annotation(x=x0, y=y0, text=t, showarrow=False, font=dict(size=11, color="#0f2a25"),
                           bgcolor="rgba(255,255,255,.75)")
    fig.update_xaxes(title="Distance from centre-line (m)", zeroline=False, gridcolor="#eef3f1")
    fig.update_yaxes(title="Depth below ground (m)", zeroline=False, gridcolor="#eef3f1")
    return _layout(fig, 330, title="Indicative cross-section (breadth direction)",
                   legend=dict(orientation="h", y=-0.25))


def chart_lulc_years(summaries):
    """100% stacked bars: one bar per LULC year, segments = classes."""
    years = [s for s in summaries if s["available"] and s["composition"]]
    classes = []
    for s in years:
        for r in s["composition"]:
            if r["LULC"] not in classes:
                classes.append(r["LULC"])
    fig = go.Figure()
    for cls in classes:
        vals = [next((r["Percent of field"] for r in s["composition"] if r["LULC"] == cls), 0)
                for s in years]
        fig.add_trace(go.Bar(
            name=cls, y=[str(s["year"]) for s in years], x=vals, orientation="h",
            marker_color=LULC_COLORS.get(cls, "#95a5a6"),
            text=[f"{v:.0f}%" if v >= 8 else "" for v in vals], textposition="inside",
            hovertemplate=f"{cls}: %{{x:.1f}}% of field<extra>%{{y}}</extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], title="% of field", gridcolor="#eef3f1")
    fig.update_yaxes(title=None, autorange="reversed")
    return _layout(fig, 230, legend=dict(orientation="h", y=-0.35, font=dict(size=11)),
                   margin=dict(l=10, r=10, t=10, b=10))


def chart_score_distribution(scores, best):
    fig = go.Figure(go.Histogram(x=scores, nbinsx=25, marker_color="#7fc8a9",
                                 marker_line=dict(color="#fff", width=1)))
    fig.add_vline(x=best, line=dict(color="#d64045", width=3, dash="dash"),
                  annotation_text="Selected site", annotation_position="top")
    fig.update_xaxes(title="Raw site score (within field)", gridcolor="#eef3f1")
    fig.update_yaxes(title="Grid cells", gridcolor="#eef3f1")
    return _layout(fig, 300, title="Within-field suitability distribution", bargap=0.05)


def chart_seepage(components):
    names = list(components.keys())
    vals = [components[n] for n in names]
    maxes = {"Soil texture": 3, "Shear zone": 3, "Lithology": 2, "Lineament density": 2}
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=[maxes[n] for n in names], orientation="h",
                         marker_color="#edf3f1", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Bar(y=names, x=vals, orientation="h", showlegend=False,
                         marker_color=["#d64045" if v >= 2 else "#e39b16" if v == 1 else "#18a058" for v in vals],
                         text=vals, textposition="inside"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Seepage points (higher = more concern)", gridcolor="#eef3f1")
    return _layout(fig, 250, title="Seepage-risk contributors")


def chart_cost(cost):
    labels = ["Excavation", "Lining", "Contingency / misc."]
    vals = [cost["excavation"], cost["liner"], cost["misc"]]
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.6,
                           marker=dict(colors=["#7b5b3a", "#1478b8", "#a0aeab"],
                                       line=dict(color="#fff", width=2))))
    return _layout(fig, 260, margin=dict(l=0, r=0, t=10, b=0))


# ============================================================
# 17. MAP HELPERS
# ============================================================

def add_base_layers(m, labels=True):
    folium.TileLayer(tiles=ESRI_IMAGERY, attr="Esri World Imagery", name="🛰 Satellite imagery",
                     overlay=False, control=True, show=True, max_zoom=20).add_to(m)
    folium.TileLayer(tiles=ESRI_TOPO, attr="Esri World Topo", name="⛰ Topographic",
                     overlay=False, control=True, show=False, max_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="🛣 Road / village map", show=False).add_to(m)
    if labels:
        folium.TileLayer(tiles=ESRI_LABELS, attr="Esri", name="Place labels",
                         overlay=True, control=True, show=True, max_zoom=20).add_to(m)


def add_tools(m, measure=True):
    if measure:
        MeasureControl(position="topleft", primary_length_unit="meters",
                       secondary_length_unit="kilometers", primary_area_unit="sqmeters",
                       secondary_area_unit="hectares").add_to(m)
    MousePosition(position="bottomleft", prefix="Lat | Lon:", separator=" | ", num_digits=6).add_to(m)
    Fullscreen(position="topleft").add_to(m)


def add_boundary(m):
    if boundary_wgs is not None:
        folium.GeoJson(
            boundary_wgs.__geo_interface__, name="Ri Bhoi district boundary",
            style_function=lambda f: {"color": "#FFD600", "weight": 2.5, "fillOpacity": 0, "dashArray": "6 4"},
        ).add_to(m)


def add_streams(m, show=False, near_geom_utm=None):
    if near_geom_utm is not None:
        sub = local_subset(streams_utm, near_geom_utm)
        g = sub.to_crs("EPSG:4326") if sub is not None else None
    else:
        g = simplified_wgs(streams_utm, "streams")
    if g is not None and not g.empty:
        folium.GeoJson(
            g[["geometry"]].__geo_interface__, name="💧 Drainage network", show=show,
            style_function=lambda f: {"color": "#29b6f6", "weight": 2, "opacity": 0.9},
        ).add_to(m)


def add_field(m, field, fill=0.18, name="Farmer field"):
    folium.GeoJson(
        field, name=name,
        style_function=lambda f: {"color": "#00E676", "weight": 4, "fillColor": "#69F0AE",
                                  "fillOpacity": fill},
        tooltip=name,
    ).add_to(m)


def add_lulc_context(m, field_utm, year, buffer_m=250, show=False):
    src = LULC_SOURCES.get(year)
    if src is None:
        return
    col = src["column"]
    area = field_utm.buffer(buffer_m)
    sub = local_subset(src["gdf"], area)
    if sub is None:
        return
    g = sub[[col, "geometry"]].copy()
    g["geometry"] = g.geometry.intersection(area)
    g["cls"] = g[col].apply(canonical_lulc_name)
    g = g[~g.geometry.is_empty][["cls", "geometry"]].to_crs("EPSG:4326")
    folium.GeoJson(
        g.__geo_interface__, name=f"🌾 LULC {year}", show=show,
        style_function=lambda f: {"color": "#ffffff", "weight": 0.6,
                                  "fillColor": LULC_COLORS.get(f["properties"]["cls"], "#95a5a6"),
                                  "fillOpacity": 0.55},
        tooltip=folium.GeoJsonTooltip(fields=["cls"], aliases=[f"LULC {year}:"]),
    ).add_to(m)


def add_geomorph_context(m, field_utm, buffer_m=500):
    sub = local_subset(geomorph_gdf, field_utm.buffer(buffer_m))
    if sub is None:
        return
    col = first_matching_column(sub.columns, GEOMORPH_FIELDS)
    if col is None:
        return
    g = sub[[col, "geometry"]].copy().rename(columns={col: "landform"})
    g["landform"] = g["landform"].apply(safe_text)
    g["geometry"] = g.geometry.intersection(field_utm.buffer(buffer_m))
    g = g[~g.geometry.is_empty].to_crs("EPSG:4326")
    palette = ["#8e44ad", "#d35400", "#16a085", "#2c3e50", "#c0392b", "#7f8c8d", "#27ae60"]
    lookup = {v: palette[i % len(palette)] for i, v in enumerate(sorted(g["landform"].unique()))}
    folium.GeoJson(
        g.__geo_interface__, name="🏔 Geomorphology (context)", show=False,
        style_function=lambda f: {"color": "#333", "weight": 0.5,
                                  "fillColor": lookup.get(f["properties"]["landform"], "#999"),
                                  "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["landform"], aliases=["Landform:"]),
    ).add_to(m)


def add_shear_context(m, field_utm, buffer_m=1500):
    sub = local_subset(shear_gdf, field_utm.buffer(buffer_m))
    if sub is None:
        return
    folium.GeoJson(
        sub[["geometry"]].to_crs("EPSG:4326").__geo_interface__, name="⚠ Shear zones", show=True,
        style_function=lambda f: {"color": "#ff3d00", "weight": 3, "dashArray": "8 5",
                                  "fillColor": "#ff3d00", "fillOpacity": 0.15},
    ).add_to(m)


def pond_polygon_utm(cx, cy, length, breadth, angle_deg=0.0):
    rect = box(cx - length / 2, cy - breadth / 2, cx + length / 2, cy + breadth / 2)
    return affinity.rotate(rect, angle_deg, origin=(cx, cy)) if angle_deg else rect


def utm_to_wgs_geom(geom):
    return gpd.GeoSeries([geom], crs=WORKING_CRS).to_crs("EPSG:4326").iloc[0]


# ============================================================
# 18. REPORT BUILDERS
# ============================================================

def build_geojson_export(a):
    site, pond = a["site"], a["pond"]
    pond_wgs = utm_to_wgs_geom(pond_polygon_utm(site["x_utm"], site["y_utm"], pond["top_length"],
                                                pond["top_breadth"], a.get("pond_rotation", 0)))
    features = [
        {"type": "Feature", "geometry": a["field"],
         "properties": {"layer": "farmer_field", "farmer": a["farmer"], "village": a["village"],
                        "area_ha": round(a["field_area"], 4), "land_use": a["current_lulc"]}},
        {"type": "Feature", "geometry": as_plain_geojson(pond_wgs),
         "properties": {"layer": "indicative_pond_footprint",
                        "storage_m3": round(pond["target_storage"], 1),
                        "top_length_m": round(pond["top_length"], 2),
                        "top_breadth_m": round(pond["top_breadth"], 2),
                        "depth_m": round(pond["excavation_depth"], 2)}},
    ]
    for c in a["alternatives"]:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [c["longitude"], c["latitude"]]},
            "properties": {"layer": "candidate_site", "rank": c["rank"],
                           "adjusted_score": round(c["adjusted_score"], 4),
                           "elevation_m": round(c["elevation"], 1), "slope_deg": round(c["slope"], 2)},
        })
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2).encode("utf-8")


def build_html_report(a):
    site, pond, scr, si, cost = a["site"], a["pond"], a["screening"], a["suitability"], a["cost"]
    when = datetime.now().strftime("%d %b %Y, %H:%M")

    def row(k, v):
        return f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"

    rows_site = "".join([
        row("Farmer", a["farmer"] or "Not entered"),
        row("Village / locality", a["village"] or "Not entered"),
        row("Field area", f"{a['field_area']:.3f} ha"),
        row("Observed land use", a["current_lulc"]),
        row("Mapped LULC 2015 → 2022", f"{a['lulc_2015']} → {a['lulc_2022']}"),
        row("LULC reconciliation", a["lulc_note"]),
        row("Candidate location (lat, lon)", f"{site['latitude']:.6f}, {site['longitude']:.6f}"),
        row("Elevation / slope", f"{site['elevation']:.1f} m / {site['slope']:.2f}°"),
        row("TWI", f"{site['twi']:.2f}"),
    ])
    rows_hydro = "".join([
        row("Annual rainfall", fmt(a["rainfall"], "{:,.0f} mm")),
        row("Runoff land-use correction",
            (f"Applied: × {a['correction']['factor']:.2f} (CN {a['cn']:.1f} → {a['correction']['cn_adj']:.1f})"
             if a["correction"]["applied"] else "Not applied")),
        row("Mean annual runoff", fmt(a["mean_runoff"], "{:,.0f} mm")),
        row("75% dependable runoff", fmt(a["dependable_runoff"], "{:,.0f} mm")),
        row("Dependable water from field", fmt(a["dependable_volume"], "{:,.0f} m³/yr")),
        row("Curve Number (AMC II)", fmt(a["cn"], "{:.1f}")),
        row("Runoff coefficient", fmt(a["runoff_coefficient"], "{:.3f}")),
    ])
    rows_pond = "".join([
        row("Storage target", f"{pond['target_storage']:,.0f} m³ (limited by {pond['limiting_factor'].lower()})"),
        row("Bottom L × B", f"{pond['bottom_length']:.1f} × {pond['bottom_breadth']:.1f} m"),
        row("Top L × B", f"{pond['top_length']:.1f} × {pond['top_breadth']:.1f} m"),
        row("Water depth + freeboard", f"{pond['water_depth']:.2f} + {pond['freeboard']:.2f} m"),
        row("Side slope", f"{pond['side_slope']:.2f}H : 1V"),
        row("Footprint / field share", f"{pond['footprint']:,.0f} m² / {pond['occupied_percent']:.1f}%"),
        row("Earthwork (excavation)", f"{pond['excavation_volume']:,.0f} m³"),
        row("Lining area (+10% procurement)", f"{pond['lining_area']:,.0f} m² ({pond['liner_purchase_area']:,.0f} m²)"),
    ])
    rows_geo = "".join([
        row("Soil texture", a["soil_info"]["texture"]),
        row("Geomorphology", a["geom_info"]["landform"]),
        row("Lithology", a["lith_info"]["lithology"]),
        row("Nearest shear zone", fmt(a["shear_distance"], "{:,.0f} m", "No data")),
        row("Lineament density", fmt(a["lineament_density"], "{:.3f}", "No data")),
        row("Terrain constraint", scr["terrain_constraint"]),
        row("Seepage risk", scr["seepage_risk"]),
    ])
    rows_cost = "".join([
        row("Excavation", f"₹ {cost['excavation']:,.0f}"),
        row("Lining", f"₹ {cost['liner']:,.0f}" + ("" if cost["liner_included"] else " (not included)")),
        row("Contingency / misc.", f"₹ {cost['misc']:,.0f}"),
        row("Indicative total", f"₹ {cost['total']:,.0f}"),
        row("Cost per m³ storage", fmt(cost["per_m3"], "₹ {:,.0f}")),
    ])
    rows_idx = "".join(row(k, f"{v:.0f} / 100 (weight {SUITABILITY_WEIGHTS[k]:.0%})")
                       for k, v in si["components"].items())
    flags = "".join(f"<li>{html.escape(f)}</li>" for f in si["flags"])
    color = {"v-high": "#10a46e", "v-good": "#1b9bc4", "v-marg": "#e39b16", "v-poor": "#d64045"}[si["css"]]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>GeoPond-DSS Assessment</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;color:#0f2a25;max-width:900px;margin:24px auto;padding:0 18px;}}
.hd{{background:linear-gradient(115deg,#032d26,#0a9a72);color:#fff;padding:18px 22px;border-radius:14px;}}
.hd small{{opacity:.8;letter-spacing:.12em;font-size:11px;font-weight:700}} h1{{margin:4px 0 2px;font-size:26px}}
.v{{margin:16px 0;border-left:6px solid {color};background:#f6faf9;padding:14px 16px;border-radius:10px}}
.v b{{font-size:22px;color:{color}}} h2{{font-size:16px;margin:22px 0 8px;color:#05634c;border-bottom:2px solid #e3eeea;padding-bottom:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th{{text-align:left;width:42%;color:#3d5a53;font-weight:600}}
th,td{{padding:6px 8px;border-bottom:1px solid #edf3f1}} .note{{font-size:11.5px;color:#5c716b;background:#fff8e6;border:1px solid #f3e1b0;padding:10px 12px;border-radius:8px;margin-top:18px}}
@media print{{.hd{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>
<div class="hd"><small>ICAR RESEARCH COMPLEX FOR NEH REGION · UMIAM, MEGHALAYA</small>
<h1>GeoPond-DSS · Farm Pond Assessment</h1><div>Generated {when} · GeoPond-DSS v{APP_VERSION}</div></div>
<div class="v">Indicative suitability: <b>{si['label']} ({si['index']:.0f}/100)</b>
<div style="margin-top:6px;font-size:13px">{html.escape(scr['lining_recommendation'])}</div>
{f'<ul style="font-size:13px;color:#9b1c22">{flags}</ul>' if flags else ''}</div>
<h2>Field & candidate site</h2><table>{rows_site}</table>
<h2>Hydrology</h2><table>{rows_hydro}</table>
<h2>Preliminary pond design</h2><table>{rows_pond}</table>
<h2>Geology & seepage screening</h2><table>{rows_geo}</table>
<h2>Suitability index components</h2><table>{rows_idx}</table>
<h2>Indicative cost (user rates)</h2><table>{rows_cost}</table>
<div class="note"><b>Engineering note.</b> GeoPond-DSS provides GIS-based screening and preliminary planning only.
The selected coordinate is a GIS candidate zone, not a surveyed construction peg. Final siting, excavation, embankment,
inlet, outlet, spillway and lining require field survey, permeability testing and engineering verification.
Cost figures use the rates entered by the user and are indicative only.</div>
</body></html>""".encode("utf-8")


# ============================================================
# 19. SIDEBAR — inputs, parameters, data status
# ============================================================

logo_b64 = image_to_base64(ICAR_LOGO)

with st.sidebar:
    logo_small = (
        f'<img src="data:image/png;base64,{logo_b64}" style="width:44px;height:44px;background:#fff;border-radius:10px;padding:3px">'
        if logo_b64 else '<div style="font-size:34px">💧</div>'
    )
    st.markdown(
        f'<div class="sb-brand">{logo_small}<div><b>GeoPond-DSS</b>'
        f'<span>Web-GIS Decision Support · v{APP_VERSION}</span></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### 👨‍🌾 Farmer & field")
    farmer_name = st.text_input("Farmer name", key="farmer_name")
    village = st.text_input("Village / locality", key="village")

    st.markdown("##### ⚙️ Design assumptions")
    with st.expander("Pond planning parameters", expanded=False):
        storage_fraction_percent = st.slider("Storage as % of 75% dependable runoff", 10, 60, 30, 5)
        maximum_land_percent = st.slider("Max. field area occupied by pond (%)", 5, 20, 10, 1)
        water_depth = st.slider("Planning water depth (m)", 1.5, 3.5, 2.5, 0.25)
        freeboard = st.slider("Freeboard (m)", 0.30, 0.75, 0.50, 0.05)
        side_slope = st.slider("Side slope (H : 1V)", 1.0, 2.5, 1.5, 0.25)
        length_breadth_ratio = st.slider("Bottom length : breadth ratio", 1.0, 2.0, 1.5, 0.1)
        pond_rotation = st.slider("Footprint rotation on map (°)", -90, 90, 0, 5,
                                  help="Align the indicative footprint with the contour direction.")

    correct_runoff = st.toggle(
        "Correct runoff for confirmed land use", value=True,
        help="Shifts the mapped CN to the confirmed field land use (e.g. crop land mapped as "
             "scrub) and rescales mean and dependable runoff accordingly.")

    with st.expander("Cost rates (enter local SoR)", expanded=False):
        excavation_rate = st.number_input("Excavation rate (₹ per m³)", 0.0, 5000.0, 200.0, 10.0)
        liner_rate = st.number_input("Liner supply & laying (₹ per m²)", 0.0, 2000.0, 120.0, 5.0)
        include_liner = st.selectbox("Include liner cost", ["If seepage ≥ MODERATE", "Always", "Never"])
        misc_percent = st.slider("Contingency / inlet-outlet / misc. (%)", 0, 30, 10, 1)
        st.caption("Default rates are placeholders. Replace them with the current district "
                   "Schedule of Rates before using the estimate.")

    st.markdown("##### 🗺 Map options")
    show_streams = st.toggle("Show drainage network", value=True)
    show_minimap = st.toggle("Show overview minimap", value=True)

    st.markdown("##### 🧭 GIS database")
    loaded = sum(LAYER_STATUS.values())
    st.markdown(
        f"<div style='font-size:12px;opacity:.9'>CRS: EPSG:32646 · UTM 46N<br>"
        f"Rasters: {RASTER_COUNT}/{RASTER_COUNT} · Vectors: {loaded}/{len(LAYER_STATUS)}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Layer status", expanded=False):
        for name, ok in LAYER_STATUS.items():
            dot = "#5dff9c" if ok else "#ff6b6b"
            st.markdown(f"<div style='font-size:12px'><span class='sb-dot' style='background:{dot}'></span>{name}</div>",
                        unsafe_allow_html=True)
        for _yr in (2015, 2022):
            _src = LULC_SOURCES.get(_yr, {}).get("source", "not found")
            st.caption(f"LULC {_yr}: {_src}")

    st.markdown("---")
    if st.button("🔄 Start new assessment", use_container_width=True):
        reset_assessment()
        st.rerun()


# ============================================================
# 20. HEADER + PROGRESS STEPPER
# ============================================================

logo_html = (
    f'<div class="gp-logo"><img src="data:image/png;base64,{logo_b64}" alt="ICAR logo"></div>'
    if logo_b64 else '<div style="font-size:55px;">🌾</div>'
)

field = st.session_state["field_geometry"]
assessment = st.session_state["assessment"]
head_area = f"{field_area_ha(field):.2f} ha" if field else "—"
head_si = f"{assessment['suitability']['index']:.0f}" if assessment else "—"

st.markdown(
    f"""
<div class="gp-header"><div class="gp-header-row">
{logo_html}
<div>
  <div class="gp-org">INDIAN COUNCIL OF AGRICULTURAL RESEARCH</div>
  <div class="gp-institute">ICAR Research Complex for NEH Region, Umiam, Meghalaya</div>
  <div class="gp-title">GeoPond-DSS <small>v{APP_VERSION}</small></div>
  <div class="gp-subtitle">Geospatial Pond Siting &amp; Preliminary Design Decision Support System</div>
  <div class="gp-chips">
    <span class="gp-chip">📍 Ri Bhoi District, Meghalaya</span>
    <span class="gp-chip">🛰 Satellite + DEM analytics</span>
    <span class="gp-chip">📱 Mobile GPS field capture</span>
    <span class="gp-chip">🧮 Multi-criteria suitability</span>
  </div>
</div>
<div class="gp-head-stats">
  <div class="gp-head-stat"><span>GIS LAYERS</span><b>{RASTER_COUNT + loaded}</b></div>
  <div class="gp-head-stat"><span>FIELD</span><b>{head_area}</b></div>
  <div class="gp-head-stat"><span>SUITABILITY</span><b>{head_si}</b></div>
</div>
</div></div>
""",
    unsafe_allow_html=True,
)


def render_stepper():
    has_field = field is not None
    has_lulc = has_field and st.session_state["current_lulc"] is not None
    has_run = assessment is not None
    steps = [
        ("Map field", "GPS / draw / upload", has_field),
        ("Verify land use", "GIS vs. observed", has_lulc),
        ("Set assumptions", "design & cost", has_lulc),
        ("Run analysis", "MCDA + hydrology", has_run),
        ("Report", "download & share", has_run),
    ]
    active_found = False
    items = []
    for i, (t, sub, done) in enumerate(steps, 1):
        cls = "done" if done else ""
        if not done and not active_found:
            cls, active_found = "active", True
        mark = "✓" if done else str(i)
        items.append(f'<div class="gp-step {cls}"><div class="n">{mark}</div>'
                     f'<div class="t">{t}<small>{sub}</small></div></div>')
    st.markdown(f'<div class="gp-steps">{"".join(items)}</div>', unsafe_allow_html=True)


render_stepper()


# ============================================================
# 21. FIELD MAPPING WORKSPACE
# ============================================================

section("🗺", "Field mapping workspace",
        "Locate the field with GPS or search, trace its boundary — or upload a boundary file")

map_col, side_col = st.columns([2.6, 1], gap="medium")

with side_col:
    st.markdown(
        """<div class="gp-hint"><b>How to capture the field</b><br>
        1. Tap <b>◎</b> for GPS, or search a place 🔍<br>
        2. Zoom to parcel level on satellite imagery<br>
        3. Pick the <b>polygon ⬠</b> tool and trace the boundary<br>
        4. Edit vertices if needed — area updates instantly<br>
        <span style="opacity:.75">Or upload a GeoJSON / KML / zipped Shapefile below.</span></div>""",
        unsafe_allow_html=True,
    )

    with st.expander("📂 Upload field boundary", expanded=False):
        uploaded = st.file_uploader("GeoJSON, KML, GPKG or zipped Shapefile",
                                    type=["geojson", "json", "kml", "gpkg", "zip"])
        if uploaded is not None and st.button("Use uploaded boundary", use_container_width=True):
            try:
                geom = parse_uploaded_boundary(uploaded)
                st.session_state["field_geometry"] = geom
                st.session_state["assessment"] = None
                st.session_state["map_nonce"] += 1
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read the boundary: {exc}")

    with st.expander("🎯 Go to coordinates", expanded=False):
        g1, g2 = st.columns(2)
        go_lat = g1.number_input("Latitude", value=MAP_CENTER[0], format="%.6f")
        go_lon = g2.number_input("Longitude", value=MAP_CENTER[1], format="%.6f")
        if st.button("Fly to location", use_container_width=True):
            st.session_state["map_center"] = [go_lat, go_lon]
            st.session_state["map_nonce"] += 1
            st.rerun()

    if field:
        fa = field_area_ha(field)
        per = field_perimeter_m(field)
        clat, clon = field_centroid_wgs(field)
        st.markdown(
            f"""<div class="gp-card" style="margin-top:10px"><div class="gp-card-title">✅ Field captured</div>
            <div class="kv">
              <div><span>Area</span><b>{fa:.3f} ha</b></div>
              <div><span>Area</span><b>{fa * 10000:,.0f} m²</b></div>
              <div><span>Perimeter</span><b>{per:,.0f} m</b></div>
              <div><span>Bigha (Meghalaya ≈ 0.1338 ha)</span><b>{fa / 0.1338:.1f}</b></div>
            </div>
            <div style="font-size:11px;color:#7a8f89;margin-top:8px" class="mono">Centroid {clat:.6f}, {clon:.6f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if fa < 0.05:
            st.warning("Field is very small (< 0.05 ha). Raster cells are ~30 m, so screening will be coarse.")
    else:
        st.info("No field yet — draw a polygon on the map to begin.")


with map_col:
    if field:
        clat, clon = field_centroid_wgs(field)
        center, zoom = [clat, clon], 17
    elif st.session_state["map_center"]:
        center, zoom = st.session_state["map_center"], 17
    else:
        center, zoom = MAP_CENTER, MAP_ZOOM

    field_map = folium.Map(location=center, zoom_start=zoom, tiles=None,
                           control_scale=True, prefer_canvas=True)
    add_base_layers(field_map)
    add_boundary(field_map)
    if show_streams:
        add_streams(field_map, show=True,
                    near_geom_utm=to_utm_geometry(field).buffer(1500) if field else None)
    if field:
        _fu = to_utm_geometry(field)
        for _yr in (2015, 2022):
            add_lulc_context(field_map, _fu, _yr, buffer_m=150)
        add_field(field_map, field, fill=0.22)
        b = shape(field).bounds
        field_map.fit_bounds([[b[1], b[0]], [b[3], b[2]]], padding=(30, 30))

    LocateControl(position="topleft", flyTo=True, strings={"title": "Use my GPS location"},
                  locateOptions={"enableHighAccuracy": True, "maximumAge": 0, "timeout": 15000}
                  ).add_to(field_map)
    if Geocoder is not None:
        try:
            Geocoder(collapsed=True, position="topleft", add_marker=True).add_to(field_map)
        except Exception:
            pass

    Draw(
        export=False, position="topleft",
        draw_options={
            "polyline": False, "rectangle": False, "circle": False,
            "circlemarker": False, "marker": False,
            "polygon": {"allowIntersection": False, "showArea": True, "metric": True,
                        "shapeOptions": {"color": "#00E676", "weight": 4,
                                         "fillColor": "#69F0AE", "fillOpacity": 0.25}},
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(field_map)

    add_tools(field_map)
    if show_minimap:
        MiniMap(toggle_display=True, position="bottomright", zoom_level_offset=-6).add_to(field_map)
    folium.LayerControl(position="topright", collapsed=True).add_to(field_map)

    map_output = st_folium(
        field_map, height=640, use_container_width=True,
        key=f"geopond_field_map_{st.session_state['map_nonce']}",
        returned_objects=["all_drawings", "last_active_drawing"],
    )

# Only react to a *new* drawing, so uploaded boundaries are not overwritten
new_drawing = get_last_polygon(map_output)
if new_drawing is not None and new_drawing != st.session_state["last_drawing"]:
    st.session_state["last_drawing"] = new_drawing
    if new_drawing != st.session_state["field_geometry"]:
        st.session_state["field_geometry"] = new_drawing
        st.session_state["assessment"] = None
        st.rerun()

field = st.session_state["field_geometry"]


# ============================================================
# 22. LAND-USE VERIFICATION (LULC 2015 + 2022)
# ============================================================

mapped_lulc = "Unknown"
lulc15 = _empty_lulc(2015)
lulc22 = _empty_lulc(2022)
recon = None

LEVEL_CSS = {"low": "v-high", "mod": "v-good", "high": "v-marg", "unk": "v-poor"}

if field:
    lulc15 = mapped_lulc_summary(field, 2015)
    lulc22 = mapped_lulc_summary(field, 2022)
    recon = reconcile_lulc(field, lulc15, lulc22)
    mapped_lulc = lulc22["dominant"] if lulc22["available"] else lulc15["dominant"]

    section("🌾", "Land-use verification",
            "LULC 2015 and 2022 compared, reconciled, and confirmed against the field observation")

    l1, l2 = st.columns([1, 1.45], gap="medium")

    with l1:
        st.markdown(
            f"""<div class="verdict {LEVEL_CSS[recon['level']]}">
            <div class="k">2015 → 2022 LAND-USE EVIDENCE</div>
            <div class="v" style="font-size:22px">{html.escape(recon['status'])}</div>
            <div class="s">{html.escape(recon['note'])}</div>
            <ul><li>Agricultural share 2015: {recon['agri_15']:.0f}% · 2022: {recon['agri_22']:.0f}%</li>
            <li>Agricultural in at least one year: {recon['agri_any']:.0f}% of field</li></ul>
            </div>""",
            unsafe_allow_html=True,
        )

        suggested = recon["suggested"]
        default_index = LAND_USE_OPTIONS.index(suggested) if suggested in LAND_USE_OPTIONS else 0
        # Keyed per field, so a new boundary starts again from the suggested class
        field_key = hashlib.md5(json.dumps(field, sort_keys=True).encode()).hexdigest()[:10]
        current_lulc = st.selectbox(
            "Confirmed field land use (used for planning)", LAND_USE_OPTIONS, index=default_index,
            key=f"lulc_select_{field_key}",
            help="Pre-selected from LULC 2015 + 2022. Change it if the field visit shows otherwise — "
                 "the confirmed class drives the runoff correction and suitability.",
        )
        st.session_state["current_lulc"] = current_lulc

        if suggested and current_lulc != suggested:
            st.caption(f"✏️ Overriding the GIS suggestion ({suggested}) with a field observation.")
        if current_lulc in RESTRICTED_LULC:
            st.error(f"'{current_lulc}' is normally excluded from farm-pond siting. "
                     "The analysis will still run but the site will be flagged.")

    with l2:
        def _year_cell(s):
            if not s["available"]:
                return "<b>Not available</b>"
            return (f"<b>{html.escape(s['dominant'])}</b>"
                    f"<span style='margin-top:3px'>core share {s['confidence']:.0f}% · "
                    f"coverage {s['coverage']:.0f}%</span>")

        st.markdown(
            f"""<div class="gp-card"><div class="gp-card-title">🗺 Mapped land use inside the field</div>
            <div class="kv">
              <div><span>LULC 2015</span>{_year_cell(lulc15)}</div>
              <div><span>LULC 2022</span>{_year_cell(lulc22)}</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        if HAS_PLOTLY and (lulc15["composition"] or lulc22["composition"]):
            st.plotly_chart(chart_lulc_years([lulc15, lulc22]), use_container_width=True,
                            config={"displayModeBar": False})
        elif not (lulc15["composition"] or lulc22["composition"]):
            st.info("No mapped LULC polygons intersect this field in either year.")

        rows = {}
        for s in (lulc15, lulc22):
            for r in s["composition"]:
                rows.setdefault(r["LULC"], {"LULC": r["LULC"], "2015 (%)": 0.0, "2022 (%)": 0.0})
                rows[r["LULC"]][f"{s['year']} (%)"] = r["Percent of field"]
        if rows:
            with st.expander("Composition table (2015 vs 2022)"):
                df = pd.DataFrame(rows.values())
                df["Change (pts)"] = (df["2022 (%)"] - df["2015 (%)"]).round(1)
                st.dataframe(df, use_container_width=True, hide_index=True)
        if 2015 not in LULC_SOURCES:
            st.warning("LULC 2015 layer not found in data/cache — only 2022 is used. "
                       "See Layer status in the sidebar.")

    # ---- runoff consequence of the confirmed land use -------------------
    _mix = lulc22["composition"] or lulc15["composition"]
    _cn_tab_map = table_cn_for_composition(_mix)
    _cn_tab_obs = CN_BY_LULC.get(current_lulc)
    if correct_runoff and _cn_tab_map is not None and _cn_tab_obs is not None \
            and abs(_cn_tab_obs - _cn_tab_map) >= 0.5:
        direction = "higher" if _cn_tab_obs > _cn_tab_map else "lower"
        st.info(f"💧 The CN / runoff rasters reflect the mapped land use (table CN ≈ {_cn_tab_map:.0f}). "
                f"For **{current_lulc}** (table CN {_cn_tab_obs}) runoff will be corrected "
                f"({direction}) when you run the analysis. Toggle this in the sidebar.")



# ============================================================
# 23. RUN ANALYSIS
# ============================================================

if field:
    section("🧮", "Run GeoPond-DSS analysis",
            "Hydrology · terrain MCDA · exclusion buffers · geology & seepage · design · cost")

    r1, r2 = st.columns([3, 1])
    with r2:
        st.markdown(
            f"""<div class="gp-card" style="padding:10px 12px"><div class="kv">
            <div><span>Storage share</span><b>{storage_fraction_percent}%</b></div>
            <div><span>Max land</span><b>{maximum_land_percent}%</b></div>
            <div><span>Depth + FB</span><b>{water_depth:.2f}+{freeboard:.2f} m</b></div>
            <div><span>Side slope</span><b>{side_slope:.2f}:1</b></div></div></div>""",
            unsafe_allow_html=True,
        )
    with r1:
        run = st.button("💧  FIND BEST GIS-SCREENED POND SITE & PRELIMINARY DESIGN",
                        type="primary", use_container_width=True)
        st.caption("Change assumptions in the sidebar (⚙️). Re-run after any change to update results.")

    if run:
        progress = st.progress(0, text="Reading hydrological rasters…")
        try:
            field_area = field_area_ha(field)
            rainfall = raster_mean(RAINFALL_RASTER, field)
            mean_runoff = raster_mean(MEAN_RUNOFF_RASTER, field)
            dependable_runoff = raster_mean(DEPENDABLE_RUNOFF_RASTER, field)
            cn = raster_mean(CN_RASTER, field)
            runoff_coefficient = raster_mean(RUNOFF_COEFFICIENT_RASTER, field)

            # Land-use correction: mapped CN/runoff -> confirmed field land use
            mapped_mean_runoff, mapped_dependable_runoff = mean_runoff, dependable_runoff
            correction = runoff_lulc_correction(
                cn, runoff_coefficient, lulc22["composition"] or lulc15["composition"],
                st.session_state["current_lulc"],
            )
            if correct_runoff and correction["applied"]:
                mean_runoff = mean_runoff * correction["factor"]
                dependable_runoff = dependable_runoff * correction["factor"]
            else:
                correction["applied"] = False

            mean_volume = mean_runoff * field_area * 10
            dependable_volume = dependable_runoff * field_area * 10
            if not np.isfinite(dependable_volume):
                dependable_volume = 0.0

            progress.progress(25, text="Sizing the preliminary pond…")
            pond = design_farm_pond(
                dependable_volume, field_area, storage_fraction_percent / 100,
                maximum_land_percent / 100, water_depth, freeboard, side_slope, length_breadth_ratio,
            )
            half_diagonal = 0.5 * math.hypot(pond["top_length"], pond["top_breadth"])

            progress.progress(45, text="Running multi-criteria site suitability…")
            site_result = find_pond_site(field, boundary_clearance=max(10, half_diagonal + 2))
            if site_result is None:
                # relax the boundary clearance once for small / narrow fields
                site_result = find_pond_site(field, boundary_clearance=5)
                relaxed = True
            else:
                relaxed = False
            if site_result is None:
                progress.empty()
                st.error("No suitable GIS-screened candidate was identified inside this field "
                         "(all cells excluded by stream / boundary rules, or no raster coverage).")
                st.stop()

            site = site_result["recommended"]

            progress.progress(70, text="Screening soil, geomorphology, lithology & shear zones…")
            sx, sy = site["x_utm"], site["y_utm"]
            soil_info = extract_record_fields(point_record(soil_gdf, sx, sy),
                                              {"texture": ["Texture", "soil_texture"]})
            geom_info = extract_record_fields(point_record(geomorph_gdf, sx, sy),
                                              {"landform": GEOMORPH_FIELDS})
            lith_info = extract_record_fields(
                point_record(lithology_gdf, sx, sy),
                {"group": ["group_name", "group"], "formation": ["formation"],
                 "lithology": ["lithologic", "lithology"], "age": ["age"]},
            )
            shear_distance = site["shear_distance"]
            if shear_distance is None:
                shear_distance = nearest_distance_to_layer(shear_gdf, sx, sy)
            lineament_density = sample_raster(LINEAMENT_DENSITY_RASTER, sx, sy)

            screening = build_screening(soil_info["texture"], geom_info["landform"],
                                        lith_info["lithology"], shear_distance,
                                        lineament_density, site["slope"])

            progress.progress(88, text="Scoring suitability and estimating cost…")
            si = suitability_index(site["slope"], site["twi"], dependable_runoff,
                                   screening["seepage_risk"], screening["terrain_constraint"],
                                   shear_distance, st.session_state["current_lulc"])
            cost = cost_estimate(pond, excavation_rate, liner_rate, screening["seepage_risk"],
                                 include_liner, misc_percent)

            st.session_state["assessment"] = {
                "farmer": farmer_name, "village": village, "field": field,
                "field_area": field_area,
                "current_lulc": st.session_state["current_lulc"], "mapped_lulc": mapped_lulc,
                "lulc_2015": lulc15["dominant"] if lulc15["available"] else "Not available",
                "lulc_2022": lulc22["dominant"] if lulc22["available"] else "Not available",
                "lulc_status": recon["status"] if recon else "—",
                "lulc_note": recon["note"] if recon else "",
                "correction": correction,
                "mapped_mean_runoff": mapped_mean_runoff,
                "mapped_dependable_runoff": mapped_dependable_runoff,
                "rainfall": rainfall, "mean_runoff": mean_runoff,
                "dependable_runoff": dependable_runoff, "mean_volume": mean_volume,
                "dependable_volume": dependable_volume, "cn": cn,
                "runoff_coefficient": runoff_coefficient,
                "site": site, "alternatives": site_result["alternatives"],
                "lowest": site_result["lowest"], "overlay": site_result["overlay"],
                "field_scores": site_result["field_scores"], "field_stats": site_result["field_stats"],
                "relaxed_clearance": relaxed,
                "pond": pond, "pond_rotation": pond_rotation,
                "soil_info": soil_info, "geom_info": geom_info, "lith_info": lith_info,
                "shear_distance": shear_distance, "lineament_density": lineament_density,
                "screening": screening, "suitability": si, "cost": cost,
                "params": {
                    "storage_fraction_percent": storage_fraction_percent,
                    "maximum_land_percent": maximum_land_percent,
                    "water_depth": water_depth, "freeboard": freeboard,
                    "side_slope": side_slope, "length_breadth_ratio": length_breadth_ratio,
                },
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            progress.progress(100, text="Done")
        except Exception as exc:
            progress.empty()
            st.exception(exc)
            st.stop()
        st.rerun()


# ============================================================
# 24. RESULTS DASHBOARD
# ============================================================

assessment = st.session_state["assessment"]

if assessment and field:
    a = assessment
    site, lowest, pond = a["site"], a["lowest"], a["pond"]
    screening, si, cost = a["screening"], a["suitability"], a["cost"]

    section("📊", "Assessment dashboard",
            f"Generated {a['timestamp'].replace('T', ' ')} · results are GIS screening, not survey")

    if pond.get("insufficient_runoff"):
        st.error("Dependable runoff from this field is too small (< 50 m³ storage) for a viable pond. "
                 "The dimensions shown are the minimum the solver produces — check for upslope catchment "
                 "inflow or consider a larger contributing area.")

    if a.get("relaxed_clearance"):
        st.warning("Boundary clearance had to be relaxed to 5 m to find a candidate. "
                   "The indicative footprint may extend beyond the field — check on the map.")

    # ---- verdict row ---------------------------------------------------
    v1, v2, v3 = st.columns([1.35, 1, 1.15], gap="medium")
    reasons = []
    reasons.append(f"Slope {site['slope']:.1f}° · TWI {site['twi']:.1f} · terrain "
                   f"{screening['terrain_constraint'].lower()}")
    reasons.append(f"Seepage screening: {screening['seepage_risk'].lower()} "
                   f"({screening['soil_screen'].lower()})")
    reasons.append(f"Storage {pond['target_storage']:,.0f} m³ — limited by "
                   f"{pond['limiting_factor'].lower()}")
    reasons.extend(si["flags"])
    with v1:
        st.markdown(
            f"""<div class="verdict {si['css']}">
            <div class="k">INDICATIVE SITE SUITABILITY · CLASS {si['class']}</div>
            <div class="v">{si['label']}</div>
            <div class="s">Best GIS-screened candidate inside the field at
            <span class="mono">{site['latitude']:.6f}, {site['longitude']:.6f}</span></div>
            <ul>{''.join(f'<li>{html.escape(r)}</li>' for r in reasons)}</ul>
            </div>""",
            unsafe_allow_html=True,
        )
    with v2:
        if HAS_PLOTLY:
            st.plotly_chart(chart_gauge(si["index"], "Suitability index"),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.metric("Suitability index", f"{si['index']:.0f} / 100")
    with v3:
        if HAS_PLOTLY:
            st.plotly_chart(chart_radar(si["components"]), use_container_width=True,
                            config={"displayModeBar": False})

    k = st.columns(6)
    k[0].metric("Field area", f"{a['field_area']:.3f} ha")
    k[1].metric("Dependable water", f"{a['dependable_volume']:,.0f} m³")
    k[2].metric("Storage target", f"{pond['target_storage']:,.0f} m³")
    k[3].metric("Pond footprint", f"{pond['footprint']:,.0f} m²", f"{pond['occupied_percent']:.1f}% of field",
                delta_color="off")
    k[4].metric("Seepage risk", screening["seepage_risk"])
    k[5].metric("Indicative cost", f"₹ {cost['total'] / 1e5:,.2f} L", fmt(cost["per_m3"], "₹ {:,.0f}/m³"),
                delta_color="off")

    tabs = st.tabs(["🗺 Site map", "🎯 Suitability", "💧 Hydrology", "📐 Pond design",
                    "🪨 Geology & seepage", "💰 Cost", "📄 Report & export", "🔬 Technical"])

    # ---- 1. site map ---------------------------------------------------
    with tabs[0]:
        mc, ic = st.columns([2.6, 1], gap="medium")
        with mc:
            result_map = folium.Map(location=[site["latitude"], site["longitude"]], zoom_start=18,
                                    tiles=None, control_scale=True)
            add_base_layers(result_map)
            field_utm = to_utm_geometry(field)

            ov = a["overlay"]
            folium.raster_layers.ImageOverlay(
                image=ov["rgba"], bounds=ov["bounds"], opacity=0.75, name="🎯 Suitability surface",
                mercator_project=True, zindex=3,
            ).add_to(result_map)

            for _yr in (2015, 2022):
                add_lulc_context(result_map, field_utm, _yr)
            add_geomorph_context(result_map, field_utm)
            add_streams(result_map, show=True, near_geom_utm=field_utm.buffer(1000))
            add_shear_context(result_map, field_utm)
            add_field(result_map, field, fill=0.05)

            # setback zone used by the site search
            half_diag = 0.5 * math.hypot(pond["top_length"], pond["top_breadth"])
            setback = field_utm.buffer(-max(10, half_diag + 2))
            if not setback.is_empty:
                folium.GeoJson(as_plain_geojson(utm_to_wgs_geom(setback)), name="Feasible centre zone",
                               show=False, style_function=lambda f: {"color": "#ffffff", "weight": 1.5,
                                                                     "dashArray": "4 4", "fillOpacity": 0}
                               ).add_to(result_map)

            pond_poly = utm_to_wgs_geom(pond_polygon_utm(site["x_utm"], site["y_utm"], pond["top_length"],
                                                         pond["top_breadth"], a.get("pond_rotation", 0)))
            folium.GeoJson(
                as_plain_geojson(pond_poly), name="📐 Indicative pond footprint",
                style_function=lambda f: {"color": "#FF1744", "weight": 3, "fillColor": "#2196F3",
                                          "fillOpacity": 0.45},
                tooltip=f"Pond {pond['top_length']:.1f} × {pond['top_breadth']:.1f} m · "
                        f"{pond['target_storage']:,.0f} m³",
            ).add_to(result_map)

            alt_group = folium.FeatureGroup(name="🔢 Alternative candidates", show=True)
            for c in a["alternatives"][1:]:
                folium.Marker(
                    [c["latitude"], c["longitude"]],
                    tooltip=f"Candidate #{c['rank']} · score {c['adjusted_score']:.3f}",
                    icon=folium.DivIcon(html=(
                        f'<div style="background:#fff;color:#04382f;border:2px solid #04382f;'
                        f'border-radius:50%;width:24px;height:24px;line-height:20px;text-align:center;'
                        f'font-weight:800;font-size:12px;box-shadow:0 2px 6px rgba(0,0,0,.35)">{c["rank"]}</div>'
                    ), icon_size=(24, 24), icon_anchor=(12, 12)),
                ).add_to(alt_group)
            alt_group.add_to(result_map)

            folium.Marker(
                [site["latitude"], site["longitude"]],
                tooltip="★ Recommended GIS candidate",
                popup=folium.Popup(
                    f"<b>Recommended candidate</b><br>Lat {site['latitude']:.6f}<br>Lon {site['longitude']:.6f}"
                    f"<br>Elev {site['elevation']:.1f} m · Slope {site['slope']:.1f}°"
                    f"<br><a href='https://www.google.com/maps/dir/?api=1&destination={site['latitude']},{site['longitude']}' "
                    f"target='_blank'>Navigate with Google Maps ↗</a>", max_width=260),
                icon=folium.Icon(color="red", icon="tint", prefix="fa"),
            ).add_to(result_map)
            folium.Marker([lowest["latitude"], lowest["longitude"]],
                          tooltip=f"Lowest DEM cell ({lowest['elevation']:.1f} m)",
                          icon=folium.Icon(color="orange", icon="arrow-down", prefix="fa")).add_to(result_map)

            add_tools(result_map)
            folium.LayerControl(collapsed=False, position="topright").add_to(result_map)
            st_folium(result_map, height=640, use_container_width=True, key="result_map",
                      returned_objects=[])

        with ic:
            st.markdown(
                """<div class="gp-card"><div class="gp-card-title">Legend</div>
                <div style="font-size:12px;margin-bottom:4px">Within-field suitability</div>
                <div class="legend-bar"></div><div class="legend-row"><span>Lower</span><span>Higher</span></div>
                <div style="font-size:12px;line-height:2;margin-top:8px">
                <span style="color:#d32f2f">●</span> Recommended candidate<br>
                <span style="color:#f57c00">●</span> Lowest DEM cell<br>
                <span style="border:2px solid #04382f;border-radius:50%;padding:0 5px;font-weight:800;font-size:10px">2</span>
                &nbsp;Alternative candidates<br>
                <span style="color:#2196F3">■</span> Indicative pond footprint<br>
                <span style="color:#29b6f6">━</span> Drainage network<br>
                <span style="color:#ff3d00">┅</span> Shear zones</div></div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">📍 Recommended candidate</div>
                <div class="kv">
                <div><span>Latitude</span><b class="mono">{site['latitude']:.6f}</b></div>
                <div><span>Longitude</span><b class="mono">{site['longitude']:.6f}</b></div>
                <div><span>Elevation</span><b>{site['elevation']:.1f} m</b></div>
                <div><span>Slope</span><b>{site['slope']:.2f}°</b></div>
                <div><span>Boundary clearance</span><b>{site['boundary_distance']:.1f} m</b></div>
                <div><span>Stream distance</span><b>{fmt(site['stream_distance'], '{:,.0f} m', '> 1 km')}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            st.link_button(
                "🧭 Navigate to site (Google Maps)",
                f"https://www.google.com/maps/dir/?api=1&destination={site['latitude']},{site['longitude']}",
                use_container_width=True,
            )
            st.caption("The pond rectangle is indicative. Rotate it in the sidebar to follow the contour; "
                       "final orientation needs field survey and inlet/outlet planning.")

    # ---- 2. suitability ------------------------------------------------
    with tabs[1]:
        s1, s2 = st.columns([1.2, 1], gap="medium")
        with s1:
            if HAS_PLOTLY and a["field_scores"]:
                st.plotly_chart(chart_score_distribution(a["field_scores"], site["site_score"]),
                                use_container_width=True, config={"displayModeBar": False})
            alt_df = pd.DataFrame([{
                "Rank": c["rank"], "Latitude": round(c["latitude"], 6), "Longitude": round(c["longitude"], 6),
                "Elevation (m)": round(c["elevation"], 1), "Slope (°)": round(c["slope"], 2),
                "TWI": round(c["twi"], 2), "Raw score": round(c["site_score"], 3),
                "Adjusted score": round(c["adjusted_score"], 3),
                "Penalties": ", ".join(c["penalties"]) or "—",
            } for c in a["alternatives"]])
            st.markdown("**Ranked candidate locations** (≥ 15 m apart)")
            st.dataframe(alt_df, use_container_width=True, hide_index=True,
                         column_config={"Adjusted score": st.column_config.ProgressColumn(
                             "Adjusted score", min_value=0.0, max_value=1.0, format="%.3f")})
        with s2:
            comp_df = pd.DataFrame([{
                "Criterion": k_, "Score (0–100)": round(v_, 0), "Weight": SUITABILITY_WEIGHTS[k_],
                "Contribution": round(v_ * SUITABILITY_WEIGHTS[k_], 1),
            } for k_, v_ in si["components"].items()])
            st.markdown("**Composite suitability index — breakdown**")
            st.dataframe(comp_df, use_container_width=True, hide_index=True,
                         column_config={
                             "Score (0–100)": st.column_config.ProgressColumn(
                                 "Score (0–100)", min_value=0, max_value=100, format="%d"),
                             "Weight": st.column_config.NumberColumn(format="%.2f"),
                         })
            st.markdown(
                """<div class="gp-hint" style="margin-top:8px"><b>Within-field site score</b> =
                0.45·(low elevation) + 0.25·TWI + 0.15·(gentle slope) + 0.10·drainage density + 0.05·flow accumulation,
                normalised within the field (5–95 percentile). Penalties: shear zone &lt;50/100/250 m
                (−0.35/−0.20/−0.08), scarp/ridge (−0.25). Exclusions: &lt;10 m from streams, inside boundary set-back.<br><br>
                <b>Composite index</b> scores the chosen site on absolute thresholds and is capped for restricted land
                use or severe terrain. Classes: S1 ≥ 75, S2 60–75, S3 45–60, N &lt; 45.</div>""",
                unsafe_allow_html=True,
            )

    # ---- 3. hydrology --------------------------------------------------
    with tabs[2]:
        h1, h2 = st.columns([1, 1.25], gap="medium")
        with h1:
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">💧 Field hydrology (zonal means)</div>
                <div class="kv">
                <div><span>Annual rainfall</span><b>{fmt(a['rainfall'], '{:,.0f} mm')}</b></div>
                <div><span>Mean annual runoff</span><b>{fmt(a['mean_runoff'], '{:,.0f} mm')}</b></div>
                <div><span>75% dependable runoff</span><b>{fmt(a['dependable_runoff'], '{:,.0f} mm')}</b></div>
                <div><span>Dependable water</span><b>{fmt(a['dependable_volume'], '{:,.0f} m³/yr')}</b></div>
                <div><span>Curve Number — mapped → used</span><b>{fmt(a['cn'], '{:.1f}')} → {fmt(a['correction']['cn_adj'] if a['correction']['applied'] else a['cn'], '{:.1f}')}</b></div>
                <div><span>Runoff coefficient — mapped → used</span><b>{fmt(a['runoff_coefficient'], '{:.3f}')} → {fmt(a['correction']['c_adj'] if a['correction']['applied'] else a['runoff_coefficient'], '{:.3f}')}</b></div>
                <div><span>Storage / dependable</span><b>{fmt(pond['target_storage'] / a['dependable_volume'] * 100 if a['dependable_volume'] else np.nan, '{:.0f}%')}</b></div>
                <div><span>Fills per season (est.)</span><b>{fmt(a['dependable_volume'] / pond['target_storage'] if pond['target_storage'] else np.nan, '{:.1f}×')}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            corr = a["correction"]
            if corr["applied"]:
                st.success(
                    f"**Land-use correction applied** — confirmed land use **{a['current_lulc']}** "
                    f"(mapped: 2015 {a['lulc_2015']}, 2022 {a['lulc_2022']}). {corr['note']} "
                    f"Dependable runoff {a['mapped_dependable_runoff']:,.0f} → {a['dependable_runoff']:,.0f} mm "
                    f"(calibrated event depth ≈ {corr['event_p']:.0f} mm).")
            elif corr.get("note"):
                st.caption(f"Land-use correction not applied: {corr['note']}")
            st.info("Rainfall keeps the information content of its source dataset; resampling does not "
                    "create new 30-m rainfall observations. Runoff is from the field itself — any upslope "
                    "catchment contribution would add to it and should be checked in the field.")
        with h2:
            if HAS_PLOTLY:
                st.plotly_chart(chart_water_balance(a, pond), use_container_width=True,
                                config={"displayModeBar": False})

    # ---- 4. design -----------------------------------------------------
    with tabs[3]:
        d1, d2 = st.columns([1, 1.35], gap="medium")
        with d1:
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">📐 Preliminary dimensions</div>
                <div class="kv">
                <div><span>Storage target</span><b>{pond['target_storage']:,.0f} m³</b></div>
                <div><span>Design volume</span><b>{pond['volume']:,.0f} m³</b></div>
                <div><span>Bottom L × B</span><b>{pond['bottom_length']:.1f} × {pond['bottom_breadth']:.1f} m</b></div>
                <div><span>Top L × B</span><b>{pond['top_length']:.1f} × {pond['top_breadth']:.1f} m</b></div>
                <div><span>Water depth</span><b>{pond['water_depth']:.2f} m</b></div>
                <div><span>Freeboard</span><b>{pond['freeboard']:.2f} m</b></div>
                <div><span>Excavation depth</span><b>{pond['excavation_depth']:.2f} m</b></div>
                <div><span>Side slope</span><b>{pond['side_slope']:.2f}H : 1V</b></div>
                <div><span>Water-spread area</span><b>{pond['water_spread_area']:,.0f} m²</b></div>
                <div><span>Footprint</span><b>{pond['footprint']:,.0f} m² ({pond['occupied_percent']:.1f}%)</b></div>
                <div><span>Earthwork</span><b>{pond['excavation_volume']:,.0f} m³</b></div>
                <div><span>Limiting factor</span><b style="font-size:12px">{pond['limiting_factor']}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            if pond["limiting_factor"].startswith("Land"):
                st.caption(f"Runoff could fill {pond['requested_storage']:,.0f} m³, but the land limit "
                           f"({a['params']['maximum_land_percent']}% of field) caps storage at "
                           f"{pond['max_storage_by_land']:,.0f} m³. Raise the limit or depth in the sidebar to "
                           "store more.")
        with d2:
            if HAS_PLOTLY:
                st.plotly_chart(chart_cross_section(pond), use_container_width=True,
                                config={"displayModeBar": False})
            st.caption("Earthwork from excavation can be used to form bunds around the pond. "
                       "Provide inlet silt trap and a protected overflow/spillway.")

    # ---- 5. geology ----------------------------------------------------
    with tabs[4]:
        soil_info, geom_info, lith_info = a["soil_info"], a["geom_info"], a["lith_info"]
        g1, g2 = st.columns([1.2, 1], gap="medium")
        with g1:
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">🪨 Site geology & terrain</div>
                <div class="kv">
                <div><span>Soil texture</span><b>{esc(soil_info['texture'])}</b></div>
                <div><span>Soil screening</span><b style="font-size:12.5px">{esc(screening['soil_screen'])}</b></div>
                <div><span>Geomorphology</span><b style="font-size:12.5px">{esc(geom_info['landform'])}</b></div>
                <div><span>Terrain constraint</span><b>{risk_badge(screening['terrain_constraint'])}</b></div>
                <div><span>Lithological group</span><b style="font-size:12.5px">{esc(lith_info['group'])}</b></div>
                <div><span>Formation</span><b style="font-size:12.5px">{esc(lith_info['formation'])}</b></div>
                <div><span>Lithology</span><b style="font-size:12.5px">{esc(lith_info['lithology'])}</b></div>
                <div><span>Geological age</span><b style="font-size:12.5px">{esc(lith_info['age'])}</b></div>
                <div><span>Nearest shear zone</span><b>{fmt(a['shear_distance'], '{:,.0f} m', 'No data')}</b></div>
                <div><span>Shear-zone screening</span><b style="font-size:12.5px">{esc(screening['shear_concern'])}</b></div>
                <div><span>Lineament density</span><b>{fmt(a['lineament_density'], '{:.3f}', 'No data')}</b></div>
                <div><span>Seepage screening</span><b>{risk_badge(screening['seepage_risk'])}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
        with g2:
            if HAS_PLOTLY:
                st.plotly_chart(chart_seepage(screening["components"]), use_container_width=True,
                                config={"displayModeBar": False})
            st.info(screening["lining_recommendation"])
            c1_, c2_ = st.columns(2)
            c1_.metric("Lining area", f"{pond['lining_area']:,.0f} m²")
            c2_.metric("Procurement (+10%)", f"{pond['liner_purchase_area']:,.0f} m²")

    # ---- 6. cost -------------------------------------------------------
    with tabs[5]:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">💰 Indicative cost estimate</div>
                <div class="kv">
                <div><span>Earthwork</span><b>{pond['excavation_volume']:,.0f} m³ × ₹{excavation_rate:,.0f}</b></div>
                <div><span>Excavation cost</span><b>₹ {cost['excavation']:,.0f}</b></div>
                <div><span>Liner</span><b>{pond['liner_purchase_area']:,.0f} m² × ₹{liner_rate:,.0f}</b></div>
                <div><span>Liner cost</span><b>₹ {cost['liner']:,.0f}{'' if cost['liner_included'] else ' (excl.)'}</b></div>
                <div><span>Contingency / misc. ({misc_percent}%)</span><b>₹ {cost['misc']:,.0f}</b></div>
                <div><span>Total</span><b>₹ {cost['total']:,.0f}</b></div>
                <div><span>Cost per m³ stored</span><b>{fmt(cost['per_m3'], '₹ {:,.0f}')}</b></div>
                <div><span>Cost per ha of field</span><b>₹ {cost['total'] / a['field_area']:,.0f}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            st.caption("Rates come from the sidebar and are applied to the dimensions computed at run time. "
                       "Update rates and re-run for a fresh estimate. Not a substitute for a detailed estimate.")
        with c2:
            if HAS_PLOTLY and cost["total"] > 0:
                st.plotly_chart(chart_cost(cost), use_container_width=True, config={"displayModeBar": False})

    # ---- 7. report & export -------------------------------------------
    with tabs[6]:
        output = {
            "DSS": f"GeoPond-DSS v{APP_VERSION}", "Timestamp": a["timestamp"],
            "Farmer": a["farmer"], "Village": a["village"],
            "Field_Area_ha": a["field_area"],
            "Current_Land_Use": a["current_lulc"], "LULC_2015": a["lulc_2015"], "LULC_2022": a["lulc_2022"],
            "LULC_Change": a["lulc_status"],
            "Runoff_Correction_Factor": a["correction"]["factor"] if a["correction"]["applied"] else 1.0,
            "CN_Used": a["correction"]["cn_adj"] if a["correction"]["applied"] else a["cn"],
            "Mapped_Dependable_Runoff_mm": a["mapped_dependable_runoff"],
            "Rainfall_mm_year": a["rainfall"], "Mean_Runoff_mm": a["mean_runoff"],
            "Dependable_Runoff_mm": a["dependable_runoff"],
            "Dependable_Water_m3_year": a["dependable_volume"],
            "CN_II": a["cn"], "Runoff_Coefficient": a["runoff_coefficient"],
            "Pond_Latitude": site["latitude"], "Pond_Longitude": site["longitude"],
            "Elevation_m": site["elevation"], "Slope_degree": site["slope"], "TWI": site["twi"],
            "Site_Score_Raw": site["site_score"], "Site_Score_Adjusted": site["adjusted_score"],
            "Suitability_Index": si["index"], "Suitability_Class": si["label"],
            "Storage_m3": pond["target_storage"], "Storage_Limiting_Factor": pond["limiting_factor"],
            "Bottom_Length_m": pond["bottom_length"], "Bottom_Breadth_m": pond["bottom_breadth"],
            "Top_Length_m": pond["top_length"], "Top_Breadth_m": pond["top_breadth"],
            "Water_Depth_m": pond["water_depth"], "Freeboard_m": pond["freeboard"],
            "Side_Slope_H_per_V": pond["side_slope"],
            "Excavation_Depth_m": pond["excavation_depth"],
            "Excavation_Volume_m3": pond["excavation_volume"],
            "Pond_Footprint_m2": pond["footprint"],
            "Soil_Texture": a["soil_info"]["texture"], "Landform": a["geom_info"]["landform"],
            "Lithology": a["lith_info"]["lithology"],
            "Shear_Zone_Distance_m": a["shear_distance"],
            "Lineament_Density": a["lineament_density"],
            "Terrain_Constraint": screening["terrain_constraint"],
            "Seepage_Risk": screening["seepage_risk"],
            "Lining_Area_m2": pond["lining_area"],
            "Liner_Procurement_Area_m2": pond["liner_purchase_area"],
            "Cost_Total_INR": cost["total"],
        }
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", (a["farmer"] or "farmer")).strip("_") or "farmer"

        e1, e2, e3 = st.columns(3)
        e1.download_button("📄 Printable report (HTML)", data=build_html_report(a),
                           file_name=f"GeoPond_Report_{safe_name}.html", mime="text/html",
                           use_container_width=True, type="primary")
        e2.download_button("📊 Assessment table (CSV)",
                           data=pd.DataFrame([output]).to_csv(index=False).encode("utf-8"),
                           file_name=f"GeoPond_Assessment_{safe_name}.csv", mime="text/csv",
                           use_container_width=True)
        e3.download_button("🗺 GIS layers (GeoJSON)", data=build_geojson_export(a),
                           file_name=f"GeoPond_Layers_{safe_name}.geojson", mime="application/geo+json",
                           use_container_width=True)
        st.caption("The HTML report opens in any browser and can be printed or saved as PDF (Ctrl/⌘ + P). "
                   "The GeoJSON (field, pond footprint, ranked candidates) opens in QGIS / ArcGIS / Google Earth.")

        st.markdown(
            """<div class="gp-card" style="margin-top:12px"><div class="gp-card-title">🌱 Construction & management guidance</div>
            <div style="font-size:13px;line-height:1.8">
            • Confirm the candidate position by field survey and micro-topographic levelling.<br>
            • Verify ownership and boundary clearance with the farmer.<br>
            • Do not obstruct a natural stream — keep the pond off-stream.<br>
            • Provide a sediment trap at the inlet and a safe, protected overflow.<br>
            • Inspect foundation and excavated material; test field permeability before choosing a liner.<br>
            • Orient the pond along the local contour and drainage direction.<br>
            • Stabilise bunds and disturbed soil with vegetation; fence for safety.</div></div>""",
            unsafe_allow_html=True,
        )

    # ---- 8. technical --------------------------------------------------
    with tabs[7]:
        fs = a["field_stats"]
        t1, t2 = st.columns(2, gap="medium")
        with t1:
            st.markdown(
                f"""<div class="gp-card"><div class="gp-card-title">🔬 Raster sampling inside field</div>
                <div class="kv">
                <div><span>Valid grid cells</span><b>{fs['cells']}</b></div>
                <div><span>Cell size</span><b>{fs['cell_size']:.1f} (raster units)</b></div>
                <div><span>Elevation range</span><b>{fs['elev_min']:.1f} – {fs['elev_max']:.1f} m</b></div>
                <div><span>Relief</span><b>{fs['elev_max'] - fs['elev_min']:.1f} m</b></div>
                <div><span>Mean slope</span><b>{fs['slope_mean']:.2f}°</b></div>
                <div><span>Mean TWI</span><b>{fs['twi_mean']:.2f}</b></div>
                <div><span>Drainage density</span><b>{site['drainage_density']:.3f}</b></div>
                <div><span>Flow accumulation</span><b>{fmt(site['flow_accumulation'], '{:,.1f}', 'No data')}</b></div>
                <div><span>Raw site score</span><b>{site['site_score']:.3f}</b></div>
                <div><span>Adjusted site score</span><b>{site['adjusted_score']:.3f}</b></div>
                </div></div>""",
                unsafe_allow_html=True,
            )
            if fs["cells"] < 6:
                st.warning("Fewer than 6 raster cells cover the field; within-field ranking has low resolution.")
        with t2:
            st.markdown("**Run parameters**")
            st.json(a["params"])
            st.markdown("**Data provenance**")
            st.markdown(
                "- Terrain: `03_Slope`, `05_Elevation`, `10_TWI`, `flow_accumulation`\n"
                "- Hydrology: `04_Rainfall`, `CN_II`, `mean_runoff_mm`, `dependable_runoff_mm`, `runoff_coefficient`\n"
                "- Geology: Soil 1:50k, Geomorphology, Lithology, Shear zones, `08_LineamentDensity`\n"
                + "".join(f"- LULC {y}: `{LULC_SOURCES[y]['source']}`\n" for y in sorted(LULC_SOURCES)) +
                "- Analysis CRS: EPSG:32646 (WGS 84 / UTM 46N)"
            )

    st.warning(
        "Engineering note: GeoPond-DSS provides GIS-based screening and preliminary planning only. "
        "The selected coordinate represents a GIS candidate zone rather than a surveyed construction peg. "
        "Final pond siting, excavation, embankment, inlet, outlet, spillway and lining require field survey "
        "and engineering verification."
    )


# ============================================================
# 25. FOOTER
# ============================================================

st.markdown(
    f"""<div style="text-align:center;font-size:11.5px;color:#7a8f89;margin-top:26px;padding-top:12px;
    border-top:1px solid #dbe7e3">GeoPond-DSS v{APP_VERSION} · ICAR Research Complex for NEH Region, Umiam,
    Meghalaya · Basemaps © Esri, © OpenStreetMap contributors</div>""",
    unsafe_allow_html=True,
)
