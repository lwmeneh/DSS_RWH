from pathlib import Path
import base64
import math
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import xy
from rasterio.warp import reproject, Resampling

import streamlit as st
import folium
from folium.plugins import Draw, Fullscreen, LocateControl
from streamlit_folium import st_folium

from shapely.geometry import shape, mapping, Point, box
from shapely.ops import unary_union
from pyproj import Transformer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoPond-DSS",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "cache"
ASSETS_DIR = BASE_DIR / "assets"
ICAR_LOGO = ASSETS_DIR / "icar_logo.png"

WORKING_CRS = "EPSG:32646"
MAP_CENTER = [25.88, 91.88]
MAP_ZOOM = 12

# Continuous / hydrological rasters
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

# Core vectors
BOUNDARY_GPKG = CACHE_DIR / "boundary_utm46.gpkg"
STREAMS_GPKG = CACHE_DIR / "streams_utm46.gpkg"
LULC_GPKG = CACHE_DIR / "lulc_2021_10k_clean_utm46.gpkg"
MGNREGA_GPKG = CACHE_DIR / "mgnrega_utm46.gpkg"
FINAL94_GPKG = CACHE_DIR / "final94_utm46.gpkg"
PUBLISHED_GPKG = CACHE_DIR / "published_sites_utm46.gpkg"

# Original thematic vectors
SOIL_GPKG = CACHE_DIR / "Soil_Meghalaya_50k.gpkg"
GEOMORPHOLOGY_GPKG = CACHE_DIR / "Geomorphology.gpkg"
LITHOLOGY_GPKG = CACHE_DIR / "Lithology.gpkg"
SHEAR_ZONE_GPKG = CACHE_DIR / "Shear_Zone.gpkg"
SOIL_C_HORIZON_GPKG = CACHE_DIR / "Soil_C_Horizon.gpkg"


# ============================================================
# 3. CSS / MOBILE UI
# ============================================================

st.markdown(
    """
<style>
.block-container {
    padding-top: 0.45rem;
    padding-left: 0.65rem;
    padding-right: 0.65rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
html, body, [class*="css"] {font-family: "Segoe UI", Arial, sans-serif;}
h1, h2, h3 {color: #143e35;}
.stButton > button, .stDownloadButton > button {
    width: 100%; min-height: 54px; border-radius: 12px;
    font-size: 16px; font-weight: 650;
}
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #d5e5df;
    border-radius: 14px; padding: 11px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.geopond-hero {
    background: linear-gradient(135deg,#07513f 0%,#08735c 55%,#10956f 100%);
    border-radius: 18px; padding: 18px 22px; color: white;
    margin-bottom: 18px; box-shadow: 0 5px 18px rgba(0,0,0,0.14);
}
.geopond-row {display:flex; align-items:center; gap:18px;}
.geopond-logo {flex:0 0 auto;}
.geopond-logo img {width:100px; max-width:100px; background:white; border-radius:13px; padding:7px;}
.geopond-icar {font-size:13px; font-weight:700; letter-spacing:.4px; opacity:.95;}
.geopond-institute {font-size:17px; font-weight:700; margin-top:2px;}
.geopond-name {font-size:31px; font-weight:850; line-height:1.05; margin-top:9px;}
.geopond-fullname {font-size:16px; font-weight:500; line-height:1.45; margin-top:5px;}
.geopond-badge {
    display:inline-block; margin-top:10px; padding:6px 12px;
    border-radius:999px; background:rgba(255,255,255,.16);
    font-size:12px; font-weight:650;
}
.gps-card {
    background:#edf7ff; border:1px solid #cfe5f6; border-left:5px solid #1373b5;
    border-radius:13px; padding:13px 15px; margin:8px 0 12px 0;
    color:#164a72; line-height:1.55;
}
.result-card {
    background:white; border:1px solid #d5e5df; border-left:5px solid #08735c;
    border-radius:13px; padding:14px; margin:8px 0;
}
@media only screen and (max-width:768px) {
    .block-container {padding-left:.25rem; padding-right:.25rem;}
    .geopond-hero {padding:15px 12px; border-radius:13px;}
    .geopond-row {flex-direction:column; text-align:center; gap:8px;}
    .geopond-logo img {width:78px; max-width:78px;}
    .geopond-icar {font-size:11px;}
    .geopond-institute {font-size:14px;}
    .geopond-name {font-size:25px;}
    .geopond-fullname {font-size:14px;}
    h1 {font-size:1.45rem !important;}
    h2 {font-size:1.20rem !important;}
    h3 {font-size:1.05rem !important;}
    .stButton > button, .stDownloadButton > button {min-height:58px; font-size:16px;}
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def image_to_base64(path: Path):
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


def norm_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def first_matching_column(columns, candidates):
    lookup = {norm_name(c): c for c in columns}
    for candidate in candidates:
        key = norm_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


# ============================================================
# 5. PROFESSIONAL HEADER
# ============================================================

logo_b64 = image_to_base64(ICAR_LOGO)
if logo_b64:
    logo_html = f'<div class="geopond-logo"><img src="data:image/png;base64,{logo_b64}" alt="ICAR Logo"></div>'
else:
    logo_html = '<div style="font-size:52px;">🌾</div>'

header_html = f"""<div class="geopond-hero">
<div class="geopond-row">
{logo_html}
<div>
<div class="geopond-icar">INDIAN COUNCIL OF AGRICULTURAL RESEARCH</div>
<div class="geopond-institute">ICAR Research Complex for NEH Region, Umiam, Meghalaya</div>
<div class="geopond-name">GeoPond-DSS</div>
<div class="geopond-fullname"><b>Geo</b>spatial <b>Pond</b> Siting and Preliminary Design Decision Support System</div>
<div class="geopond-badge">Ri Bhoi District • Meghalaya • Mobile GPS Enabled</div>
</div>
</div>
</div>"""
st.markdown(header_html, unsafe_allow_html=True)


# ============================================================
# 6. REQUIRED FILE CHECK
# ============================================================

REQUIRED_FILES = [
    SLOPE_RASTER,
    RAINFALL_RASTER,
    ELEVATION_RASTER,
    DRAINAGE_DENSITY_RASTER,
    LINEAMENT_DENSITY_RASTER,
    TWI_RASTER,
    FLOW_ACCUMULATION_RASTER,
    CN_RASTER,
    MEAN_RUNOFF_RASTER,
    DEPENDABLE_RUNOFF_RASTER,
    RUNOFF_COEFFICIENT_RASTER,
    BOUNDARY_GPKG,
    STREAMS_GPKG,
]
missing = [str(p.relative_to(BASE_DIR)) for p in REQUIRED_FILES if not p.exists()]
if missing:
    st.error("GeoPond-DSS GIS database is incomplete.")
    st.code("\n".join(missing))
    st.stop()


# ============================================================
# 7. VECTOR LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vector(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            return gdf
        return gdf
    except Exception:
        return None


boundary_utm = load_vector(str(BOUNDARY_GPKG))
streams_utm = load_vector(str(STREAMS_GPKG))
lulc_utm = load_vector(str(LULC_GPKG))
mgnrega_utm = load_vector(str(MGNREGA_GPKG))
final94_utm = load_vector(str(FINAL94_GPKG))
published_utm = load_vector(str(PUBLISHED_GPKG))
soil_gdf = load_vector(str(SOIL_GPKG))
geomorph_gdf = load_vector(str(GEOMORPHOLOGY_GPKG))
lithology_gdf = load_vector(str(LITHOLOGY_GPKG))
shear_gdf = load_vector(str(SHEAR_ZONE_GPKG))
c_horizon_gdf = load_vector(str(SOIL_C_HORIZON_GPKG))

boundary_wgs = None
if boundary_utm is not None and not boundary_utm.empty and boundary_utm.crs is not None:
    boundary_wgs = boundary_utm.to_crs("EPSG:4326")


# ============================================================
# 8. SESSION STATE
# ============================================================

for key, default in {
    "field_geometry": None,
    "assessment": None,
    "current_lulc": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_assessment():
    st.session_state.field_geometry = None
    st.session_state.assessment = None
    st.session_state.current_lulc = None


# ============================================================
# 9. GEOMETRY HELPERS
# ============================================================

def geometry_gdf(geometry_geojson, crs="EPSG:4326"):
    return gpd.GeoDataFrame({"id": [1]}, geometry=[shape(geometry_geojson)], crs=crs)


def to_utm_geometry(geometry_geojson):
    return geometry_gdf(geometry_geojson).to_crs(WORKING_CRS).geometry.iloc[0]


def field_area_ha(geometry_geojson):
    return float(to_utm_geometry(geometry_geojson).area / 10000.0)


def get_last_polygon(map_output):
    if not map_output:
        return None
    drawings = map_output.get("all_drawings") or []
    polygon = None
    for drawing in drawings:
        if not drawing:
            continue
        geom = drawing.get("geometry", {})
        if geom.get("type") in ["Polygon", "MultiPolygon"]:
            polygon = geom
    return polygon


# ============================================================
# 10. RASTER HELPERS
# ============================================================

def raster_mean(raster_path, field_geojson):
    polygon = geometry_gdf(field_geojson)
    with rasterio.open(raster_path) as src:
        geom = polygon.to_crs(src.crs).geometry.iloc[0]
        values = np.array([], dtype=float)
        try:
            clipped, _ = rio_mask(
                src,
                [mapping(geom)],
                crop=True,
                indexes=1,
                filled=False,
                all_touched=True,
            )
            values = clipped.compressed() if np.ma.isMaskedArray(clipped) else clipped.reshape(-1)
            values = values[np.isfinite(values)]
            if src.nodata is not None:
                values = values[values != src.nodata]
        except Exception:
            values = np.array([], dtype=float)

        if len(values) == 0:
            centroid = geom.centroid
            try:
                sampled = next(src.sample([(centroid.x, centroid.y)], indexes=1, masked=True))
                sampled = np.asarray(sampled, dtype=float).reshape(-1)
                sampled = sampled[np.isfinite(sampled)]
                if len(sampled) > 0:
                    values = sampled
            except Exception:
                pass

        return float(np.mean(values)) if len(values) else np.nan


def sample_raster(raster_path, x, y):
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs(WORKING_CRS, src.crs, always_xy=True)
        rx, ry = transformer.transform(x, y)
        try:
            val = next(src.sample([(rx, ry)], indexes=1, masked=True))[0]
            if np.ma.is_masked(val):
                return np.nan
            val = float(val)
            return val if np.isfinite(val) else np.nan
        except Exception:
            return np.nan


def field_reference_grid(field_geojson):
    polygon = geometry_gdf(field_geojson)
    with rasterio.open(ELEVATION_RASTER) as src:
        geom = polygon.to_crs(src.crs).geometry.iloc[0]
        data, transform = rio_mask(
            src,
            [mapping(geom)],
            crop=True,
            indexes=1,
            filled=False,
            all_touched=True,
        )
        if np.ma.isMaskedArray(data):
            arr = data.filled(np.nan).astype(float)
            valid = ~np.ma.getmaskarray(data)
        else:
            arr = data.astype(float)
            valid = np.isfinite(arr)
        return arr, valid, transform, src.crs


def raster_to_reference(raster_path, reference_shape, reference_transform, reference_crs):
    dest = np.full(reference_shape, np.nan, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        source = src.read(1).astype(np.float64)
        if src.nodata is not None:
            source[source == src.nodata] = np.nan
        reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dest


def normalize_array(array, valid_mask):
    vals = array[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros_like(array, dtype=float)
    low = float(np.percentile(vals, 5))
    high = float(np.percentile(vals, 95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(array, dtype=float)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


# ============================================================
# 11. VECTOR POINT / DISTANCE LOOKUPS
# ============================================================

def point_record(gdf, x, y, point_crs=WORKING_CRS):
    if gdf is None or gdf.empty or gdf.crs is None:
        return None
    point_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(x, y)], crs=point_crs).to_crs(gdf.crs)
    p = point_gdf.geometry.iloc[0]
    try:
        idx = list(gdf.sindex.intersection(p.bounds))
        candidates = gdf.iloc[idx] if idx else gdf
    except Exception:
        candidates = gdf
    hit = candidates[candidates.intersects(p)]
    if hit.empty:
        return None
    return hit.iloc[0]


def nearest_distance_to_layer(gdf, x, y, point_crs=WORKING_CRS):
    if gdf is None or gdf.empty or gdf.crs is None:
        return None
    point = gpd.GeoSeries([Point(x, y)], crs=point_crs).to_crs(gdf.crs).iloc[0]
    distances = gdf.geometry.distance(point)
    if distances.empty:
        return None
    # If geographic CRS, convert both to UTM for meaningful metres
    if getattr(gdf.crs, "is_geographic", False):
        gdf_m = gdf.to_crs(WORKING_CRS)
        p_m = gpd.GeoSeries([point], crs=gdf.crs).to_crs(WORKING_CRS).iloc[0]
        distances = gdf_m.geometry.distance(p_m)
    return float(distances.min())


def nearest_record(gdf, x, y, point_crs=WORKING_CRS):
    if gdf is None or gdf.empty or gdf.crs is None:
        return None, None
    gdf_m = gdf.to_crs(WORKING_CRS)
    p = Point(x, y)
    distances = gdf_m.geometry.distance(p)
    if distances.empty:
        return None, None
    idx = distances.idxmin()
    return gdf.loc[idx], float(distances.loc[idx])


def extract_record_fields(record, mapping_dict):
    if record is None:
        return {k: "Unknown" for k in mapping_dict}
    out = {}
    cols = list(record.index)
    for out_name, candidates in mapping_dict.items():
        col = first_matching_column(cols, candidates)
        out[out_name] = safe_text(record[col]) if col else "Unknown"
    return out


# ============================================================
# 12. LULC SUMMARY
# ============================================================

def mapped_lulc_summary(field_geojson):
    if lulc_utm is None or lulc_utm.empty or lulc_utm.crs is None:
        return []
    lulc_col = first_matching_column(lulc_utm.columns, ["LULC_2022", "LULC", "Class", "class_name"])
    if not lulc_col:
        return []

    field_geom = geometry_gdf(field_geojson).to_crs(lulc_utm.crs).geometry.iloc[0]
    try:
        ids = list(lulc_utm.sindex.intersection(field_geom.bounds))
        candidates = lulc_utm.iloc[ids].copy() if ids else lulc_utm.copy()
    except Exception:
        candidates = lulc_utm.copy()

    candidates = candidates[candidates.intersects(field_geom)].copy()
    if candidates.empty:
        return []

    try:
        candidates["geometry"] = candidates.geometry.make_valid()
    except Exception:
        candidates["geometry"] = candidates.geometry.buffer(0)

    candidates["_area"] = candidates.geometry.intersection(field_geom).area
    grouped = candidates.groupby(lulc_col, dropna=False)["_area"].sum().reset_index()
    total = float(grouped["_area"].sum())
    if total <= 0:
        return []

    grouped["Area (ha)"] = grouped["_area"] / 10000.0
    grouped["Percent"] = grouped["_area"] / total * 100.0
    grouped = grouped.sort_values("Percent", ascending=False)

    return [
        {
            "LULC": safe_text(row[lulc_col]),
            "Area (ha)": round(float(row["Area (ha)"]), 3),
            "Percent": round(float(row["Percent"]), 1),
        }
        for _, row in grouped.iterrows()
    ]


# ============================================================
# 13. POND SITE SELECTION
# ============================================================

def find_pond_site(field_geojson, boundary_clearance=10.0):
    elevation, elevation_valid, transform, raster_crs = field_reference_grid(field_geojson)
    shape_ref = elevation.shape
    slope = raster_to_reference(SLOPE_RASTER, shape_ref, transform, raster_crs)
    twi = raster_to_reference(TWI_RASTER, shape_ref, transform, raster_crs)
    drainage = raster_to_reference(DRAINAGE_DENSITY_RASTER, shape_ref, transform, raster_crs)
    flow = raster_to_reference(FLOW_ACCUMULATION_RASTER, shape_ref, transform, raster_crs)

    valid = elevation_valid & np.isfinite(elevation) & np.isfinite(slope) & np.isfinite(twi) & np.isfinite(drainage)
    if np.count_nonzero(valid) == 0:
        return None

    elev_search = np.where(valid, elevation, np.inf)
    lowest_idx = np.unravel_index(np.argmin(elev_search), elev_search.shape)
    lowest_x, lowest_y = xy(transform, lowest_idx[0], lowest_idx[1], offset="center")

    e_norm = normalize_array(elevation, valid)
    s_norm = normalize_array(slope, valid)
    t_norm = normalize_array(twi, valid)
    d_norm = normalize_array(drainage, valid)

    score = 0.45 * (1 - e_norm) + 0.25 * t_norm + 0.15 * (1 - s_norm) + 0.10 * d_norm

    flow_valid = valid & np.isfinite(flow)
    if np.count_nonzero(flow_valid):
        f_norm = normalize_array(np.log1p(np.maximum(flow, 0)), flow_valid)
        score = score + 0.05 * f_norm

    preferred = valid & (slope <= 8.0)
    if np.count_nonzero(preferred) == 0:
        preferred = valid & (slope <= 15.0)
    if np.count_nonzero(preferred) == 0:
        preferred = valid
    score = np.where(preferred, score, -np.inf)

    field_geom = geometry_gdf(field_geojson).to_crs(raster_crs).geometry.iloc[0]

    stream_union = None
    if streams_utm is not None and not streams_utm.empty and streams_utm.crs is not None:
        streams_local = streams_utm.to_crs(raster_crs)
        search_zone = field_geom.buffer(1000)
        try:
            ids = list(streams_local.sindex.intersection(search_zone.bounds))
            local = streams_local.iloc[ids].copy() if ids else streams_local.iloc[0:0].copy()
        except Exception:
            local = streams_local.copy()
        local = local[local.intersects(search_zone)]
        if not local.empty:
            stream_union = unary_union(local.geometry.tolist())

    ordering = np.argsort(score.ravel())[::-1]
    best = None

    for flat_idx in ordering[:2000]:
        r, c = np.unravel_index(flat_idx, score.shape)
        if not np.isfinite(score[r, c]):
            continue
        px, py = xy(transform, r, c, offset="center")
        p = Point(px, py)
        boundary_dist = float(p.distance(field_geom.boundary))
        if boundary_dist < boundary_clearance:
            continue
        stream_dist = None
        if stream_union is not None:
            stream_dist = float(p.distance(stream_union))
            if stream_dist < 10.0:
                continue

        # Soft penalty for mapped shear zone proximity if available
        shear_dist = nearest_distance_to_layer(shear_gdf, px, py)
        adjusted = float(score[r, c])
        if shear_dist is not None:
            if shear_dist < 50:
                adjusted -= 0.35
            elif shear_dist < 100:
                adjusted -= 0.20
            elif shear_dist < 250:
                adjusted -= 0.08

        # Hard avoid clearly unsuitable geomorphology where identified
        geom_rec = point_record(geomorph_gdf, px, py)
        geom_info = extract_record_fields(
            geom_rec,
            {
                "landform": ["legend_sho", "description", "descriptio", "geomorphology", "landform", "class_name"],
            },
        )
        landform_lower = geom_info["landform"].lower()
        severe_terms = ["scarp", "ridge"]
        if any(term in landform_lower for term in severe_terms):
            adjusted -= 0.25

        candidate = {
            "x": float(px),
            "y": float(py),
            "elevation": float(elevation[r, c]),
            "slope": float(slope[r, c]),
            "twi": float(twi[r, c]),
            "drainage_density": float(drainage[r, c]),
            "flow_accumulation": float(flow[r, c]) if np.isfinite(flow[r, c]) else np.nan,
            "site_score": float(score[r, c]),
            "adjusted_score": float(adjusted),
            "boundary_distance": boundary_dist,
            "stream_distance": stream_dist,
            "shear_distance": shear_dist,
        }

        if best is None or candidate["adjusted_score"] > best["adjusted_score"]:
            best = candidate

    if best is None:
        return None

    transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(best["x"], best["y"])
    low_lon, low_lat = transformer.transform(lowest_x, lowest_y)
    best["longitude"] = float(lon)
    best["latitude"] = float(lat)

    return {
        "recommended": best,
        "lowest": {
            "longitude": float(low_lon),
            "latitude": float(low_lat),
            "elevation": float(elevation[lowest_idx]),
        },
    }


# ============================================================
# 14. POND DESIGN
# ============================================================

def calculate_pond_geometry(bottom_breadth, water_depth, freeboard, side_slope, length_breadth_ratio):
    bottom_length = bottom_breadth * length_breadth_ratio
    water_top_breadth = bottom_breadth + 2 * side_slope * water_depth
    water_top_length = bottom_length + 2 * side_slope * water_depth
    bottom_area = bottom_length * bottom_breadth
    top_water_area = water_top_length * water_top_breadth
    storage_volume = water_depth / 3.0 * (bottom_area + top_water_area + math.sqrt(bottom_area * top_water_area))

    excavation_depth = water_depth + freeboard
    top_breadth = bottom_breadth + 2 * side_slope * excavation_depth
    top_length = bottom_length + 2 * side_slope * excavation_depth
    footprint = top_length * top_breadth

    side_slant = math.sqrt(water_depth ** 2 + (side_slope * water_depth) ** 2)
    long_side_area = ((bottom_length + water_top_length) / 2) * side_slant
    short_side_area = ((bottom_breadth + water_top_breadth) / 2) * side_slant
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
        "excavation_depth": float(excavation_depth),
        "volume": float(storage_volume),
        "footprint": float(footprint),
        "lining_area": float(lining_area),
        "liner_purchase_area": float(lining_area * 1.10),
    }


def design_farm_pond(available_water_m3, field_area_ha_value, storage_fraction, maximum_land_fraction,
                     water_depth, freeboard, side_slope, length_breadth_ratio):
    field_area_m2 = field_area_ha_value * 10000.0
    requested_storage = available_water_m3 * storage_fraction
    max_footprint = field_area_m2 * maximum_land_fraction

    low, high = 0.5, 120.0
    for _ in range(80):
        mid = (low + high) / 2
        geom = calculate_pond_geometry(mid, water_depth, freeboard, side_slope, length_breadth_ratio)
        if geom["footprint"] > max_footprint:
            high = mid
        else:
            low = mid

    max_geom = calculate_pond_geometry(low, water_depth, freeboard, side_slope, length_breadth_ratio)
    target_storage = min(requested_storage, max_geom["volume"])

    lower, upper = 0.5, max(low, 0.5)
    for _ in range(80):
        mid = (lower + upper) / 2
        geom = calculate_pond_geometry(mid, water_depth, freeboard, side_slope, length_breadth_ratio)
        if geom["volume"] > target_storage:
            upper = mid
        else:
            lower = mid

    result = calculate_pond_geometry(lower, water_depth, freeboard, side_slope, length_breadth_ratio)
    result["requested_storage"] = float(requested_storage)
    result["target_storage"] = float(target_storage)
    result["maximum_storage_by_land"] = float(max_geom["volume"])
    result["occupied_percent"] = float(result["footprint"] / field_area_m2 * 100.0)
    result["remaining_area_ha"] = float(max(field_area_m2 - result["footprint"], 0.0) / 10000.0)
    return result


# ============================================================
# 15. THEMATIC INTERPRETATION
# ============================================================

def classify_soil_texture(texture):
    t = safe_text(texture).lower()
    if t == "unknown":
        return "Unknown", 1
    if any(k in t for k in ["clay", "clayey"]):
        # sandy clay handled below
        if "sandy" not in t:
            return "Low to moderate seepage tendency", 0
    if any(k in t for k in ["silty clay loam", "silt clay loam", "silt loam", "fine loamy", "fine mixed", "fine"]):
        return "Moderate seepage tendency", 1
    if any(k in t for k in ["loam", "loamy", "skeletal"]):
        if "sandy" not in t:
            return "Moderate to high seepage tendency", 2
    if any(k in t for k in ["sandy clay", "sandy loam", "sandy clay loam"]):
        return "High seepage tendency", 3
    if "misc" in t:
        return "Field verification required", 1
    return "Field verification required", 1


def terrain_constraint_from_geomorph(landform, slope_deg):
    text = safe_text(landform).lower()
    severe = ["scarp", "ridge"]
    moderate = ["highly dissected", "structural hills", "steep"]
    if any(k in text for k in severe) or slope_deg >= 24:
        return "SEVERE", 3
    if any(k in text for k in moderate) or slope_deg >= 14:
        return "HIGH", 2
    if slope_deg >= 8:
        return "MODERATE", 1
    return "LOW", 0


def shear_zone_class(distance_m):
    if distance_m is None:
        return "Unknown", 0
    if distance_m < 50:
        return "VERY HIGH CONCERN", 3
    if distance_m < 100:
        return "HIGH CONCERN", 2
    if distance_m < 250:
        return "MODERATE CONCERN", 1
    return "LOWER MAPPED CONCERN", 0


def lithology_permeability_score(lithology_text):
    t = safe_text(lithology_text).lower()
    if t == "unknown":
        return 1, "Lithology unavailable; field verification required."
    high_terms = ["sandstone", "grit", "conglomerate", "limestone", "alluv", "gravel", "sand"]
    low_terms = ["shale", "clay", "phyllite", "schist"]
    if any(k in t for k in high_terms):
        return 2, "Mapped lithology may permit higher seepage or fracture-related losses; verify in the field."
    if any(k in t for k in low_terms):
        return 1, "Mapped lithology is not automatically impermeable; weathering and fractures still require verification."
    return 1, "Lithology requires field verification for weathering, joints and permeability."


def build_geotechnical_screening(soil_texture, landform, lithology_text, shear_distance, lineament_density, slope_deg):
    soil_desc, soil_score = classify_soil_texture(soil_texture)
    terrain_class, terrain_score = terrain_constraint_from_geomorph(landform, slope_deg)
    shear_class, shear_score = shear_zone_class(shear_distance)
    lith_score, lith_note = lithology_permeability_score(lithology_text)

    lineament_score = 0
    lineament_note = "Lineament density unavailable."
    if np.isfinite(lineament_density):
        if lineament_density >= 0.55:
            lineament_score = 2
            lineament_note = "High lineament density indicates elevated fracture-related seepage concern."
        elif lineament_density >= 0.19:
            lineament_score = 1
            lineament_note = "Moderate lineament density indicates some fracture-related seepage concern."
        else:
            lineament_note = "Low lineament density reduces mapped fracture-related seepage concern."

    seep_score = soil_score + shear_score + lith_score + lineament_score
    if seep_score >= 6:
        seepage = "HIGH"
    elif seep_score >= 3:
        seepage = "MODERATE"
    else:
        seepage = "LOW"

    if seepage == "HIGH":
        lining = (
            "High seepage-risk screening: lining is likely to be required, subject to field permeability testing, "
            "excavation-scale geological inspection and suitable subgrade preparation."
        )
    elif seepage == "MODERATE":
        lining = (
            "Moderate seepage-risk screening: assess field permeability and available clay material. "
            "Compacted clay or geomembrane lining may be required depending on field results."
        )
    else:
        lining = (
            "Lower GIS-screened seepage concern. Field permeability testing is still required before deciding that lining is unnecessary."
        )

    return {
        "soil_screen": soil_desc,
        "terrain_constraint": terrain_class,
        "shear_concern": shear_class,
        "seepage_risk": seepage,
        "lining_recommendation": lining,
        "notes": [lith_note, lineament_note],
    }


# ============================================================
# 16. FARMER INFO
# ============================================================

with st.expander("👨‍🌾 Farmer / Field Information", expanded=False):
    c1, c2 = st.columns(2)
    farmer_name = c1.text_input("Farmer Name")
    village = c2.text_input("Village / Locality")


# ============================================================
# 17. FIELD LOCATION / GPS
# ============================================================

st.header("1️⃣ Locate and Draw Farmer Field")
st.markdown(
    """<div class="gps-card">
<b>📍 Mobile workflow</b><br>
1. Stand in or near the farmer's field.<br>
2. Tap the <b>GPS/location button</b> on the map.<br>
3. Allow location permission in the mobile browser.<br>
4. Keep <b>Satellite Imagery</b> selected and zoom in.<br>
5. Use the polygon tool to trace the actual field boundary.<br>
6. Close the polygon by tapping the first point again.
</div>""",
    unsafe_allow_html=True,
)

field_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=None, control_scale=True, prefer_canvas=True)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite Imagery",
    overlay=False,
    control=True,
    show=True,
    max_zoom=20,
).add_to(field_map)
folium.TileLayer("OpenStreetMap", name="Road / Village Map", overlay=False, control=True, show=False).add_to(field_map)

if boundary_wgs is not None:
    folium.GeoJson(
        boundary_wgs.__geo_interface__,
        name="Ri Bhoi District Boundary",
        style_function=lambda f: {"color": "#FFD600", "weight": 2, "fillOpacity": 0},
    ).add_to(field_map)

LocateControl(
    auto_start=False,
    position="topleft",
    flyTo=True,
    keepCurrentZoomLevel=False,
    strings={"title": "Use my GPS location", "popup": "You are approximately here"},
    locateOptions={"enableHighAccuracy": True, "maximumAge": 0, "timeout": 15000},
).add_to(field_map)

if st.session_state.field_geometry:
    folium.GeoJson(
        st.session_state.field_geometry,
        name="Farmer Field",
        style_function=lambda f: {
            "color": "#00C853", "weight": 5, "fillColor": "#69F0AE", "fillOpacity": 0.30
        },
    ).add_to(field_map)

Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False,
        "rectangle": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": {
            "allowIntersection": False,
            "showArea": True,
            "metric": True,
            "shapeOptions": {"color": "#00C853", "weight": 5, "fillColor": "#69F0AE", "fillOpacity": 0.30},
        },
    },
    edit_options={"edit": True, "remove": True},
).add_to(field_map)

Fullscreen(position="topleft").add_to(field_map)
folium.LayerControl(position="topright", collapsed=True).add_to(field_map)

map_output = st_folium(
    field_map,
    height=570,
    use_container_width=True,
    key="geopond_field_map",
    returned_objects=["all_drawings", "last_active_drawing"],
)

new_field = get_last_polygon(map_output)
if new_field is not None and new_field != st.session_state.field_geometry:
    st.session_state.field_geometry = new_field
    st.session_state.assessment = None
    st.rerun()

st.caption(
    "📱 If GPS does not work, allow Location permission for the browser and tap the GPS button again. "
    "Use Satellite Imagery and zoom to approximately level 17–20 for field-boundary identification."
)

field = st.session_state.field_geometry


# ============================================================
# 18. FIELD INFO / LAND USE
# ============================================================

if field:
    area_ha = field_area_ha(field)
    lulc_comp = mapped_lulc_summary(field)
    mapped_lulc = lulc_comp[0]["LULC"] if lulc_comp else "Unknown"

    st.success(f"✅ Farmer field captured successfully: {area_ha:.3f} ha")
    c1, c2 = st.columns(2)
    c1.metric("Field Area", f"{area_ha:.3f} ha")
    c2.metric("GIS-Mapped LULC", mapped_lulc)

    LAND_USE_OPTIONS = [
        "Crop land", "Plantation", "Shifting Cultivation", "Scrub land", "Forest Open",
        "Forest Moderately Dense", "Forest Dense", "Built up", "Waterbody", "River/Stream"
    ]
    if st.session_state.current_lulc in LAND_USE_OPTIONS:
        default_idx = LAND_USE_OPTIONS.index(st.session_state.current_lulc)
    elif mapped_lulc in LAND_USE_OPTIONS:
        default_idx = LAND_USE_OPTIONS.index(mapped_lulc)
    else:
        default_idx = 0

    current_lulc = st.selectbox(
        "🌾 Confirm Present Land Use",
        LAND_USE_OPTIONS,
        index=default_idx,
        help="If mapped LULC differs from the current field condition, select the present land use observed in the field.",
    )
    st.session_state.current_lulc = current_lulc

    if current_lulc != mapped_lulc:
        st.warning(
            f"GIS layer indicates **{mapped_lulc}**, while present land use is confirmed as **{current_lulc}**. "
            "GeoPond-DSS retains the GIS class but uses the confirmed present land use for planning."
        )

    if lulc_comp:
        with st.expander("🗺️ LULC composition inside field"):
            st.dataframe(pd.DataFrame(lulc_comp), use_container_width=True, hide_index=True)


# ============================================================
# 19. PLANNING SETTINGS
# ============================================================

if field:
    st.markdown("---")
    st.header("2️⃣ Preliminary Pond Planning Settings")
    with st.expander("⚙️ Adjust planning assumptions", expanded=False):
        storage_fraction_percent = st.slider("One-time storage as % of 75% dependable annual runoff", 10, 60, 30, 5)
        maximum_land_percent = st.slider("Maximum field area occupied by pond (%)", 5, 20, 10, 1)
        water_depth = st.slider("Planning water depth (m)", 1.5, 3.5, 2.5, 0.25)
        freeboard = st.slider("Freeboard (m)", 0.30, 0.75, 0.50, 0.05)
        side_slope = st.slider("Side slope (H:1V)", 1.0, 2.5, 1.5, 0.25)
        length_breadth_ratio = st.slider("Bottom length : breadth ratio", 1.0, 2.0, 1.5, 0.1)


# ============================================================
# 20. RUN DSS
# ============================================================

if field:
    st.markdown("---")
    st.header("3️⃣ Run GeoPond-DSS")

    if st.button("💧 FIND BEST GIS-SCREENED POND SITE & PRELIMINARY DESIGN", type="primary", use_container_width=True):
        with st.spinner("Analysing water availability, terrain, geology, soil and pond design..."):
            area_ha = field_area_ha(field)
            rainfall = raster_mean(RAINFALL_RASTER, field)
            mean_runoff = raster_mean(MEAN_RUNOFF_RASTER, field)
            dependable_runoff = raster_mean(DEPENDABLE_RUNOFF_RASTER, field)
            cn = raster_mean(CN_RASTER, field)
            runoff_coeff = raster_mean(RUNOFF_COEFFICIENT_RASTER, field)

            mean_volume = mean_runoff * area_ha * 10.0 if np.isfinite(mean_runoff) else np.nan
            dependable_volume = dependable_runoff * area_ha * 10.0 if np.isfinite(dependable_runoff) else np.nan
            available_water = dependable_volume if np.isfinite(dependable_volume) else mean_volume
            if not np.isfinite(available_water) or available_water <= 0:
                st.error("Valid runoff data could not be obtained for this field.")
                st.stop()

            pond = design_farm_pond(
                available_water_m3=available_water,
                field_area_ha_value=area_ha,
                storage_fraction=storage_fraction_percent / 100.0,
                maximum_land_fraction=maximum_land_percent / 100.0,
                water_depth=water_depth,
                freeboard=freeboard,
                side_slope=side_slope,
                length_breadth_ratio=length_breadth_ratio,
            )

            half_diag = 0.5 * math.sqrt(pond["top_length"] ** 2 + pond["top_breadth"] ** 2)
            site_result = find_pond_site(field, boundary_clearance=max(10.0, half_diag + 2.0))
            if site_result is None:
                st.error("GeoPond-DSS could not identify a valid GIS candidate inside the selected field.")
                st.stop()

            site = site_result["recommended"]
            lowest = site_result["lowest"]

            # Original thematic lookups
            soil_rec = point_record(soil_gdf, site["x"], site["y"])
            geom_rec = point_record(geomorph_gdf, site["x"], site["y"])
            lith_rec = point_record(lithology_gdf, site["x"], site["y"])
            c_rec, c_dist = nearest_record(c_horizon_gdf, site["x"], site["y"])

            soil_info = extract_record_fields(
                soil_rec,
                {
                    "texture": ["Texture", "soil_texture", "texture_class"],
                    "soil_name": ["Soil", "soil_name", "soilseries", "soil_series", "mapping_unit"],
                },
            )
            geom_info = extract_record_fields(
                geom_rec,
                {
                    "landform": ["legend_sho", "description", "descriptio", "geomorphology", "landform", "class_name"],
                    "geomorph_desc": ["description", "descriptio", "legend_long", "class_desc"],
                },
            )
            lith_info = extract_record_fields(
                lith_rec,
                {
                    "group": ["group_name", "group", "grp_name"],
                    "formation": ["formation", "formation_name"],
                    "member": ["member", "member_name"],
                    "lithology": ["lithologic", "lithology", "rock_type", "description", "descriptio"],
                    "age": ["age", "geological_age", "stratigrap"],
                    "intrusive": ["intrusive", "intrusion"],
                },
            )
            c_info = extract_record_fields(
                c_rec,
                {
                    "sample_id": ["sample_id", "sampleid", "id", "sample_no", "sample"],
                },
            )

            shear_distance = nearest_distance_to_layer(shear_gdf, site["x"], site["y"])
            lineament_density = sample_raster(LINEAMENT_DENSITY_RASTER, site["x"], site["y"])

            screening = build_geotechnical_screening(
                soil_texture=soil_info["texture"],
                landform=geom_info["landform"],
                lithology_text=lith_info["lithology"],
                shear_distance=shear_distance,
                lineament_density=lineament_density,
                slope_deg=site["slope"],
            )

            st.session_state.assessment = {
                "farmer": farmer_name,
                "village": village,
                "field_area": area_ha,
                "mapped_lulc": mapped_lulc,
                "current_lulc": current_lulc,
                "rainfall": rainfall,
                "mean_runoff": mean_runoff,
                "dependable_runoff": dependable_runoff,
                "mean_runoff_volume": mean_volume,
                "dependable_volume": dependable_volume,
                "cn": cn,
                "runoff_coefficient": runoff_coeff,
                "site": site,
                "lowest": lowest,
                "pond": pond,
                "soil_info": soil_info,
                "geom_info": geom_info,
                "lith_info": lith_info,
                "c_horizon_info": c_info,
                "c_horizon_distance": c_dist,
                "shear_distance": shear_distance,
                "lineament_density": lineament_density,
                "screening": screening,
                "nearest_stream": nearest_distance_to_layer(streams_utm, site["x"], site["y"]),
            }
        st.rerun()


# ============================================================
# 21. RESULTS
# ============================================================

assessment = st.session_state.assessment
if assessment:
    site = assessment["site"]
    lowest = assessment["lowest"]
    pond = assessment["pond"]
    screening = assessment["screening"]
    soil_info = assessment["soil_info"]
    geom_info = assessment["geom_info"]
    lith_info = assessment["lith_info"]

    st.markdown("---")
    st.header("✅ GeoPond-DSS Assessment Result")
    st.success(
        "### Best GIS-screened candidate site\n"
        "This location is the highest-ranked GIS candidate within the farmer field after terrain, hydrology, boundary, stream and geological screening. "
        "It is not a final construction peg."
    )

    st.subheader("🌧️ Water Availability")
    c1, c2 = st.columns(2)
    c1.metric("Field Area", f"{assessment['field_area']:.3f} ha")
    c2.metric("Spatial Annual Rainfall", f"{assessment['rainfall']:.0f} mm/year" if np.isfinite(assessment['rainfall']) else "No data")
    c1, c2 = st.columns(2)
    c1.metric("Mean Annual Runoff", f"{assessment['mean_runoff']:.0f} mm" if np.isfinite(assessment['mean_runoff']) else "No data")
    c2.metric("75% Dependable Runoff", f"{assessment['dependable_runoff']:.0f} mm" if np.isfinite(assessment['dependable_runoff']) else "No data")
    c1, c2 = st.columns(2)
    c1.metric("Mean Annual Runoff Volume", f"{assessment['mean_runoff_volume']:,.0f} m³/year")
    c2.metric("75% Dependable Water", f"{assessment['dependable_volume']:,.0f} m³/year")
    st.caption("Rainfall retains the information content of the source rainfall dataset; resampling does not create true 30-m rainfall observations.")

    st.subheader("📍 Best GIS-Screened Pond Candidate")
    c1, c2 = st.columns(2)
    c1.metric("Latitude", f"{site['latitude']:.6f}")
    c2.metric("Longitude", f"{site['longitude']:.6f}")
    c1, c2 = st.columns(2)
    c1.metric("Elevation", f"{site['elevation']:.1f} m")
    c2.metric("Local Slope", f"{site['slope']:.2f}°")
    c1, c2 = st.columns(2)
    c1.metric("TWI", f"{site['twi']:.2f}")
    c2.metric("Boundary Clearance", f"{site['boundary_distance']:.1f} m")
    if site.get("stream_distance") is not None:
        st.metric("Distance to Mapped Stream", f"{site['stream_distance']:.1f} m")

    # Result map
    result_map = folium.Map(location=[site["latitude"], site["longitude"]], zoom_start=18, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite Imagery", overlay=False, show=True, max_zoom=20
    ).add_to(result_map)
    folium.TileLayer("OpenStreetMap", name="Road / Village Map", overlay=False, show=False).add_to(result_map)
    folium.GeoJson(
        field, name="Farmer Field",
        style_function=lambda f: {"color":"#00C853","weight":5,"fillColor":"#69F0AE","fillOpacity":0.18}
    ).add_to(result_map)

    pond_box_utm = box(
        site["x"] - pond["top_length"] / 2,
        site["y"] - pond["top_breadth"] / 2,
        site["x"] + pond["top_length"] / 2,
        site["y"] + pond["top_breadth"] / 2,
    )
    pond_box_wgs = gpd.GeoSeries([pond_box_utm], crs=WORKING_CRS).to_crs("EPSG:4326").iloc[0]
    folium.GeoJson(
        mapping(pond_box_wgs),
        name="Indicative Pond Footprint",
        style_function=lambda f: {"color":"#D50000","weight":3,"fillColor":"#FF5252","fillOpacity":0.30},
        tooltip="Indicative pond footprint — not construction alignment",
    ).add_to(result_map)
    folium.Marker(
        [site["latitude"], site["longitude"]],
        tooltip="Best GIS-screened candidate",
        icon=folium.Icon(color="red", icon="tint"),
    ).add_to(result_map)
    folium.Marker(
        [lowest["latitude"], lowest["longitude"]],
        tooltip="Absolute lowest DEM cell",
        icon=folium.Icon(color="orange"),
    ).add_to(result_map)
    Fullscreen(position="topleft").add_to(result_map)
    folium.LayerControl(collapsed=True).add_to(result_map)
    st_folium(result_map, height=570, use_container_width=True, key="geopond_result_map")
    st.caption("The rectangle is an indicative footprint only; final pond orientation should follow field survey, local contour and safe inlet/outlet arrangement.")

    st.subheader("📐 Preliminary Pond Design")
    c1, c2 = st.columns(2)
    c1.metric("Planning Storage Target", f"{pond['target_storage']:,.0f} m³")
    c2.metric("Water Depth", f"{pond['water_depth']:.2f} m")
    c1, c2 = st.columns(2)
    c1.metric("Bottom Length", f"{pond['bottom_length']:.1f} m")
    c2.metric("Bottom Breadth", f"{pond['bottom_breadth']:.1f} m")
    c1, c2 = st.columns(2)
    c1.metric("Top Length", f"{pond['top_length']:.1f} m")
    c2.metric("Top Breadth", f"{pond['top_breadth']:.1f} m")
    c1, c2 = st.columns(2)
    c1.metric("Excavation Depth", f"{pond['excavation_depth']:.2f} m")
    c2.metric("Freeboard", f"{pond['freeboard']:.2f} m")
    c1, c2 = st.columns(2)
    c1.metric("Pond Footprint", f"{pond['footprint']:,.0f} m²")
    c2.metric("Field Area Occupied", f"{pond['occupied_percent']:.1f}%")
    st.caption("Planning storage target is derived from the selected fraction of 75% dependable annual runoff, subject to the maximum land-area constraint.")

    st.subheader("🪨 Geological, Terrain & Seepage Screening")
    c1, c2 = st.columns(2)
    c1.metric("Soil Texture", soil_info["texture"])
    c2.metric("Soil Screening", screening["soil_screen"])
    c1, c2 = st.columns(2)
    c1.metric("Geomorphology", geom_info["landform"])
    c2.metric("Terrain Constraint", screening["terrain_constraint"])
    c1, c2 = st.columns(2)
    c1.metric("Lithological Group", lith_info["group"])
    c2.metric("Formation", lith_info["formation"])
    st.write("**Lithology:**", lith_info["lithology"])
    st.write("**Geological age:**", lith_info["age"])

    c1, c2 = st.columns(2)
    shear_text = f"{assessment['shear_distance']:.0f} m" if assessment['shear_distance'] is not None else "No data"
    c1.metric("Nearest Shear Zone", shear_text)
    c2.metric("Shear-Zone Screening", screening["shear_concern"])

    c1, c2 = st.columns(2)
    c1.metric(
        "Lineament Density",
        f"{assessment['lineament_density']:.3f}" if np.isfinite(assessment['lineament_density']) else "No data"
    )
    c2.metric("Seepage Screening", screening["seepage_risk"])

    if assessment.get("c_horizon_distance") is not None:
        st.write(
            f"**Nearest C-horizon geochemical sample:** {assessment['c_horizon_distance']:.0f} m "
            f"(Sample ID: {assessment['c_horizon_info'].get('sample_id','Unknown')})"
        )
        st.caption("C-horizon data are ancillary geochemical observations and are not used as a substitute for mapped soil texture or field permeability testing.")

    if screening["terrain_constraint"] in ["SEVERE", "HIGH"]:
        st.error("Terrain constraint detected. An alternative pond location should be considered unless field survey confirms that construction is safe and practical.")

    if screening["seepage_risk"] == "HIGH":
        st.error("GIS seepage screening: HIGH")
    elif screening["seepage_risk"] == "MODERATE":
        st.warning("GIS seepage screening: MODERATE")
    else:
        st.success("GIS seepage screening: LOW")

    for note in screening["notes"]:
        st.write("•", note)
    st.info(screening["lining_recommendation"])

    c1, c2 = st.columns(2)
    c1.metric("Estimated Wetted Lining Area", f"{pond['lining_area']:,.0f} m²")
    c2.metric("Indicative Procurement Area (+10%)", f"{pond['liner_purchase_area']:,.0f} m²")

    with st.expander("🔬 Technical Site Information"):
        st.write("**Farmer:**", assessment["farmer"] or "Not entered")
        st.write("**Village / Locality:**", assessment["village"] or "Not entered")
        st.write("**Current land use:**", assessment["current_lulc"])
        st.write("**GIS-mapped LULC:**", assessment["mapped_lulc"])
        st.write("**CN-II:**", round(assessment["cn"], 1) if np.isfinite(assessment["cn"]) else "No data")
        st.write("**Runoff coefficient:**", round(assessment["runoff_coefficient"], 3) if np.isfinite(assessment["runoff_coefficient"]) else "No data")
        st.write("**Drainage density at site:**", round(site["drainage_density"], 3))
        st.write("**Flow accumulation at site:**", round(site["flow_accumulation"], 2) if np.isfinite(site["flow_accumulation"]) else "No data")
        st.write("**Raw site score:**", round(site["site_score"], 3))
        st.write("**Adjusted site score:**", round(site["adjusted_score"], 3))

    st.subheader("🌱 Construction & Management Guidance")
    st.markdown(
        """<div class="result-card">
<b>Farm Pond Planning Guidance</b><br><br>
• Confirm the candidate site by field survey and micro-topographic levelling.<br>
• Verify ownership and boundary clearance before excavation.<br>
• Avoid obstruction of a natural stream for an off-stream farm pond.<br>
• Provide a sediment trap at the inlet and a safe overflow/spill arrangement.<br>
• Inspect excavation material and foundation for joints, fractures and weathered zones.<br>
• Conduct a field permeability/infiltration test before final lining selection.<br>
• Orient the final pond layout to the local contour and practical inlet/outlet arrangement.<br>
• Stabilize bunds and disturbed soil with appropriate vegetation.<br>
</div>""",
        unsafe_allow_html=True,
    )

    output = {
        "DSS": "GeoPond-DSS",
        "Farmer": assessment["farmer"],
        "Village": assessment["village"],
        "Field_Area_ha": assessment["field_area"],
        "GIS_LULC": assessment["mapped_lulc"],
        "Confirmed_Land_Use": assessment["current_lulc"],
        "Rainfall_mm_year": assessment["rainfall"],
        "Mean_Runoff_mm": assessment["mean_runoff"],
        "Dependable_Runoff_mm": assessment["dependable_runoff"],
        "Dependable_Water_m3_year": assessment["dependable_volume"],
        "CN_II": assessment["cn"],
        "Runoff_Coefficient": assessment["runoff_coefficient"],
        "Pond_Latitude": site["latitude"],
        "Pond_Longitude": site["longitude"],
        "Elevation_m": site["elevation"],
        "Slope_degree": site["slope"],
        "TWI": site["twi"],
        "Drainage_Density": site["drainage_density"],
        "Flow_Accumulation": site["flow_accumulation"],
        "Raw_Site_Score": site["site_score"],
        "Adjusted_Site_Score": site["adjusted_score"],
        "Soil_Texture": soil_info["texture"],
        "Soil_Screening": screening["soil_screen"],
        "Geomorphology": geom_info["landform"],
        "Terrain_Constraint": screening["terrain_constraint"],
        "Lithology_Group": lith_info["group"],
        "Formation": lith_info["formation"],
        "Lithology": lith_info["lithology"],
        "Geological_Age": lith_info["age"],
        "Nearest_Shear_Zone_m": assessment["shear_distance"],
        "Shear_Zone_Screening": screening["shear_concern"],
        "Lineament_Density": assessment["lineament_density"],
        "Seepage_Screening": screening["seepage_risk"],
        "Storage_m3": pond["target_storage"],
        "Bottom_Length_m": pond["bottom_length"],
        "Bottom_Breadth_m": pond["bottom_breadth"],
        "Top_Length_m": pond["top_length"],
        "Top_Breadth_m": pond["top_breadth"],
        "Water_Depth_m": pond["water_depth"],
        "Freeboard_m": pond["freeboard"],
        "Excavation_Depth_m": pond["excavation_depth"],
        "Pond_Footprint_m2": pond["footprint"],
        "Field_Occupied_percent": pond["occupied_percent"],
        "Lining_Area_m2": pond["lining_area"],
        "Liner_Procurement_Area_m2": pond["liner_purchase_area"],
        "Lining_Recommendation": screening["lining_recommendation"],
    }

    csv_data = pd.DataFrame([output]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download GeoPond-DSS Farmer Assessment",
        data=csv_data,
        file_name="GeoPond_DSS_Farmer_Assessment.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.warning(
        "**Engineering note:** GeoPond-DSS provides GIS-based screening and preliminary planning. "
        "The selected coordinates represent a GIS candidate zone, not a surveyed construction peg. "
        "Final siting, pond orientation, excavation, embankment, inlet, outlet, spillway, lining and structural design require field survey, permeability testing and engineering verification."
    )


# ============================================================
# 22. RESET
# ============================================================

st.markdown("---")
if st.button("🔄 Start New Farmer Assessment", use_container_width=True):
    reset_assessment()
    st.rerun()
