# ============================================================
# GeoPond-DSS
# Geospatial Pond Siting and Preliminary Design
# Decision Support System
#
# ICAR Research Complex for NEH Region
# Umiam, Meghalaya
#
# ADVANCED WEB-GIS USER INTERFACE
# ============================================================

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

from folium.plugins import (
    Draw,
    Fullscreen,
    LocateControl,
    MeasureControl,
    MiniMap,
    MousePosition,
)

from streamlit_folium import st_folium

from shapely.geometry import (
    shape,
    mapping,
    Point,
    box,
)

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

CACHE_DIR = (
    BASE_DIR
    / "data"
    / "cache"
)

ASSETS_DIR = (
    BASE_DIR
    / "assets"
)

ICAR_LOGO = (
    ASSETS_DIR
    / "icar_logo.png"
)


WORKING_CRS = "EPSG:32646"

MAP_CENTER = [
    25.88,
    91.88,
]

MAP_ZOOM = 12


# ============================================================
# HYDROLOGICAL / TERRAIN RASTERS
# ============================================================

SLOPE_RASTER = (
    CACHE_DIR
    / "03_Slope.tif"
)

RAINFALL_RASTER = (
    CACHE_DIR
    / "04_Rainfall.tif"
)

ELEVATION_RASTER = (
    CACHE_DIR
    / "05_Elevation.tif"
)

DRAINAGE_DENSITY_RASTER = (
    CACHE_DIR
    / "06_DrainageDensity.tif"
)

LINEAMENT_DENSITY_RASTER = (
    CACHE_DIR
    / "08_LineamentDensity.tif"
)

TWI_RASTER = (
    CACHE_DIR
    / "10_TWI.tif"
)

FLOW_ACCUMULATION_RASTER = (
    CACHE_DIR
    / "flow_accumulation.tif"
)

CN_RASTER = (
    CACHE_DIR
    / "CN_II.tif"
)

MEAN_RUNOFF_RASTER = (
    CACHE_DIR
    / "mean_runoff_mm.tif"
)

DEPENDABLE_RUNOFF_RASTER = (
    CACHE_DIR
    / "dependable_runoff_mm.tif"
)

RUNOFF_COEFFICIENT_RASTER = (
    CACHE_DIR
    / "runoff_coefficient.tif"
)


# ============================================================
# VECTOR DATA
# ============================================================

BOUNDARY_GPKG = (
    CACHE_DIR
    / "boundary_utm46.gpkg"
)

STREAMS_GPKG = (
    CACHE_DIR
    / "streams_utm46.gpkg"
)

LULC_2022_GPKG = (
    CACHE_DIR
    / "lulc_2021_10k_clean_utm46.gpkg"
)

PROJECTED_LULC_2025_SHP = (
    CACHE_DIR
    / "2025_Projected_LULC.shp"
)

SOIL_GPKG = (
    CACHE_DIR
    / "Soil_Meghalaya_50k.gpkg"
)

SOIL_SHP = (
    CACHE_DIR
    / "Soil_Meghalaya_50k.shp"
)

GEOMORPHOLOGY_GPKG = (
    CACHE_DIR
    / "Geomorphology.gpkg"
)

LITHOLOGY_GPKG = (
    CACHE_DIR
    / "Lithology.gpkg"
)

SHEAR_ZONE_GPKG = (
    CACHE_DIR
    / "Shear_Zone.gpkg"
)

SOIL_C_HORIZON_GPKG = (
    CACHE_DIR
    / "Soil_C_Horizon.gpkg"
)


# ============================================================
# 3. ADVANCED UI CSS
# ============================================================

st.markdown(
    """
<style>

html,
body,
[class*="css"] {
    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;
}


/* =========================================================
   APPLICATION
========================================================= */

[data-testid="stAppViewContainer"] {

    background:
        linear-gradient(
            180deg,
            #edf4f2 0%,
            #f8fbfa 20%,
            #ffffff 55%
        );
}


[data-testid="stHeader"] {

    background:
        rgba(
            255,
            255,
            255,
            0.65
        );

    backdrop-filter:
        blur(12px);
}


.block-container {

    max-width:
        1800px;

    padding-top:
        0.20rem;

    padding-left:
        0.75rem;

    padding-right:
        0.75rem;

    padding-bottom:
        3rem;
}


/* =========================================================
   HEADER
========================================================= */

.gp-header {

    background:
        linear-gradient(
            115deg,
            #053c32 0%,
            #05694f 46%,
            #07996f 100%
        );

    padding:
        18px 25px;

    color:
        white;

    border-radius:
        0 0 18px 18px;

    box-shadow:
        0 6px 22px
        rgba(
            0,
            60,
            48,
            0.23
        );

}


.gp-header-row {

    display:
        flex;

    align-items:
        center;

    gap:
        20px;
}


.gp-logo img {

    width:
        88px;

    height:
        88px;

    object-fit:
        contain;

    background:
        white;

    border-radius:
        12px;

    padding:
        6px;
}


.gp-org {

    font-size:
        11px;

    font-weight:
        800;

    letter-spacing:
        .11em;

    opacity:
        .80;
}


.gp-institute {

    font-size:
        16px;

    font-weight:
        700;

    margin-top:
        2px;
}


.gp-title {

    font-size:
        31px;

    font-weight:
        850;

    line-height:
        1.05;

    margin-top:
        7px;
}


.gp-subtitle {

    font-size:
        14px;

    margin-top:
        4px;

    opacity:
        .92;
}


.gp-tag {

    display:
        inline-block;

    margin-top:
        9px;

    padding:
        5px 11px;

    background:
        rgba(
            255,
            255,
            255,
            .15
        );

    border-radius:
        50px;

    font-size:
        11px;

    font-weight:
        650;
}


/* =========================================================
   NAVIGATION
========================================================= */

.gp-nav {

    display:
        flex;

    gap:
        6px;

    align-items:
        center;

    overflow-x:
        auto;

    background:
        #173e38;

    padding:
        8px;

    border-radius:
        11px;

    margin:
        8px 0 13px 0;

    box-shadow:
        0 3px 11px
        rgba(
            0,
            0,
            0,
            .11
        );
}


.gp-nav-item {

    padding:
        9px 14px;

    border-radius:
        7px;

    color:
        #dceae7;

    font-size:
        12px;

    font-weight:
        650;

    white-space:
        nowrap;
}


.gp-nav-active {

    color:
        white;

    background:
        #07966f;
}


.gp-nav-status {

    margin-left:
        auto;

    background:
        #102e2a;

    color:
        #71eab3;

    padding:
        6px 11px;

    border-radius:
        50px;

    font-size:
        11px;

    font-weight:
        700;

    white-space:
        nowrap;
}


/* =========================================================
   PANEL
========================================================= */

.gp-panel {

    background:
        white;

    border:
        1px solid
        #d8e5e1;

    border-radius:
        14px;

    padding:
        14px;

    margin-bottom:
        10px;

    box-shadow:
        0 3px 12px
        rgba(
            25,
            65,
            58,
            .06
        );
}


.gp-panel-title {

    font-size:
        14px;

    font-weight:
        800;

    color:
        #16463b;

    margin-bottom:
        9px;

    padding-bottom:
        7px;

    border-bottom:
        1px solid
        #e4ece9;
}


/* =========================================================
   WORKFLOW CARD
========================================================= */

.gp-workflow {

    background:
        linear-gradient(
            120deg,
            #e9f5fc,
            #f6fbfd
        );

    color:
        #245970;

    border:
        1px solid
        #d3e8f2;

    border-left:
        5px solid
        #1682b5;

    border-radius:
        12px;

    padding:
        12px 15px;

    line-height:
        1.55;

    margin-bottom:
        11px;
}


/* =========================================================
   DECISION CARD
========================================================= */

.decision-card {

    background:
        linear-gradient(
            125deg,
            #063e34,
            #08785d
        );

    color:
        white;

    padding:
        16px 18px;

    border-radius:
        15px;

    margin:
        9px 0 12px 0;

    box-shadow:
        0 5px 17px
        rgba(
            0,
            65,
            52,
            .18
        );
}


.decision-kicker {

    font-size:
        10px;

    letter-spacing:
        .13em;

    font-weight:
        800;

    opacity:
        .72;
}


.decision-value {

    font-size:
        25px;

    font-weight:
        850;

    margin-top:
        3px;
}


.decision-note {

    font-size:
        12px;

    opacity:
        .88;

    margin-top:
        3px;
}


/* =========================================================
   GIS CARD
========================================================= */

.gis-card {

    background:
        white;

    border:
        1px solid
        #d8e6e1;

    border-radius:
        15px;

    padding:
        15px 17px;

    margin:
        10px 0;

    box-shadow:
        0 3px 11px
        rgba(
            20,
            70,
            60,
            .06
        );
}


.gis-card-title {

    color:
        #075f4b;

    font-size:
        15px;

    font-weight:
        800;

    margin-bottom:
        9px;
}


.gis-grid {

    display:
        grid;

    grid-template-columns:
        repeat(
            4,
            minmax(
                0,
                1fr
            )
        );

    gap:
        8px;
}


.gis-grid div {

    background:
        #f3f8f6;

    border-radius:
        9px;

    padding:
        9px 11px;
}


.gis-grid span {

    display:
        block;

    color:
        #71827c;

    font-size:
        10px;
}


.gis-grid b {

    display:
        block;

    color:
        #193e36;

    font-size:
        14px;

    margin-top:
        2px;
}


/* =========================================================
   RESULT CARDS
========================================================= */

.result-card {

    background:
        white;

    border:
        1px solid
        #d8e6e2;

    border-left:
        5px solid
        #087c61;

    border-radius:
        12px;

    padding:
        14px;

    box-shadow:
        0 2px 9px
        rgba(
            15,
            60,
            50,
            .05
        );
}


/* =========================================================
   METRICS
========================================================= */

[data-testid="stMetric"] {

    background:
        white;

    border:
        1px solid
        #d8e3e0;

    border-radius:
        13px;

    padding:
        11px 13px;

    box-shadow:
        0 2px 8px
        rgba(
            20,
            60,
            55,
            .05
        );

    transition:
        all .18s ease;
}


[data-testid="stMetric"]:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 7px 17px
        rgba(
            20,
            60,
            55,
            .10
        );
}


[data-testid="stMetricValue"] {

    color:
        #173f37;

    font-weight:
        800;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button,
.stDownloadButton > button {

    width:
        100%;

    min-height:
        49px;

    border-radius:
        10px;

    font-size:
        14px;

    font-weight:
        750;
}


.stButton > button[kind="primary"] {

    color:
        white;

    border:
        0;

    background:
        linear-gradient(
            90deg,
            #06765a,
            #09a177
        );

    box-shadow:
        0 5px 15px
        rgba(
            5,
            120,
            90,
            .22
        );
}


/* =========================================================
   EXPANDERS
========================================================= */

[data-testid="stExpander"] {

    background:
        white;

    border:
        1px solid
        #dae5e2;

    border-radius:
        12px;

    overflow:
        hidden;
}


/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap:
        5px;

    padding:
        5px;

    background:
        #edf4f2;

    border-radius:
        11px;
}


.stTabs [data-baseweb="tab"] {

    border-radius:
        8px;

    height:
        42px;

    font-weight:
        650;
}


/* =========================================================
   MOBILE
========================================================= */

@media only screen and (max-width:850px) {

    .block-container {

        padding-left:
            .25rem;

        padding-right:
            .25rem;
    }


    .gp-header {

        padding:
            13px 11px;
    }


    .gp-header-row {

        flex-direction:
            column;

        text-align:
            center;

        gap:
            7px;
    }


    .gp-logo img {

        width:
            72px;

        height:
            72px;
    }


    .gp-title {

        font-size:
            25px;
    }


    .gp-institute {

        font-size:
            13px;
    }


    .gp-nav-status {

        display:
            none;
    }


    .gis-grid {

        grid-template-columns:
            repeat(
                2,
                minmax(
                    0,
                    1fr
                )
            );
    }


    h1 {

        font-size:
            1.45rem !important;
    }


    h2 {

        font-size:
            1.2rem !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. GENERAL UTILITIES
# ============================================================

def image_to_base64(
    path
):

    try:

        if not path.exists():

            return None

        return (
            base64
            .b64encode(
                path.read_bytes()
            )
            .decode(
                "utf-8"
            )
        )

    except Exception:

        return None


def safe_text(
    value,
    default="Unknown",
):

    if value is None:

        return default

    if (
        isinstance(
            value,
            float,
        )
        and
        not np.isfinite(
            value
        )
    ):

        return default

    text = str(
        value
    ).strip()

    if (
        not text
        or
        text.lower()
        in {
            "nan",
            "none",
            "null",
        }
    ):

        return default

    return text


def norm_name(
    value
):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            value
        ).lower(),
    )


def first_matching_column(
    columns,
    candidates,
):

    lookup = {
        norm_name(c):
        c

        for c
        in columns
    }

    for candidate in candidates:

        key = norm_name(
            candidate
        )

        if key in lookup:

            return lookup[
                key
            ]

    return None


# ============================================================
# 5. HEADER
# ============================================================

logo_b64 = image_to_base64(
    ICAR_LOGO
)

if logo_b64:

    logo_html = (
        '<div class="gp-logo">'
        f'<img src="data:image/png;base64,{logo_b64}" '
        'alt="ICAR Logo">'
        '</div>'
    )

else:

    logo_html = (
        '<div style="font-size:55px;">🌾</div>'
    )


st.markdown(
    f"""
<div class="gp-header">

<div class="gp-header-row">

{logo_html}

<div>

<div class="gp-org">
INDIAN COUNCIL OF AGRICULTURAL RESEARCH
</div>

<div class="gp-institute">
ICAR Research Complex for NEH Region,
Umiam, Meghalaya
</div>

<div class="gp-title">
GeoPond-DSS
</div>

<div class="gp-subtitle">
Geospatial Pond Siting and Preliminary
Design Decision Support System
</div>

<div class="gp-tag">
Ri Bhoi District • Meghalaya •
Mobile GPS Enabled • Web-GIS Decision Support
</div>

</div>

</div>

</div>


<div class="gp-nav">

<div class="gp-nav-item gp-nav-active">
🗺 Field Map
</div>

<div class="gp-nav-item">
🌾 Land Use
</div>

<div class="gp-nav-item">
💧 Hydrology
</div>

<div class="gp-nav-item">
📍 Site Screening
</div>

<div class="gp-nav-item">
📐 Pond Design
</div>

<div class="gp-nav-item">
🪨 Geology
</div>

<div class="gp-nav-item">
📄 Assessment
</div>

<div class="gp-nav-status">
● GIS DATABASE ONLINE
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 6. REQUIRED GIS FILE CHECK
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


missing_files = [

    str(
        path.relative_to(
            BASE_DIR
        )
    )

    for path
    in REQUIRED_FILES

    if not path.exists()

]


if missing_files:

    st.error(
        "GeoPond-DSS GIS database is incomplete."
    )

    st.code(
        "\n".join(
            missing_files
        )
    )

    st.stop()


# ============================================================
# 7. VECTOR LOADING
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_vector_utm(
    path_string,
    fallback_path_string=None,
):

    candidates = [
        Path(
            path_string
        )
    ]

    if fallback_path_string:

        candidates.append(
            Path(
                fallback_path_string
            )
        )

    path = next(
        (
            p

            for p
            in candidates

            if p.exists()
        ),
        None,
    )

    if path is None:

        return None

    try:

        gdf = gpd.read_file(
            path
        )

        if gdf.empty:

            return gdf

        if gdf.crs is None:

            return None

        try:

            invalid = (
                ~gdf.geometry.is_valid
            )

            if invalid.any():

                gdf = (
                    gdf.copy()
                )

                gdf.loc[
                    invalid,
                    "geometry",
                ] = (
                    gdf.loc[
                        invalid,
                        "geometry",
                    ]
                    .make_valid()
                )

        except Exception:

            pass

        if (
            gdf.crs
            .to_epsg()
            !=
            32646
        ):

            gdf = gdf.to_crs(
                WORKING_CRS
            )

        return gdf

    except Exception:

        return None


boundary_utm = load_vector_utm(
    str(
        BOUNDARY_GPKG
    )
)

streams_utm = load_vector_utm(
    str(
        STREAMS_GPKG
    )
)

lulc_2022_utm = load_vector_utm(
    str(
        LULC_2022_GPKG
    )
)

projected_lulc_2025_utm = (
    load_vector_utm(
        str(
            PROJECTED_LULC_2025_SHP
        )
    )
)

soil_gdf = load_vector_utm(

    str(
        SOIL_GPKG
    ),

    str(
        SOIL_SHP
    ),
)

geomorph_gdf = load_vector_utm(
    str(
        GEOMORPHOLOGY_GPKG
    )
)

lithology_gdf = load_vector_utm(
    str(
        LITHOLOGY_GPKG
    )
)

shear_gdf = load_vector_utm(
    str(
        SHEAR_ZONE_GPKG
    )
)

c_horizon_gdf = load_vector_utm(
    str(
        SOIL_C_HORIZON_GPKG
    )
)


boundary_wgs = None

if (
    boundary_utm
    is not None
    and
    not boundary_utm.empty
):

    boundary_wgs = (
        boundary_utm
        .to_crs(
            "EPSG:4326"
        )
    )


# ============================================================
# GIS STATUS
# ============================================================

THEMATIC_STATUS = {

    "Soil":
        soil_gdf
        is not None
        and
        not soil_gdf.empty,

    "Geomorphology":
        geomorph_gdf
        is not None
        and
        not geomorph_gdf.empty,

    "Lithology":
        lithology_gdf
        is not None
        and
        not lithology_gdf.empty,

    "Shear Zone":
        shear_gdf
        is not None
        and
        not shear_gdf.empty,

    "C-Horizon":
        c_horizon_gdf
        is not None
        and
        not c_horizon_gdf.empty,
}


with st.expander(
    "🧭 GIS Database & CRS Status",
    expanded=False,
):

    st.write(
        "**Analysis CRS:** "
        "EPSG:32646 — WGS 84 / UTM Zone 46N"
    )

    for name, loaded in (
        THEMATIC_STATUS.items()
    ):

        st.write(
            (
                "✅ "
                if loaded
                else
                "❌ "
            )
            +
            name
        )

    lulc_ok = (

        lulc_2022_utm
        is not None

        and

        not lulc_2022_utm.empty

        and

        "LULC_2022"
        in
        lulc_2022_utm.columns

    )

    st.write(
        (
            "✅ "
            if lulc_ok
            else
            "❌ "
        )
        +
        "Mapped 2022 LULC — "
        "attribute: LULC_2022"
    )


# ============================================================
# 8. SESSION STATE
# ============================================================

SESSION_DEFAULTS = {

    "field_geometry":
        None,

    "assessment":
        None,

    "current_lulc":
        None,

}


for key, value in (
    SESSION_DEFAULTS.items()
):

    if key not in (
        st.session_state
    ):

        st.session_state[
            key
        ] = value


def reset_assessment():

    st.session_state[
        "field_geometry"
    ] = None

    st.session_state[
        "assessment"
    ] = None

    st.session_state[
        "current_lulc"
    ] = None


# ============================================================
# 9. GEOMETRY HELPERS
# ============================================================

def geometry_gdf(
    geometry_geojson,
    crs="EPSG:4326",
):

    return gpd.GeoDataFrame(

        {
            "id": [1]
        },

        geometry=[
            shape(
                geometry_geojson
            )
        ],

        crs=crs,
    )


def to_utm_geometry(
    geometry_geojson
):

    return (
        geometry_gdf(
            geometry_geojson
        )
        .to_crs(
            WORKING_CRS
        )
        .geometry
        .iloc[0]
    )


def field_area_ha(
    geometry_geojson
):

    return (
        float(
            to_utm_geometry(
                geometry_geojson
            ).area
        )
        /
        10000.0
    )


def get_last_polygon(
    map_output
):

    if not map_output:

        return None

    drawings = (
        map_output
        .get(
            "all_drawings"
        )
        or
        []
    )

    polygon = None

    for drawing in drawings:

        if not drawing:

            continue

        geom = (
            drawing
            .get(
                "geometry",
                {},
            )
        )

        if (
            geom.get(
                "type"
            )
            in
            [
                "Polygon",
                "MultiPolygon",
            ]
        ):

            polygon = geom

    return polygon


# ============================================================
# 10. RASTER HELPERS
# ============================================================

def raster_mean(
    raster_path,
    field_geojson,
):

    polygon = geometry_gdf(
        field_geojson
    )

    with rasterio.open(
        raster_path
    ) as src:

        geom = (
            polygon
            .to_crs(
                src.crs
            )
            .geometry
            .iloc[0]
        )

        values = np.array(
            [],
            dtype=float,
        )

        try:

            clipped, _ = rio_mask(

                src,

                [
                    mapping(
                        geom
                    )
                ],

                crop=True,

                indexes=1,

                filled=False,

                all_touched=True,

            )

            if np.ma.isMaskedArray(
                clipped
            ):

                values = (
                    clipped
                    .compressed()
                )

            else:

                values = (
                    clipped
                    .reshape(-1)
                )

            values = values[
                np.isfinite(
                    values
                )
            ]

            if src.nodata is not None:

                values = values[
                    values
                    !=
                    src.nodata
                ]

        except Exception:

            pass

        if len(
            values
        ) == 0:

            centroid = (
                geom.centroid
            )

            try:

                sample = next(

                    src.sample(

                        [
                            (
                                centroid.x,
                                centroid.y,
                            )
                        ],

                        indexes=1,

                        masked=True,

                    )
                )

                sample = (
                    np.asarray(
                        sample,
                        dtype=float,
                    )
                    .reshape(-1)
                )

                sample = sample[
                    np.isfinite(
                        sample
                    )
                ]

                values = sample

            except Exception:

                pass

        if len(
            values
        ) == 0:

            return np.nan

        return float(
            np.mean(
                values
            )
        )


def sample_raster(
    raster_path,
    x,
    y,
):

    with rasterio.open(
        raster_path
    ) as src:

        transformer = (
            Transformer
            .from_crs(
                WORKING_CRS,
                src.crs,
                always_xy=True,
            )
        )

        rx, ry = (
            transformer
            .transform(
                x,
                y,
            )
        )

        try:

            value = next(

                src.sample(

                    [
                        (
                            rx,
                            ry,
                        )
                    ],

                    indexes=1,

                    masked=True,

                )
            )[0]

            if np.ma.is_masked(
                value
            ):

                return np.nan

            value = float(
                value
            )

            return (
                value
                if np.isfinite(
                    value
                )
                else np.nan
            )

        except Exception:

            return np.nan


def field_reference_grid(
    field_geojson
):

    polygon = geometry_gdf(
        field_geojson
    )

    with rasterio.open(
        ELEVATION_RASTER
    ) as src:

        geom = (
            polygon
            .to_crs(
                src.crs
            )
            .geometry
            .iloc[0]
        )

        data, transform = rio_mask(

            src,

            [
                mapping(
                    geom
                )
            ],

            crop=True,

            indexes=1,

            filled=False,

            all_touched=True,

        )

        if np.ma.isMaskedArray(
            data
        ):

            arr = (
                data
                .filled(
                    np.nan
                )
                .astype(
                    float
                )
            )

            valid = (
                ~np.ma
                .getmaskarray(
                    data
                )
            )

        else:

            arr = (
                data
                .astype(
                    float
                )
            )

            valid = (
                np.isfinite(
                    arr
                )
            )

        return (
            arr,
            valid,
            transform,
            src.crs,
        )


def raster_to_reference(
    raster_path,
    reference_shape,
    reference_transform,
    reference_crs,
):

    destination = np.full(

        reference_shape,

        np.nan,

        dtype=np.float64,

    )

    with rasterio.open(
        raster_path
    ) as src:

        source = (
            src.read(1)
            .astype(
                np.float64
            )
        )

        if src.nodata is not None:

            source[
                source
                ==
                src.nodata
            ] = np.nan

        reproject(

            source=
                source,

            destination=
                destination,

            src_transform=
                src.transform,

            src_crs=
                src.crs,

            dst_transform=
                reference_transform,

            dst_crs=
                reference_crs,

            src_nodata=
                np.nan,

            dst_nodata=
                np.nan,

            resampling=
                Resampling.bilinear,

        )

    return destination


def normalize_array(
    array,
    valid_mask,
):

    values = (
        array[
            valid_mask
        ]
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return np.zeros_like(
            array,
            dtype=float,
        )

    low = np.percentile(
        values,
        5,
    )

    high = np.percentile(
        values,
        95,
    )

    if (
        not np.isfinite(
            low
        )
        or
        not np.isfinite(
            high
        )
        or
        high <= low
    ):

        return np.zeros_like(
            array,
            dtype=float,
        )

    return np.clip(

        (
            array
            -
            low
        )
        /
        (
            high
            -
            low
        ),

        0,
        1,

    )


# ============================================================
# 11. VECTOR LOOKUP HELPERS
# ============================================================

def point_record(
    gdf,
    x,
    y,
):

    if (
        gdf is None
        or
        gdf.empty
    ):

        return None

    point = Point(
        x,
        y,
    )

    try:

        indices = list(

            gdf.sindex
            .intersection(
                point.bounds
            )

        )

        candidate = (

            gdf.iloc[
                indices
            ]

            if indices

            else

            gdf.iloc[
                0:0
            ]

        )

    except Exception:

        candidate = gdf

    if candidate.empty:

        return None

    try:

        hit = candidate[
            candidate.geometry.covers(
                point
            )
        ]

    except Exception:

        hit = candidate[
            candidate.intersects(
                point
            )
        ]

    if hit.empty:

        return None

    return hit.iloc[
        0
    ]


def nearest_distance_to_layer(
    gdf,
    x,
    y,
):

    if (
        gdf is None
        or
        gdf.empty
    ):

        return None

    point = Point(
        x,
        y,
    )

    distances = (
        gdf.geometry
        .distance(
            point
        )
    )

    if distances.empty:

        return None

    return float(
        distances.min()
    )


def nearest_record(
    gdf,
    x,
    y,
):

    if (
        gdf is None
        or
        gdf.empty
    ):

        return (
            None,
            None,
        )

    point = Point(
        x,
        y,
    )

    distances = (
        gdf.geometry
        .distance(
            point
        )
    )

    if distances.empty:

        return (
            None,
            None,
        )

    index = (
        distances.idxmin()
    )

    return (
        gdf.loc[
            index
        ],
        float(
            distances.loc[
                index
            ]
        ),
    )


def extract_record_fields(
    record,
    mapping_dict,
):

    if record is None:

        return {

            key:
                "Unknown"

            for key
            in mapping_dict

        }

    output = {}

    columns = list(
        record.index
    )

    for output_name, candidates in (
        mapping_dict.items()
    ):

        column = first_matching_column(

            columns,

            candidates,

        )

        output[
            output_name
        ] = (

            safe_text(
                record[
                    column
                ]
            )

            if column

            else

            "Unknown"

        )

    return output


# ============================================================
# 12. LULC
# ============================================================

def canonical_lulc_name(
    value
):

    text = safe_text(
        value
    )

    key = norm_name(
        text
    )

    mapping_dict = {

        "cropland":
            "Crop land",

        "agriculture":
            "Crop land",

        "agriculturalland":
            "Crop land",

        "plantation":
            "Plantation",

        "shiftingcultivation":
            "Shifting Cultivation",

        "jhum":
            "Shifting Cultivation",

        "scrub":
            "Scrub land",

        "scrubland":
            "Scrub land",

        "forestopen":
            "Forest Open",

        "openforest":
            "Forest Open",

        "forestmoderatelydense":
            "Forest Moderately Dense",

        "moderatelydenseforest":
            "Forest Moderately Dense",

        "forestdense":
            "Forest Dense",

        "denseforest":
            "Forest Dense",

        "builtup":
            "Built up",

        "settlement":
            "Built up",

        "waterbody":
            "Waterbody",

        "riverstream":
            "River/Stream",

    }

    return mapping_dict.get(
        key,
        text,
    )


def mapped_lulc_summary(
    field_geojson
):

    empty_result = {

        "composition":
            [],

        "dominant":
            "Unknown",

        "confidence":
            0.0,

        "coverage":
            0.0,

        "classification":
            "Unavailable",

    }

    if (
        lulc_2022_utm
        is None
        or
        lulc_2022_utm.empty
    ):

        return empty_result

    if (
        "LULC_2022"
        not in
        lulc_2022_utm.columns
    ):

        return empty_result

    field_geom = (
        to_utm_geometry(
            field_geojson
        )
    )

    if not field_geom.is_valid:

        field_geom = (
            field_geom
            .buffer(0)
        )

    field_area = (
        field_geom.area
    )

    if field_area <= 0:

        return empty_result

    core_buffer = min(

        5.0,

        max(
            1.0,
            0.02
            *
            math.sqrt(
                field_area
            ),
        ),
    )

    field_core = (
        field_geom
        .buffer(
            -core_buffer
        )
    )

    if (
        field_core.is_empty
        or
        field_core.area
        <
        0.35
        *
        field_area
    ):

        field_core = (
            field_geom
        )

    try:

        ids = list(

            lulc_2022_utm
            .sindex
            .intersection(
                field_geom.bounds
            )

        )

        candidate = (

            lulc_2022_utm
            .iloc[
                ids
            ]
            .copy()

            if ids

            else

            lulc_2022_utm
            .iloc[
                0:0
            ]
            .copy()

        )

    except Exception:

        candidate = (
            lulc_2022_utm
            .copy()
        )

    candidate = candidate[
        candidate.intersects(
            field_geom
        )
    ].copy()

    if candidate.empty:

        return empty_result

    candidate[
        "_class"
    ] = (

        candidate[
            "LULC_2022"
        ]
        .apply(
            canonical_lulc_name
        )

    )

    candidate[
        "_full_area"
    ] = (

        candidate.geometry
        .intersection(
            field_geom
        )
        .area

    )

    candidate[
        "_core_area"
    ] = (

        candidate.geometry
        .intersection(
            field_core
        )
        .area

    )

    grouped = (

        candidate
        .groupby(
            "_class",
            dropna=False,
        )[
            "_full_area"
        ]
        .sum()
        .reset_index()

    )

    grouped = grouped[
        grouped[
            "_full_area"
        ]
        >
        0.5
    ].copy()

    grouped = (
        grouped
        .sort_values(
            "_full_area",
            ascending=False,
        )
    )

    mapped_area = float(
        grouped[
            "_full_area"
        ].sum()
    )

    coverage = min(

        100.0,

        mapped_area
        /
        field_area
        *
        100,

    )

    composition = []

    for _, row in (
        grouped.iterrows()
    ):

        composition.append(
            {

                "LULC":
                    row[
                        "_class"
                    ],

                "Area (ha)":
                    round(
                        row[
                            "_full_area"
                        ]
                        /
                        10000,
                        3,
                    ),

                "Percent of field":
                    round(
                        row[
                            "_full_area"
                        ]
                        /
                        field_area
                        *
                        100,
                        1,
                    ),

            }
        )

    core_grouped = (

        candidate
        .groupby(
            "_class"
        )[
            "_core_area"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "_core_area",
            ascending=False,
        )

    )

    if core_grouped.empty:

        dominant = (
            "Unknown"
        )

        confidence = 0.0

    else:

        dominant = (
            core_grouped
            .iloc[0][
                "_class"
            ]
        )

        core_total = (
            core_grouped[
                "_core_area"
            ]
            .sum()
        )

        confidence = (

            core_grouped
            .iloc[0][
                "_core_area"
            ]
            /
            core_total
            *
            100

            if core_total > 0

            else 0

        )

    if coverage < 70:

        classification = (
            "Incomplete GIS coverage"
        )

    elif confidence >= 70:

        classification = (
            f"2022 GIS reference: {dominant}"
        )

    else:

        classification = (
            f"Mixed GIS reference: {dominant}"
        )

    return {

        "composition":
            composition,

        "dominant":
            dominant,

        "confidence":
            float(
                confidence
            ),

        "coverage":
            float(
                coverage
            ),

        "classification":
            classification,

    }


# ============================================================
# 13. SITE SELECTION
# ============================================================

def find_pond_site(
    field_geojson,
    boundary_clearance=10,
):

    elevation, \
    elevation_valid, \
    transform, \
    raster_crs = (
        field_reference_grid(
            field_geojson
        )
    )

    shape_ref = (
        elevation.shape
    )

    slope = raster_to_reference(

        SLOPE_RASTER,

        shape_ref,

        transform,

        raster_crs,

    )

    twi = raster_to_reference(

        TWI_RASTER,

        shape_ref,

        transform,

        raster_crs,

    )

    drainage = raster_to_reference(

        DRAINAGE_DENSITY_RASTER,

        shape_ref,

        transform,

        raster_crs,

    )

    flow = raster_to_reference(

        FLOW_ACCUMULATION_RASTER,

        shape_ref,

        transform,

        raster_crs,

    )

    valid = (

        elevation_valid

        &

        np.isfinite(
            elevation
        )

        &

        np.isfinite(
            slope
        )

        &

        np.isfinite(
            twi
        )

        &

        np.isfinite(
            drainage
        )

    )

    if (
        np.count_nonzero(
            valid
        )
        ==
        0
    ):

        return None

    lowest_array = np.where(

        valid,

        elevation,

        np.inf,

    )

    lowest_idx = (
        np.unravel_index(

            np.argmin(
                lowest_array
            ),

            elevation.shape,

        )
    )

    lowest_x, \
    lowest_y = xy(

        transform,

        lowest_idx[0],

        lowest_idx[1],

        offset="center",

    )

    elevation_norm = normalize_array(
        elevation,
        valid,
    )

    slope_norm = normalize_array(
        slope,
        valid,
    )

    twi_norm = normalize_array(
        twi,
        valid,
    )

    drainage_norm = normalize_array(
        drainage,
        valid,
    )

    score = (

        0.45
        *
        (
            1
            -
            elevation_norm
        )

        +

        0.25
        *
        twi_norm

        +

        0.15
        *
        (
            1
            -
            slope_norm
        )

        +

        0.10
        *
        drainage_norm

    )

    flow_valid = (

        valid

        &

        np.isfinite(
            flow
        )

    )

    if np.count_nonzero(
        flow_valid
    ):

        flow_norm = normalize_array(

            np.log1p(
                np.maximum(
                    flow,
                    0,
                )
            ),

            flow_valid,

        )

        score += (
            0.05
            *
            flow_norm
        )

    preferred = (

        valid

        &

        (
            slope
            <=
            8
        )

    )

    if np.count_nonzero(
        preferred
    ) == 0:

        preferred = (

            valid

            &

            (
                slope
                <=
                15
            )

        )

    if np.count_nonzero(
        preferred
    ) == 0:

        preferred = valid

    score = np.where(

        preferred,

        score,

        -np.inf,

    )

    field_geom = (

        geometry_gdf(
            field_geojson
        )
        .to_crs(
            raster_crs
        )
        .geometry
        .iloc[0]

    )

    stream_union = None

    if (
        streams_utm
        is not None
        and
        not streams_utm.empty
    ):

        streams_local = (
            streams_utm
            .to_crs(
                raster_crs
            )
        )

        search_area = (
            field_geom
            .buffer(
                1000
            )
        )

        try:

            ids = list(

                streams_local
                .sindex
                .intersection(
                    search_area.bounds
                )

            )

            local_streams = (

                streams_local
                .iloc[
                    ids
                ]

                if ids

                else

                streams_local
                .iloc[
                    0:0
                ]

            )

        except Exception:

            local_streams = (
                streams_local
            )

        local_streams = (
            local_streams[
                local_streams
                .intersects(
                    search_area
                )
            ]
        )

        if not local_streams.empty:

            stream_union = (
                unary_union(
                    local_streams
                    .geometry
                    .tolist()
                )
            )

    order = (
        np.argsort(
            score.ravel()
        )[::-1]
    )

    best = None

    for flat_index in (
        order[:2000]
    ):

        r, c = (
            np.unravel_index(
                flat_index,
                score.shape,
            )
        )

        if not np.isfinite(
            score[
                r,
                c,
            ]
        ):

            continue

        px, py = xy(

            transform,

            r,

            c,

            offset="center",

        )

        point = Point(
            px,
            py,
        )

        boundary_distance = (

            point.distance(
                field_geom.boundary
            )

        )

        if (
            boundary_distance
            <
            boundary_clearance
        ):

            continue

        stream_distance = None

        if stream_union is not None:

            stream_distance = (

                point.distance(
                    stream_union
                )

            )

            if stream_distance < 10:

                continue

        shear_distance = (
            nearest_distance_to_layer(

                shear_gdf,

                px,

                py,

            )
        )

        adjusted_score = float(
            score[
                r,
                c,
            ]
        )

        if shear_distance is not None:

            if shear_distance < 50:

                adjusted_score -= (
                    0.35
                )

            elif shear_distance < 100:

                adjusted_score -= (
                    0.20
                )

            elif shear_distance < 250:

                adjusted_score -= (
                    0.08
                )

        geom_record = (
            point_record(

                geomorph_gdf,

                px,

                py,

            )
        )

        geom_info = (
            extract_record_fields(

                geom_record,

                {
                    "landform":
                        [
                            "legend_sho",
                            "description",
                            "descriptio",
                            "geomorphology",
                            "landform",
                        ]
                },

            )
        )

        landform = (
            geom_info[
                "landform"
            ]
            .lower()
        )

        if (
            "scarp"
            in landform
            or
            "ridge"
            in landform
        ):

            adjusted_score -= (
                0.25
            )

        candidate = {

            "x":
                float(
                    px
                ),

            "y":
                float(
                    py
                ),

            "elevation":
                float(
                    elevation[
                        r,
                        c,
                    ]
                ),

            "slope":
                float(
                    slope[
                        r,
                        c,
                    ]
                ),

            "twi":
                float(
                    twi[
                        r,
                        c,
                    ]
                ),

            "drainage_density":
                float(
                    drainage[
                        r,
                        c,
                    ]
                ),

            "flow_accumulation":
                float(
                    flow[
                        r,
                        c,
                    ]
                )
                if np.isfinite(
                    flow[
                        r,
                        c,
                    ]
                )
                else np.nan,

            "site_score":
                float(
                    score[
                        r,
                        c,
                    ]
                ),

            "adjusted_score":
                adjusted_score,

            "boundary_distance":
                float(
                    boundary_distance
                ),

            "stream_distance":
                (
                    float(
                        stream_distance
                    )
                    if
                    stream_distance
                    is not None
                    else
                    None
                ),

        }

        if (
            best is None
            or
            candidate[
                "adjusted_score"
            ]
            >
            best[
                "adjusted_score"
            ]
        ):

            best = candidate

    if best is None:

        return None

    transformer = (
        Transformer
        .from_crs(
            raster_crs,
            "EPSG:4326",
            always_xy=True,
        )
    )

    lon, lat = (
        transformer
        .transform(
            best[
                "x"
            ],
            best[
                "y"
            ],
        )
    )

    lowest_lon, \
    lowest_lat = (
        transformer
        .transform(
            lowest_x,
            lowest_y,
        )
    )

    best[
        "longitude"
    ] = float(
        lon
    )

    best[
        "latitude"
    ] = float(
        lat
    )

    return {

        "recommended":
            best,

        "lowest":
            {

                "longitude":
                    float(
                        lowest_lon
                    ),

                "latitude":
                    float(
                        lowest_lat
                    ),

                "elevation":
                    float(
                        elevation[
                            lowest_idx
                        ]
                    ),

            },

    }


# ============================================================
# 14. POND DESIGN
# ============================================================

def calculate_pond_geometry(

    bottom_breadth,

    water_depth,

    freeboard,

    side_slope,

    length_breadth_ratio,

):

    bottom_length = (

        bottom_breadth
        *
        length_breadth_ratio

    )

    water_top_breadth = (

        bottom_breadth

        +

        2
        *
        side_slope
        *
        water_depth

    )

    water_top_length = (

        bottom_length

        +

        2
        *
        side_slope
        *
        water_depth

    )

    bottom_area = (

        bottom_length
        *
        bottom_breadth

    )

    top_water_area = (

        water_top_length
        *
        water_top_breadth

    )

    volume = (

        water_depth
        /
        3

        *

        (
            bottom_area
            +
            top_water_area
            +
            math.sqrt(
                bottom_area
                *
                top_water_area
            )
        )

    )

    excavation_depth = (

        water_depth
        +
        freeboard

    )

    top_breadth = (

        bottom_breadth

        +

        2
        *
        side_slope
        *
        excavation_depth

    )

    top_length = (

        bottom_length

        +

        2
        *
        side_slope
        *
        excavation_depth

    )

    footprint = (

        top_length
        *
        top_breadth

    )

    side_slant = math.sqrt(

        water_depth ** 2

        +

        (
            side_slope
            *
            water_depth
        ) ** 2

    )

    long_side_area = (

        (
            bottom_length
            +
            water_top_length
        )
        /
        2

        *
        side_slant

    )

    short_side_area = (

        (
            bottom_breadth
            +
            water_top_breadth
        )
        /
        2

        *
        side_slant

    )

    lining_area = (

        bottom_area

        +

        2
        *
        long_side_area

        +

        2
        *
        short_side_area

    )

    return {

        "bottom_length":
            float(
                bottom_length
            ),

        "bottom_breadth":
            float(
                bottom_breadth
            ),

        "top_length":
            float(
                top_length
            ),

        "top_breadth":
            float(
                top_breadth
            ),

        "water_depth":
            float(
                water_depth
            ),

        "freeboard":
            float(
                freeboard
            ),

        "excavation_depth":
            float(
                excavation_depth
            ),

        "volume":
            float(
                volume
            ),

        "footprint":
            float(
                footprint
            ),

        "lining_area":
            float(
                lining_area
            ),

        "liner_purchase_area":
            float(
                lining_area
                *
                1.10
            ),

    }


def design_farm_pond(

    available_water_m3,

    field_area_ha_value,

    storage_fraction,

    maximum_land_fraction,

    water_depth,

    freeboard,

    side_slope,

    length_breadth_ratio,

):

    field_area_m2 = (

        field_area_ha_value
        *
        10000

    )

    requested_storage = (

        available_water_m3
        *
        storage_fraction

    )

    maximum_footprint = (

        field_area_m2
        *
        maximum_land_fraction

    )

    low = 0.5

    high = 120

    for _ in range(
        80
    ):

        middle = (
            low
            +
            high
        ) / 2

        geom = calculate_pond_geometry(

            middle,

            water_depth,

            freeboard,

            side_slope,

            length_breadth_ratio,

        )

        if (
            geom[
                "footprint"
            ]
            >
            maximum_footprint
        ):

            high = middle

        else:

            low = middle

    maximum_geom = (
        calculate_pond_geometry(

            low,

            water_depth,

            freeboard,

            side_slope,

            length_breadth_ratio,

        )
    )

    target_storage = min(

        requested_storage,

        maximum_geom[
            "volume"
        ],

    )

    lower = 0.5

    upper = max(
        low,
        0.5,
    )

    for _ in range(
        80
    ):

        middle = (
            lower
            +
            upper
        ) / 2

        geom = (
            calculate_pond_geometry(

                middle,

                water_depth,

                freeboard,

                side_slope,

                length_breadth_ratio,

            )
        )

        if (
            geom[
                "volume"
            ]
            >
            target_storage
        ):

            upper = middle

        else:

            lower = middle

    result = (
        calculate_pond_geometry(

            lower,

            water_depth,

            freeboard,

            side_slope,

            length_breadth_ratio,

        )
    )

    result[
        "requested_storage"
    ] = requested_storage

    result[
        "target_storage"
    ] = target_storage

    result[
        "occupied_percent"
    ] = (

        result[
            "footprint"
        ]
        /
        field_area_m2
        *
        100

    )

    return result


# ============================================================
# 15. GEOLOGICAL / SEEPAGE SCREENING
# ============================================================

def classify_soil_texture(
    texture
):

    text = safe_text(
        texture
    ).lower()

    if text == "unknown":

        return (
            "Unknown",
            1,
        )

    if (
        "clay"
        in text
        and
        "sandy"
        not in text
    ):

        return (
            "Low to moderate seepage tendency",
            0,
        )

    if any(
        term
        in text

        for term
        in [
            "fine",
            "silt loam",
            "silty clay",
        ]
    ):

        return (
            "Moderate seepage tendency",
            1,
        )

    if (
        "loam"
        in text
        and
        "sandy"
        not in text
    ):

        return (
            "Moderate to high seepage tendency",
            2,
        )

    if "sandy" in text:

        return (
            "High seepage tendency",
            3,
        )

    return (
        "Field verification required",
        1,
    )


def terrain_constraint(
    landform,
    slope
):

    text = safe_text(
        landform
    ).lower()

    if (
        "scarp"
        in text
        or
        "ridge"
        in text
        or
        slope >= 24
    ):

        return (
            "SEVERE",
            3,
        )

    if (
        "highly dissected"
        in text
        or
        "structural hills"
        in text
        or
        slope >= 14
    ):

        return (
            "HIGH",
            2,
        )

    if slope >= 8:

        return (
            "MODERATE",
            1,
        )

    return (
        "LOW",
        0,
    )


def shear_zone_class(
    distance
):

    if distance is None:

        return (
            "Unknown",
            0,
        )

    if distance < 50:

        return (
            "VERY HIGH CONCERN",
            3,
        )

    if distance < 100:

        return (
            "HIGH CONCERN",
            2,
        )

    if distance < 250:

        return (
            "MODERATE CONCERN",
            1,
        )

    return (
        "LOWER MAPPED CONCERN",
        0,
    )


def build_screening(

    soil_texture,

    landform,

    lithology,

    shear_distance,

    lineament_density,

    slope,

):

    soil_text, \
    soil_score = (
        classify_soil_texture(
            soil_texture
        )
    )

    terrain_text, \
    terrain_score = (
        terrain_constraint(
            landform,
            slope,
        )
    )

    shear_text, \
    shear_score = (
        shear_zone_class(
            shear_distance
        )
    )

    lithology_score = 1

    lith_text = (
        safe_text(
            lithology
        )
        .lower()
    )

    if any(
        term
        in lith_text

        for term
        in [
            "sandstone",
            "limestone",
            "gravel",
            "alluv",
        ]
    ):

        lithology_score = 2

    lineament_score = 0

    if np.isfinite(
        lineament_density
    ):

        if (
            lineament_density
            >=
            0.55
        ):

            lineament_score = 2

        elif (
            lineament_density
            >=
            0.19
        ):

            lineament_score = 1

    seepage_score = (

        soil_score

        +

        shear_score

        +

        lithology_score

        +

        lineament_score

    )

    if seepage_score >= 6:

        seepage = (
            "HIGH"
        )

    elif seepage_score >= 3:

        seepage = (
            "MODERATE"
        )

    else:

        seepage = (
            "LOW"
        )

    if seepage == "HIGH":

        lining = (
            "High seepage-risk screening. "
            "Field permeability testing and "
            "engineered lining are likely required."
        )

    elif seepage == "MODERATE":

        lining = (
            "Moderate seepage-risk screening. "
            "Assess field permeability and available "
            "clay material. Compacted clay or "
            "geomembrane lining may be required."
        )

    else:

        lining = (
            "Lower GIS-screened seepage concern. "
            "Field permeability verification is "
            "still required before deciding that "
            "lining is unnecessary."
        )

    return {

        "soil_screen":
            soil_text,

        "terrain_constraint":
            terrain_text,

        "shear_concern":
            shear_text,

        "seepage_risk":
            seepage,

        "lining_recommendation":
            lining,

    }


# ============================================================
# 16. FARMER INFORMATION
# ============================================================

with st.expander(
    "👨‍🌾 Farmer / Field Information",
    expanded=False,
):

    f1, f2 = (
        st.columns(2)
    )

    farmer_name = (
        f1.text_input(
            "Farmer Name"
        )
    )

    village = (
        f2.text_input(
            "Village / Locality"
        )
    )


# ============================================================
# 17. FIELD MAP
# ============================================================

st.markdown(
    "## 1️⃣ Interactive Farmer Field Mapping"
)


st.markdown(
    """
<div class="gp-workflow">

<b>📍 Field Mapping Workflow</b><br><br>

<b>1.</b> Locate the farmer field using GPS or
satellite imagery.<br>

<b>2.</b> Zoom to parcel level.<br>

<b>3.</b> Select the polygon drawing tool.<br>

<b>4.</b> Trace the actual field boundary.<br>

<b>5.</b> Edit the polygon where necessary.<br>

<b>6.</b> GeoPond-DSS automatically calculates
area and retrieves GIS information.

</div>
""",
    unsafe_allow_html=True,
)


field_map = folium.Map(

    location=
        MAP_CENTER,

    zoom_start=
        MAP_ZOOM,

    tiles=
        None,

    control_scale=
        True,

    prefer_canvas=
        True,

)


# Satellite
folium.TileLayer(

    tiles=
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}",

    attr=
        "Esri World Imagery",

    name=
        "Satellite Imagery",

    overlay=
        False,

    control=
        True,

    show=
        True,

    max_zoom=
        20,

).add_to(
    field_map
)


# OSM
folium.TileLayer(

    "OpenStreetMap",

    name=
        "Road / Village Map",

    show=
        False,

).add_to(
    field_map
)


# District boundary
if boundary_wgs is not None:

    folium.GeoJson(

        boundary_wgs
        .__geo_interface__,

        name=
            "Ri Bhoi District",

        style_function=
            lambda feature: {

                "color":
                    "#FFD600",

                "weight":
                    2,

                "fillOpacity":
                    0,

            },

    ).add_to(
        field_map
    )


# Current field
if (
    st.session_state[
        "field_geometry"
    ]
):

    folium.GeoJson(

        st.session_state[
            "field_geometry"
        ],

        name=
            "Farmer Field",

        style_function=
            lambda feature: {

                "color":
                    "#00E676",

                "weight":
                    5,

                "fillColor":
                    "#69F0AE",

                "fillOpacity":
                    0.25,

            },

    ).add_to(
        field_map
    )


# GPS
LocateControl(

    position=
        "topleft",

    flyTo=
        True,

    strings={
        "title":
            "Use my GPS location"
    },

    locateOptions={

        "enableHighAccuracy":
            True,

        "maximumAge":
            0,

        "timeout":
            15000,

    },

).add_to(
    field_map
)


# Draw tool
Draw(

    export=False,

    position=
        "topleft",

    draw_options={

        "polyline":
            False,

        "rectangle":
            False,

        "circle":
            False,

        "circlemarker":
            False,

        "marker":
            False,

        "polygon": {

            "allowIntersection":
                False,

            "showArea":
                True,

            "metric":
                True,

            "shapeOptions": {

                "color":
                    "#00E676",

                "weight":
                    5,

                "fillColor":
                    "#69F0AE",

                "fillOpacity":
                    0.25,

            },

        },

    },

    edit_options={

        "edit":
            True,

        "remove":
            True,

    },

).add_to(
    field_map
)


# Measure
MeasureControl(

    position=
        "topleft",

    primary_length_unit=
        "meters",

    secondary_length_unit=
        "kilometers",

    primary_area_unit=
        "sqmeters",

).add_to(
    field_map
)


# Mini map
MiniMap(

    toggle_display=
        True,

    position=
        "bottomright",

).add_to(
    field_map
)


# Coordinates
MousePosition(

    position=
        "bottomleft",

    prefix=
        "Coordinates:",

    separator=
        " | ",

    num_digits=
        5,

).add_to(
    field_map
)


Fullscreen(
    position="topleft"
).add_to(
    field_map
)


folium.LayerControl(

    position=
        "topright",

    collapsed=
        True,

).add_to(
    field_map
)


st.markdown(
    """
<div class="gp-panel">

<div class="gp-panel-title">
🗺 Interactive Web-GIS Workspace
</div>

GPS • Satellite Imagery • Field Drawing •
Measurement • Coordinates • Layer Control

</div>
""",
    unsafe_allow_html=True,
)


map_output = st_folium(

    field_map,

    height=
        650,

    use_container_width=
        True,

    key=
        "geopond_field_map",

    returned_objects=[
        "all_drawings",
        "last_active_drawing",
    ],

)


new_field = (
    get_last_polygon(
        map_output
    )
)


if (
    new_field is not None
    and
    new_field
    !=
    st.session_state[
        "field_geometry"
    ]
):

    st.session_state[
        "field_geometry"
    ] = new_field

    st.session_state[
        "assessment"
    ] = None

    st.rerun()


field = (
    st.session_state[
        "field_geometry"
    ]
)


# ============================================================
# 18. FIELD LAND USE
# ============================================================

LAND_USE_OPTIONS = [

    "Crop land",

    "Plantation",

    "Shifting Cultivation",

    "Scrub land",

    "Forest Open",

    "Forest Moderately Dense",

    "Forest Dense",

    "Built up",

    "Waterbody",

    "River/Stream",

]


if field:

    field_area = (
        field_area_ha(
            field
        )
    )

    lulc = (
        mapped_lulc_summary(
            field
        )
    )

    mapped_lulc = (
        lulc[
            "dominant"
        ]
    )

    st.success(
        f"✅ Farmer field captured: "
        f"{field_area:.3f} ha"
    )


    st.markdown(
        "## 🌾 Field Land-Use Assessment"
    )


    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "Field Area",
        f"{field_area:.3f} ha",
    )

    c2.metric(
        "Mapped 2022 LULC",
        mapped_lulc,
    )

    c3.metric(
        "GIS Coverage",
        f"{lulc['coverage']:.1f}%",
    )


    if (
        st.session_state[
            "current_lulc"
        ]
        in
        LAND_USE_OPTIONS
    ):

        default_index = (
            LAND_USE_OPTIONS
            .index(
                st.session_state[
                    "current_lulc"
                ]
            )
        )

    elif (
        mapped_lulc
        in
        LAND_USE_OPTIONS
    ):

        default_index = (
            LAND_USE_OPTIONS
            .index(
                mapped_lulc
            )
        )

    else:

        default_index = 0


    current_lulc = (
        st.selectbox(

            "Current Observed Field Land Use",

            LAND_USE_OPTIONS,

            index=
                default_index,

            help=(
                "Current field observation is authoritative "
                "for farm-level planning."
            ),

        )
    )


    st.session_state[
        "current_lulc"
    ] = current_lulc


    st.markdown(
        f"""
<div class="decision-card">

<div class="decision-kicker">
ACTIVE PLANNING LAND USE
</div>

<div class="decision-value">
{current_lulc}
</div>

<div class="decision-note">
Used for GeoPond-DSS pond planning
and farmer assessment.
</div>

</div>
""",
        unsafe_allow_html=True,
    )


    agreement = (
        current_lulc
        ==
        mapped_lulc
    )


    status = (
        "HIGH CONFIDENCE"
        if agreement
        else
        "FIELD VERIFIED / GIS CONFLICT"
    )


    st.markdown(
        f"""
<div class="gis-card">

<div class="gis-card-title">
🗺 GIS Land-Use Evidence
</div>

<div class="gis-grid">

<div>
<span>Mapped class</span>
<b>{mapped_lulc}</b>
</div>

<div>
<span>Class share</span>
<b>{lulc['confidence']:.1f}%</b>
</div>

<div>
<span>Coverage</span>
<b>{lulc['coverage']:.1f}%</b>
</div>

<div>
<span>Status</span>
<b>{status}</b>
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )


    with st.expander(
        "View LULC composition inside farmer field"
    ):

        if lulc[
            "composition"
        ]:

            st.dataframe(

                pd.DataFrame(
                    lulc[
                        "composition"
                    ]
                ),

                use_container_width=
                    True,

                hide_index=
                    True,

            )


# ============================================================
# 19. PLANNING PARAMETERS
# ============================================================

if field:

    st.markdown(
        "## ⚙️ Pond Planning Parameters"
    )

    with st.expander(
        "Adjust preliminary planning assumptions",
        expanded=False,
    ):

        storage_fraction_percent = (
            st.slider(

                "Storage as % of 75% dependable annual runoff",

                10,

                60,

                30,

                5,

            )
        )

        maximum_land_percent = (
            st.slider(

                "Maximum field area occupied by pond (%)",

                5,

                20,

                10,

                1,

            )
        )

        water_depth = (
            st.slider(

                "Planning water depth (m)",

                1.5,

                3.5,

                2.5,

                0.25,

            )
        )

        freeboard = (
            st.slider(

                "Freeboard (m)",

                0.30,

                0.75,

                0.50,

                0.05,

            )
        )

        side_slope = (
            st.slider(

                "Side slope H:1V",

                1.0,

                2.5,

                1.5,

                0.25,

            )
        )

        length_breadth_ratio = (
            st.slider(

                "Bottom length : breadth ratio",

                1.0,

                2.0,

                1.5,

                0.1,

            )
        )


# ============================================================
# 20. RUN GEOPOND-DSS
# ============================================================

if field:

    st.markdown(
        "## 💧 Run GeoPond-DSS"
    )


    if st.button(

        "💧 FIND BEST GIS-SCREENED POND SITE & PRELIMINARY DESIGN",

        type=
            "primary",

        use_container_width=
            True,

    ):

        with st.spinner(
            "Running hydrology, terrain, pond-site and geological screening..."
        ):

            field_area = (
                field_area_ha(
                    field
                )
            )

            rainfall = raster_mean(
                RAINFALL_RASTER,
                field,
            )

            mean_runoff = raster_mean(
                MEAN_RUNOFF_RASTER,
                field,
            )

            dependable_runoff = raster_mean(
                DEPENDABLE_RUNOFF_RASTER,
                field,
            )

            cn = raster_mean(
                CN_RASTER,
                field,
            )

            runoff_coefficient = raster_mean(
                RUNOFF_COEFFICIENT_RASTER,
                field,
            )

            mean_volume = (

                mean_runoff
                *
                field_area
                *
                10

            )

            dependable_volume = (

                dependable_runoff
                *
                field_area
                *
                10

            )

            pond = design_farm_pond(

                dependable_volume,

                field_area,

                storage_fraction_percent
                /
                100,

                maximum_land_percent
                /
                100,

                water_depth,

                freeboard,

                side_slope,

                length_breadth_ratio,

            )

            half_diagonal = (

                0.5
                *
                math.sqrt(

                    pond[
                        "top_length"
                    ] ** 2

                    +

                    pond[
                        "top_breadth"
                    ] ** 2

                )

            )

            site_result = (
                find_pond_site(

                    field,

                    boundary_clearance=
                        max(
                            10,
                            half_diagonal
                            +
                            2,
                        ),

                )
            )

            if site_result is None:

                st.error(
                    "No suitable GIS-screened candidate was identified."
                )

                st.stop()

            site = (
                site_result[
                    "recommended"
                ]
            )

            lowest = (
                site_result[
                    "lowest"
                ]
            )


            # Soil
            soil_record = (
                point_record(

                    soil_gdf,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )

            soil_info = (
                extract_record_fields(

                    soil_record,

                    {
                        "texture":
                            [
                                "Texture",
                                "soil_texture",
                            ]
                    },

                )
            )


            # Geomorphology
            geom_record = (
                point_record(

                    geomorph_gdf,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )

            geom_info = (
                extract_record_fields(

                    geom_record,

                    {
                        "landform":
                            [
                                "legend_sho",
                                "description",
                                "descriptio",
                                "geomorphology",
                            ]
                    },

                )
            )


            # Lithology
            lith_record = (
                point_record(

                    lithology_gdf,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )

            lith_info = (
                extract_record_fields(

                    lith_record,

                    {

                        "group":
                            [
                                "group_name",
                                "group",
                            ],

                        "formation":
                            [
                                "formation",
                            ],

                        "lithology":
                            [
                                "lithologic",
                                "lithology",
                            ],

                        "age":
                            [
                                "age",
                            ],

                    },

                )
            )


            shear_distance = (
                nearest_distance_to_layer(

                    shear_gdf,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            lineament_density = (
                sample_raster(

                    LINEAMENT_DENSITY_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            screening = (
                build_screening(

                    soil_info[
                        "texture"
                    ],

                    geom_info[
                        "landform"
                    ],

                    lith_info[
                        "lithology"
                    ],

                    shear_distance,

                    lineament_density,

                    site[
                        "slope"
                    ],

                )
            )


            st.session_state[
                "assessment"
            ] = {

                "farmer":
                    farmer_name,

                "village":
                    village,

                "field_area":
                    field_area,

                "current_lulc":
                    st.session_state[
                        "current_lulc"
                    ],

                "mapped_lulc":
                    mapped_lulc,

                "rainfall":
                    rainfall,

                "mean_runoff":
                    mean_runoff,

                "dependable_runoff":
                    dependable_runoff,

                "mean_volume":
                    mean_volume,

                "dependable_volume":
                    dependable_volume,

                "cn":
                    cn,

                "runoff_coefficient":
                    runoff_coefficient,

                "site":
                    site,

                "lowest":
                    lowest,

                "pond":
                    pond,

                "soil_info":
                    soil_info,

                "geom_info":
                    geom_info,

                "lith_info":
                    lith_info,

                "shear_distance":
                    shear_distance,

                "lineament_density":
                    lineament_density,

                "screening":
                    screening,

            }


        st.rerun()


# ============================================================
# 21. RESULT DASHBOARD
# ============================================================

assessment = (
    st.session_state[
        "assessment"
    ]
)


if assessment:

    site = (
        assessment[
            "site"
        ]
    )

    lowest = (
        assessment[
            "lowest"
        ]
    )

    pond = (
        assessment[
            "pond"
        ]
    )

    screening = (
        assessment[
            "screening"
        ]
    )


    st.markdown(
        "## ✅ GeoPond-DSS Assessment Dashboard"
    )


    terrain_status = (

        "Favourable"

        if screening[
            "terrain_constraint"
        ]
        ==
        "LOW"

        else

        "Review"

    )


    st.markdown(
        f"""
<div class="gis-card">

<div class="gis-card-title">
📊 Decision Summary
</div>

<div class="gis-grid">

<div>
<span>Planning land use</span>
<b>{assessment['current_lulc']}</b>
</div>

<div>
<span>Terrain</span>
<b>{terrain_status}</b>
</div>

<div>
<span>Seepage risk</span>
<b>{screening['seepage_risk']}</b>
</div>

<div>
<span>Site score</span>
<b>{site['adjusted_score']:.3f}</b>
</div>

</div>

</div>
""",
        unsafe_allow_html=True,
    )


    k1, k2, k3, k4 = (
        st.columns(4)
    )

    k1.metric(
        "Field Area",
        f"{assessment['field_area']:.3f} ha",
    )

    k2.metric(
        "Dependable Water",
        f"{assessment['dependable_volume']:,.0f} m³/yr",
    )

    k3.metric(
        "Storage Target",
        f"{pond['target_storage']:,.0f} m³",
    )

    k4.metric(
        "Pond Footprint",
        f"{pond['footprint']:,.0f} m²",
    )


    (
        tab_overview,
        tab_map,
        tab_water,
        tab_design,
        tab_geology,
        tab_technical,
    ) = st.tabs(
        [

            "📊 Overview",

            "🗺 Candidate Map",

            "💧 Hydrology",

            "📐 Pond Design",

            "🪨 Geology & Seepage",

            "🔬 Technical",

        ]
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab_overview:

        st.success(
            "The displayed location is the highest-ranked "
            "GIS-screened candidate inside the farmer field. "
            "It is not a surveyed construction peg."
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Candidate Latitude",
            f"{site['latitude']:.6f}",
        )

        c2.metric(
            "Candidate Longitude",
            f"{site['longitude']:.6f}",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Elevation",
            f"{site['elevation']:.1f} m",
        )

        c2.metric(
            "Slope",
            f"{site['slope']:.2f}°",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "TWI",
            f"{site['twi']:.2f}",
        )

        c2.metric(
            "Boundary Clearance",
            f"{site['boundary_distance']:.1f} m",
        )


    # ========================================================
    # MAP
    # ========================================================

    with tab_map:

        result_map = folium.Map(

            location=[
                site[
                    "latitude"
                ],
                site[
                    "longitude"
                ],
            ],

            zoom_start=
                18,

            tiles=
                None,

            control_scale=
                True,

        )


        folium.TileLayer(

            tiles=
                "https://server.arcgisonline.com/"
                "ArcGIS/rest/services/"
                "World_Imagery/MapServer/"
                "tile/{z}/{y}/{x}",

            attr=
                "Esri World Imagery",

            name=
                "Satellite",

            show=
                True,

            max_zoom=
                20,

        ).add_to(
            result_map
        )


        folium.TileLayer(

            "OpenStreetMap",

            name=
                "Road Map",

            show=
                False,

        ).add_to(
            result_map
        )


        folium.GeoJson(

            field,

            name=
                "Farmer Field",

            style_function=
                lambda feature: {

                    "color":
                        "#00E676",

                    "weight":
                        5,

                    "fillColor":
                        "#69F0AE",

                    "fillOpacity":
                        0.18,

                },

        ).add_to(
            result_map
        )


        pond_box_utm = box(

            site[
                "x"
            ]
            -
            pond[
                "top_length"
            ]
            /
            2,

            site[
                "y"
            ]
            -
            pond[
                "top_breadth"
            ]
            /
            2,

            site[
                "x"
            ]
            +
            pond[
                "top_length"
            ]
            /
            2,

            site[
                "y"
            ]
            +
            pond[
                "top_breadth"
            ]
            /
            2,

        )


        pond_box_wgs = (

            gpd.GeoSeries(

                [
                    pond_box_utm
                ],

                crs=
                    WORKING_CRS,

            )

            .to_crs(
                "EPSG:4326"
            )

            .iloc[0]

        )


        folium.GeoJson(

            mapping(
                pond_box_wgs
            ),

            name=
                "Indicative Pond Footprint",

            style_function=
                lambda feature: {

                    "color":
                        "#FF1744",

                    "weight":
                        4,

                    "fillColor":
                        "#FF5252",

                    "fillOpacity":
                        0.30,

                },

        ).add_to(
            result_map
        )


        folium.Marker(

            [
                site[
                    "latitude"
                ],
                site[
                    "longitude"
                ],
            ],

            tooltip=
                "Best GIS-screened candidate",

            icon=
                folium.Icon(
                    color="red",
                    icon="tint",
                ),

        ).add_to(
            result_map
        )


        folium.Marker(

            [
                lowest[
                    "latitude"
                ],
                lowest[
                    "longitude"
                ],
            ],

            tooltip=
                "Lowest DEM cell",

            icon=
                folium.Icon(
                    color="orange"
                ),

        ).add_to(
            result_map
        )


        MeasureControl(
            position="topleft"
        ).add_to(
            result_map
        )


        MousePosition(
            prefix="Coordinates:"
        ).add_to(
            result_map
        )


        Fullscreen(
            position="topleft"
        ).add_to(
            result_map
        )


        folium.LayerControl(
            collapsed=False
        ).add_to(
            result_map
        )


        st_folium(

            result_map,

            height=
                650,

            use_container_width=
                True,

            key=
                "result_map",

        )


        st.caption(
            "The pond rectangle is indicative only. "
            "Final orientation should follow contour, "
            "field survey and inlet/outlet requirements."
        )


    # ========================================================
    # HYDROLOGY
    # ========================================================

    with tab_water:

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Annual Rainfall",
            f"{assessment['rainfall']:.0f} mm/year",
        )

        c2.metric(
            "Mean Annual Runoff",
            f"{assessment['mean_runoff']:.0f} mm",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "75% Dependable Runoff",
            f"{assessment['dependable_runoff']:.0f} mm",
        )

        c2.metric(
            "75% Dependable Water",
            f"{assessment['dependable_volume']:,.0f} m³/year",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Curve Number II",
            f"{assessment['cn']:.1f}",
        )

        c2.metric(
            "Runoff Coefficient",
            f"{assessment['runoff_coefficient']:.3f}",
        )

        st.info(
            "Rainfall retains the information content "
            "of the source rainfall dataset. Spatial "
            "resampling does not create new 30-m rainfall "
            "observations."
        )


    # ========================================================
    # DESIGN
    # ========================================================

    with tab_design:

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Storage Target",
            f"{pond['target_storage']:,.0f} m³",
        )

        c2.metric(
            "Water Depth",
            f"{pond['water_depth']:.2f} m",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Bottom Length",
            f"{pond['bottom_length']:.1f} m",
        )

        c2.metric(
            "Bottom Breadth",
            f"{pond['bottom_breadth']:.1f} m",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Top Length",
            f"{pond['top_length']:.1f} m",
        )

        c2.metric(
            "Top Breadth",
            f"{pond['top_breadth']:.1f} m",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Excavation Depth",
            f"{pond['excavation_depth']:.2f} m",
        )

        c2.metric(
            "Freeboard",
            f"{pond['freeboard']:.2f} m",
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Pond Footprint",
            f"{pond['footprint']:,.0f} m²",
        )

        c2.metric(
            "Field Occupied",
            f"{pond['occupied_percent']:.1f}%",
        )


    # ========================================================
    # GEOLOGY
    # ========================================================

    with tab_geology:

        soil_info = (
            assessment[
                "soil_info"
            ]
        )

        geom_info = (
            assessment[
                "geom_info"
            ]
        )

        lith_info = (
            assessment[
                "lith_info"
            ]
        )


        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Soil Texture",
            soil_info[
                "texture"
            ],
        )

        c2.metric(
            "Soil Screening",
            screening[
                "soil_screen"
            ],
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Geomorphology",
            geom_info[
                "landform"
            ],
        )

        c2.metric(
            "Terrain Constraint",
            screening[
                "terrain_constraint"
            ],
        )

        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Lithological Group",
            lith_info[
                "group"
            ],
        )

        c2.metric(
            "Formation",
            lith_info[
                "formation"
            ],
        )

        st.write(
            "**Lithology:**",
            lith_info[
                "lithology"
            ],
        )

        st.write(
            "**Geological age:**",
            lith_info[
                "age"
            ],
        )


        c1, c2 = (
            st.columns(2)
        )

        c1.metric(

            "Nearest Shear Zone",

            (
                f"{assessment['shear_distance']:.0f} m"

                if
                assessment[
                    "shear_distance"
                ]
                is not None

                else

                "No data"
            ),

        )

        c2.metric(
            "Shear-Zone Screening",
            screening[
                "shear_concern"
            ],
        )


        c1, c2 = (
            st.columns(2)
        )

        c1.metric(

            "Lineament Density",

            (
                f"{assessment['lineament_density']:.3f}"

                if np.isfinite(
                    assessment[
                        "lineament_density"
                    ]
                )

                else

                "No data"
            ),

        )

        c2.metric(
            "Seepage Screening",
            screening[
                "seepage_risk"
            ],
        )


        st.info(
            screening[
                "lining_recommendation"
            ]
        )


        c1, c2 = (
            st.columns(2)
        )

        c1.metric(
            "Estimated Lining Area",
            f"{pond['lining_area']:,.0f} m²",
        )

        c2.metric(
            "Procurement Area +10%",
            f"{pond['liner_purchase_area']:,.0f} m²",
        )


    # ========================================================
    # TECHNICAL
    # ========================================================

    with tab_technical:

        st.write(
            "**Farmer:**",
            assessment[
                "farmer"
            ]
            or
            "Not entered",
        )

        st.write(
            "**Village:**",
            assessment[
                "village"
            ]
            or
            "Not entered",
        )

        st.write(
            "**Observed land use:**",
            assessment[
                "current_lulc"
            ],
        )

        st.write(
            "**Mapped 2022 LULC:**",
            assessment[
                "mapped_lulc"
            ],
        )

        st.write(
            "**Drainage density:**",
            round(
                site[
                    "drainage_density"
                ],
                3,
            ),
        )

        st.write(
            "**Flow accumulation:**",
            round(
                site[
                    "flow_accumulation"
                ],
                2,
            )
            if np.isfinite(
                site[
                    "flow_accumulation"
                ]
            )
            else
            "No data",
        )

        st.write(
            "**Raw site score:**",
            round(
                site[
                    "site_score"
                ],
                3,
            ),
        )

        st.write(
            "**Adjusted site score:**",
            round(
                site[
                    "adjusted_score"
                ],
                3,
            ),
        )


# ============================================================
# 22. REPORT DOWNLOAD
# ============================================================

    output = {

        "DSS":
            "GeoPond-DSS",

        "Farmer":
            assessment[
                "farmer"
            ],

        "Village":
            assessment[
                "village"
            ],

        "Field_Area_ha":
            assessment[
                "field_area"
            ],

        "Current_Land_Use":
            assessment[
                "current_lulc"
            ],

        "Mapped_2022_LULC":
            assessment[
                "mapped_lulc"
            ],

        "Rainfall_mm_year":
            assessment[
                "rainfall"
            ],

        "Mean_Runoff_mm":
            assessment[
                "mean_runoff"
            ],

        "Dependable_Runoff_mm":
            assessment[
                "dependable_runoff"
            ],

        "Dependable_Water_m3_year":
            assessment[
                "dependable_volume"
            ],

        "CN_II":
            assessment[
                "cn"
            ],

        "Runoff_Coefficient":
            assessment[
                "runoff_coefficient"
            ],

        "Pond_Latitude":
            site[
                "latitude"
            ],

        "Pond_Longitude":
            site[
                "longitude"
            ],

        "Elevation_m":
            site[
                "elevation"
            ],

        "Slope_degree":
            site[
                "slope"
            ],

        "TWI":
            site[
                "twi"
            ],

        "Storage_m3":
            pond[
                "target_storage"
            ],

        "Bottom_Length_m":
            pond[
                "bottom_length"
            ],

        "Bottom_Breadth_m":
            pond[
                "bottom_breadth"
            ],

        "Top_Length_m":
            pond[
                "top_length"
            ],

        "Top_Breadth_m":
            pond[
                "top_breadth"
            ],

        "Water_Depth_m":
            pond[
                "water_depth"
            ],

        "Freeboard_m":
            pond[
                "freeboard"
            ],

        "Excavation_Depth_m":
            pond[
                "excavation_depth"
            ],

        "Pond_Footprint_m2":
            pond[
                "footprint"
            ],

        "Seepage_Risk":
            screening[
                "seepage_risk"
            ],

        "Lining_Area_m2":
            pond[
                "lining_area"
            ],

        "Liner_Procurement_Area_m2":
            pond[
                "liner_purchase_area"
            ],

    }


    csv_data = (

        pd.DataFrame(
            [
                output
            ]
        )

        .to_csv(
            index=False
        )

        .encode(
            "utf-8"
        )

    )


    st.download_button(

        "⬇️ Download GeoPond-DSS Farmer Assessment",

        data=
            csv_data,

        file_name=
            "GeoPond_DSS_Farmer_Assessment.csv",

        mime=
            "text/csv",

        use_container_width=
            True,

    )


    st.markdown(
        """
<div class="result-card">

<b>🌱 Construction & Management Guidance</b>

<br><br>

• Confirm candidate position by field survey and
micro-topographic levelling.<br>

• Verify farmer ownership and boundary clearance.<br>

• Do not obstruct a natural stream for an off-stream pond.<br>

• Provide sediment trap and safe overflow arrangement.<br>

• Inspect foundation and excavation material.<br>

• Conduct field permeability testing before final lining choice.<br>

• Orient the pond according to local contour and drainage.<br>

• Stabilize bunds and disturbed soil using vegetation.

</div>
""",
        unsafe_allow_html=True,
    )


    st.warning(
        "Engineering note: GeoPond-DSS provides GIS-based "
        "screening and preliminary planning only. The selected "
        "coordinate represents a GIS candidate zone rather than "
        "a surveyed construction peg. Final pond siting, "
        "excavation, embankment, inlet, outlet, spillway and "
        "lining require field survey and engineering verification."
    )


# ============================================================
# 23. RESET
# ============================================================

st.markdown(
    "---"
)


if st.button(
    "🔄 Start New Farmer Assessment",
    use_container_width=True,
):

    reset_assessment()

<<<<<<< HEAD
    st.rerun()
=======
    st.rerun()
>>>>>>> b9aaff721a59b98f8315461236c4653bf3a2eb44
