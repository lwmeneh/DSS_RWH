# ============================================================
# GeoPond-DSS
# Geospatial Pond Siting and Preliminary Design
# Decision Support System
#
# ICAR Research Complex for NEH Region
# Umiam, Meghalaya
#
# FAST + ADVANCED WEB-GIS VERSION
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
import base64
import math
import re

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio

from rasterio.mask import (
    mask as rio_mask
)

from rasterio.transform import xy

from rasterio.warp import (
    reproject,
    Resampling,
)

import streamlit as st

import folium

from folium.plugins import (
    Draw,
    Fullscreen,
    LocateControl,
    MeasureControl,
    MousePosition,
)

from streamlit_folium import (
    st_folium
)

from shapely.geometry import (
    shape,
    mapping,
    Point,
    box,
)

from shapely.ops import (
    unary_union
)

from pyproj import (
    Transformer
)


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GeoPond-DSS",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

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


WORKING_CRS = (
    "EPSG:32646"
)

MAP_CENTER = [
    25.88,
    91.88,
]

MAP_ZOOM = 12


# ============================================================
# RASTERS
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
# VECTORS
# ============================================================

BOUNDARY_GPKG = (
    CACHE_DIR
    / "boundary_utm46.gpkg"
)

STREAMS_GPKG = (
    CACHE_DIR
    / "streams_utm46.gpkg"
)


# 2022 LULC
LULC_2022_GPKG = (
    CACHE_DIR
    / "lulc_2021_10k_clean_utm46.gpkg"
)


# Optional projected 2025
PROJECTED_LULC_2025_SHP = (
    CACHE_DIR
    / "2025_Projected_LULC.shp"
)


# Thematic data
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
# 4. ADVANCED CSS
# ============================================================

st.markdown(
    """
<style>

html,
body,
[class*="css"] {

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;

}


/* ---------------------------------------------------------
APP
--------------------------------------------------------- */

[data-testid="stAppViewContainer"] {

    background:
        linear-gradient(
            180deg,
            #eef5f3 0%,
            #f8fbfa 22%,
            #ffffff 60%
        );

}


[data-testid="stHeader"] {

    background:
        rgba(
            255,
            255,
            255,
            .62
        );

    backdrop-filter:
        blur(10px);

}


.block-container {

    max-width:
        1750px;

    padding-top:
        .2rem;

    padding-left:
        .7rem;

    padding-right:
        .7rem;

    padding-bottom:
        3rem;

}


/* ---------------------------------------------------------
HEADER
--------------------------------------------------------- */

.gp-header {

    background:
        linear-gradient(
            115deg,
            #043d32 0%,
            #056b51 47%,
            #07976f 100%
        );

    color:
        white;

    border-radius:
        0 0 18px 18px;

    padding:
        17px 24px;

    box-shadow:
        0 6px 20px
        rgba(
            0,
            60,
            48,
            .22
        );

}


.gp-header-row {

    display:
        flex;

    align-items:
        center;

    gap:
        18px;

}


.gp-logo img {

    width:
        82px;

    height:
        82px;

    object-fit:
        contain;

    background:
        white;

    padding:
        6px;

    border-radius:
        12px;

}


.gp-org {

    font-size:
        11px;

    letter-spacing:
        .11em;

    font-weight:
        800;

    opacity:
        .8;

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
        6px;

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

    padding:
        5px 11px;

    margin-top:
        8px;

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


/* ---------------------------------------------------------
NAV
--------------------------------------------------------- */

.gp-nav {

    display:
        flex;

    align-items:
        center;

    gap:
        5px;

    background:
        #163d37;

    border-radius:
        11px;

    padding:
        7px;

    margin:
        8px 0 12px 0;

    overflow-x:
        auto;

}


.gp-nav-item {

    padding:
        8px 13px;

    border-radius:
        7px;

    color:
        #dceae6;

    font-size:
        12px;

    font-weight:
        650;

    white-space:
        nowrap;

}


.gp-nav-active {

    background:
        #07936c;

    color:
        white;

}


.gp-online {

    margin-left:
        auto;

    background:
        #102f2a;

    color:
        #77eab6;

    border-radius:
        50px;

    padding:
        6px 10px;

    font-size:
        10px;

    font-weight:
        750;

}


/* ---------------------------------------------------------
PANELS
--------------------------------------------------------- */

.gp-panel {

    background:
        white;

    border:
        1px solid
        #d9e5e2;

    border-radius:
        13px;

    padding:
        13px 15px;

    margin-bottom:
        9px;

    box-shadow:
        0 3px 10px
        rgba(
            20,
            60,
            55,
            .05
        );

}


.gp-panel-title {

    color:
        #16483d;

    font-size:
        14px;

    font-weight:
        800;

    padding-bottom:
        7px;

    margin-bottom:
        8px;

    border-bottom:
        1px solid
        #e4ece9;

}


/* ---------------------------------------------------------
WORKFLOW
--------------------------------------------------------- */

.gp-workflow {

    background:
        #edf8fd;

    border:
        1px solid
        #d2e9f3;

    border-left:
        5px solid
        #1782b3;

    color:
        #25586e;

    border-radius:
        12px;

    padding:
        12px 14px;

    line-height:
        1.5;

    margin-bottom:
        10px;

}


/* ---------------------------------------------------------
DECISION CARD
--------------------------------------------------------- */

.decision-card {

    background:
        linear-gradient(
            125deg,
            #063e34,
            #08785d
        );

    color:
        white;

    border-radius:
        14px;

    padding:
        15px 17px;

    margin:
        9px 0 12px 0;

    box-shadow:
        0 5px 15px
        rgba(
            5,
            65,
            52,
            .16
        );

}


.decision-kicker {

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        .12em;

    opacity:
        .72;

}


.decision-value {

    font-size:
        24px;

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


/* ---------------------------------------------------------
GIS CARD
--------------------------------------------------------- */

.gis-card {

    background:
        white;

    border:
        1px solid
        #d8e5e1;

    border-radius:
        14px;

    padding:
        14px 16px;

    box-shadow:
        0 3px 10px
        rgba(
            20,
            60,
            55,
            .05
        );

    margin:
        9px 0;

}


.gis-card-title {

    color:
        #07614d;

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
        9px 10px;

}


.gis-grid span {

    display:
        block;

    color:
        #71817c;

    font-size:
        10px;

}


.gis-grid b {

    display:
        block;

    color:
        #183f36;

    font-size:
        14px;

    margin-top:
        2px;

}


/* ---------------------------------------------------------
METRICS
--------------------------------------------------------- */

[data-testid="stMetric"] {

    background:
        white;

    border:
        1px solid
        #d8e4e0;

    border-radius:
        12px;

    padding:
        10px 12px;

    box-shadow:
        0 2px 7px
        rgba(
            20,
            60,
            55,
            .05
        );

}


[data-testid="stMetricValue"] {

    color:
        #173f36;

    font-weight:
        800;

}


/* ---------------------------------------------------------
BUTTON
--------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {

    width:
        100%;

    min-height:
        48px;

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
            #09a176
        );

}


/* ---------------------------------------------------------
TABS
--------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {

    background:
        #edf4f2;

    padding:
        5px;

    gap:
        4px;

    border-radius:
        10px;

}


/* ---------------------------------------------------------
MOBILE
--------------------------------------------------------- */

@media only screen and (max-width:850px) {

    .block-container {

        padding-left:
            .25rem;

        padding-right:
            .25rem;

    }


    .gp-header-row {

        flex-direction:
            column;

        text-align:
            center;

    }


    .gp-title {

        font-size:
            25px;

    }


    .gp-logo img {

        width:
            70px;

        height:
            70px;

    }


    .gp-online {

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

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

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

        norm_name(
            c
        ):
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


# ============================================================
# 6. HEADER
# ============================================================

logo_base64 = (
    image_to_base64(
        ICAR_LOGO
    )
)


if logo_base64:

    logo_html = (

        '<div class="gp-logo">'

        f'<img src="data:image/png;base64,{logo_base64}">'

        '</div>'

    )

else:

    logo_html = (
        '<div style="font-size:52px;">🌾</div>'
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
Geospatial Pond Siting and Preliminary Design
Decision Support System
</div>

<div class="gp-tag">
Ri Bhoi • Meghalaya • Mobile GPS • Web-GIS
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
📍 Site
</div>

<div class="gp-nav-item">
📐 Design
</div>

<div class="gp-nav-item">
🪨 Geology
</div>

<div class="gp-nav-item">
📄 Report
</div>

<div class="gp-online">
● SYSTEM READY
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 7. REQUIRED FILE CHECK
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

]


missing = [

    str(
        path.relative_to(
            BASE_DIR
        )
    )

    for path
    in REQUIRED_FILES

    if not path.exists()

]


if missing:

    st.error(
        "GeoPond-DSS GIS database is incomplete."
    )

    st.code(
        "\n".join(
            missing
        )
    )

    st.stop()


# ============================================================
# 8. FAST VECTOR READERS
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def read_full_vector(
    path_string
):

    path = Path(
        path_string
    )

    if not path.exists():

        return None

    try:

        gdf = gpd.read_file(
            path,
            engine="pyogrio",
        )

    except Exception:

        gdf = gpd.read_file(
            path
        )

    if gdf.empty:

        return gdf

    if gdf.crs is None:

        return None

    if (
        gdf.crs.to_epsg()
        !=
        32646
    ):

        gdf = gdf.to_crs(
            WORKING_CRS
        )

    return gdf


@st.cache_data(
    show_spinner=False,
    max_entries=40,
)
def read_bbox_vector(
    path_string,
    bounds_tuple,
):

    path = Path(
        path_string
    )

    if not path.exists():

        return None

    bbox_value = tuple(
        float(
            v
        )
        for v
        in bounds_tuple
    )

    try:

        gdf = gpd.read_file(

            path,

            bbox=
                bbox_value,

            engine=
                "pyogrio",

        )

    except Exception:

        try:

            gdf = gpd.read_file(

                path,

                bbox=
                    bbox_value,

            )

        except Exception:

            return None

    if (
        gdf is None
        or
        gdf.empty
    ):

        return gdf

    if gdf.crs is None:

        return None

    if (
        gdf.crs.to_epsg()
        !=
        32646
    ):

        gdf = gdf.to_crs(
            WORKING_CRS
        )

    return gdf


def read_bbox_with_fallback(
    primary_path,
    fallback_path,
    bounds_tuple,
):

    result = (
        read_bbox_vector(

            str(
                primary_path
            ),

            tuple(
                bounds_tuple
            ),

        )
    )

    if (
        result is None
        or
        result.empty
    ):

        if fallback_path.exists():

            result = (
                read_bbox_vector(

                    str(
                        fallback_path
                    ),

                    tuple(
                        bounds_tuple
                    ),

                )
            )

    return result


# ============================================================
# ONLY DISTRICT BOUNDARY LOADS AT STARTUP
# ============================================================

boundary_utm = (
    read_full_vector(
        str(
            BOUNDARY_GPKG
        )
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
# 9. SESSION STATE
# ============================================================

for key, default in {

    "field_geometry":
        None,

    "assessment":
        None,

    "current_lulc":
        None,

}.items():

    if key not in (
        st.session_state
    ):

        st.session_state[
            key
        ] = default


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
# 10. GEOMETRY HELPERS
# ============================================================

def geometry_gdf(
    geometry_geojson,
    crs="EPSG:4326",
):

    return gpd.GeoDataFrame(

        {
            "id":
                [1]
        },

        geometry=[
            shape(
                geometry_geojson
            )
        ],

        crs=
            crs,

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

        to_utm_geometry(
            geometry_geojson
        )
        .area

        /
        10000.0

    )


def get_last_polygon(
    map_output
):

    if not map_output:

        return None

    drawings = (

        map_output.get(
            "all_drawings"
        )

        or

        []

    )

    polygon = None

    for drawing in drawings:

        if not drawing:

            continue

        geometry = (
            drawing.get(
                "geometry",
                {}
            )
        )

        if geometry.get(
            "type"
        ) in [

            "Polygon",

            "MultiPolygon",

        ]:

            polygon = geometry

    return polygon


# ============================================================
# 11. RASTER FUNCTIONS
# ============================================================

def raster_mean(
    raster_path,
    field_geojson,
):

    polygon = (
        geometry_gdf(
            field_geojson
        )
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

        try:

            clipped, _ = rio_mask(

                src,

                [
                    mapping(
                        geom
                    )
                ],

                crop=
                    True,

                indexes=
                    1,

                filled=
                    False,

                all_touched=
                    True,

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

            values = np.array(
                []
            )

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

                    indexes=
                        1,

                    masked=
                        True,

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

                else

                np.nan
            )

        except Exception:

            return np.nan


def field_reference_grid(
    field_geojson
):

    polygon = (
        geometry_gdf(
            field_geojson
        )
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

            crop=
                True,

            indexes=
                1,

            filled=
                False,

            all_touched=
                True,

        )

        if np.ma.isMaskedArray(
            data
        ):

            array = (

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

            array = data.astype(
                float
            )

            valid = (
                np.isfinite(
                    array
                )
            )

        return (

            array,

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

        dtype=
            np.float64,

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

    low = float(
        np.percentile(
            values,
            5,
        )
    )

    high = float(
        np.percentile(
            values,
            95,
        )
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
# 12. VECTOR HELPERS
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

    hit = gdf[
        gdf.intersects(
            point
        )
    ]

    if hit.empty:

        return None

    return hit.iloc[
        0
    ]


def nearest_distance(
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

    idx = (
        distances.idxmin()
    )

    return (

        gdf.loc[
            idx
        ],

        float(
            distances.loc[
                idx
            ]
        ),

    )


def extract_record_fields(
    record,
    mapping_dict,
):

    if record is None:

        return {

            name:
                "Unknown"

            for name
            in mapping_dict

        }

    output = {}

    columns = list(
        record.index
    )

    for name, candidates in (
        mapping_dict.items()
    ):

        column = (
            first_matching_column(
                columns,
                candidates,
            )
        )

        output[
            name
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
# 13. LULC
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

        "crop":
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

        "water":
            "Waterbody",

        "riverstream":
            "River/Stream",

    }

    return mapping_dict.get(
        key,
        text,
    )


def get_field_lulc(
    field_geojson
):

    field_geom = (
        to_utm_geometry(
            field_geojson
        )
    )

    field_area = float(
        field_geom.area
    )

    if field_area <= 0:

        return {

            "dominant":
                "Unknown",

            "confidence":
                0,

            "coverage":
                0,

            "composition":
                [],

        }

    bounds = (
        field_geom.bounds
    )

    lulc = (
        read_bbox_vector(

            str(
                LULC_2022_GPKG
            ),

            tuple(
                bounds
            ),

        )
    )

    if (
        lulc is None
        or
        lulc.empty
        or
        "LULC_2022"
        not in lulc.columns
    ):

        return {

            "dominant":
                "Unknown",

            "confidence":
                0,

            "coverage":
                0,

            "composition":
                [],

        }

    lulc = lulc[
        lulc.intersects(
            field_geom
        )
    ].copy()

    if lulc.empty:

        return {

            "dominant":
                "Unknown",

            "confidence":
                0,

            "coverage":
                0,

            "composition":
                [],

        }

    lulc[
        "_class"
    ] = (

        lulc[
            "LULC_2022"
        ]

        .apply(
            canonical_lulc_name
        )

    )

    lulc[
        "_area"
    ] = (

        lulc.geometry

        .intersection(
            field_geom
        )

        .area

    )

    grouped = (

        lulc

        .groupby(
            "_class"
        )[
            "_area"
        ]

        .sum()

        .reset_index()

        .sort_values(
            "_area",
            ascending=False,
        )

    )

    grouped = grouped[
        grouped[
            "_area"
        ]
        >
        0.5
    ]

    if grouped.empty:

        return {

            "dominant":
                "Unknown",

            "confidence":
                0,

            "coverage":
                0,

            "composition":
                [],

        }

    mapped_area = float(
        grouped[
            "_area"
        ]
        .sum()
    )

    coverage = min(

        100.0,

        mapped_area
        /
        field_area
        *
        100,

    )

    dominant = (
        grouped
        .iloc[0][
            "_class"
        ]
    )

    confidence = (

        grouped
        .iloc[0][
            "_area"
        ]
        /
        mapped_area
        *
        100

        if mapped_area > 0

        else 0

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
                            "_area"
                        ]
                        /
                        10000,
                        3,
                    ),

                "Percent of field":
                    round(
                        row[
                            "_area"
                        ]
                        /
                        field_area
                        *
                        100,
                        1,
                    ),

            }
        )

    return {

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

        "composition":
            composition,

    }


# ============================================================
# OPTIONAL 2025 PROJECTED LULC
# ============================================================

def get_projected_2025(
    field_geojson
):

    if not PROJECTED_LULC_2025_SHP.exists():

        return None

    field_geom = (
        to_utm_geometry(
            field_geojson
        )
    )

    layer = (
        read_bbox_vector(

            str(
                PROJECTED_LULC_2025_SHP
            ),

            tuple(
                field_geom.bounds
            ),

        )
    )

    if (
        layer is None
        or
        layer.empty
    ):

        return None

    class_field = (
        first_matching_column(

            layer.columns,

            [
                "Class_name",
                "class_name",
                "Class",
                "LULC",
            ],

        )
    )

    if class_field is None:

        return None

    layer = layer[
        layer.intersects(
            field_geom
        )
    ].copy()

    if layer.empty:

        return None

    layer[
        "_class"
    ] = (

        layer[
            class_field
        ]

        .apply(
            canonical_lulc_name
        )

    )

    layer[
        "_area"
    ] = (

        layer.geometry

        .intersection(
            field_geom
        )

        .area

    )

    grouped = (

        layer

        .groupby(
            "_class"
        )[
            "_area"
        ]

        .sum()

        .sort_values(
            ascending=False
        )

    )

    if grouped.empty:

        return None

    return {

        "dominant":
            grouped.index[
                0
            ],

        "share":
            (
                grouped.iloc[
                    0
                ]
                /
                grouped.sum()
                *
                100
            ),

    }


# ============================================================
# 14. SITE SELECTION
# ============================================================

def find_pond_site(
    field_geojson,
    streams_local,
    geomorph_local,
    shear_local,
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

    reference_shape = (
        elevation.shape
    )

    slope = (
        raster_to_reference(

            SLOPE_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
    )

    twi = (
        raster_to_reference(

            TWI_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
    )

    drainage = (
        raster_to_reference(

            DRAINAGE_DENSITY_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
    )

    flow = (
        raster_to_reference(

            FLOW_ACCUMULATION_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
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

    if np.count_nonzero(
        valid
    ) == 0:

        return None

    lowest_grid = np.where(

        valid,

        elevation,

        np.inf,

    )

    lowest_idx = (
        np.unravel_index(

            np.argmin(
                lowest_grid
            ),

            elevation.shape,

        )
    )

    lowest_x, lowest_y = xy(

        transform,

        lowest_idx[
            0
        ],

        lowest_idx[
            1
        ],

        offset="center",

    )

    elev_norm = (
        normalize_array(
            elevation,
            valid,
        )
    )

    slope_norm = (
        normalize_array(
            slope,
            valid,
        )
    )

    twi_norm = (
        normalize_array(
            twi,
            valid,
        )
    )

    drainage_norm = (
        normalize_array(
            drainage,
            valid,
        )
    )

    score = (

        0.45
        *
        (
            1
            -
            elev_norm
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

        flow_norm = (
            normalize_array(

                np.log1p(
                    np.maximum(
                        flow,
                        0,
                    )
                ),

                flow_valid,

            )
        )

        score = (
            score
            +
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
        streams_local
        is not None
        and
        not streams_local.empty
    ):

        stream_union = (
            unary_union(
                streams_local.geometry
                .tolist()
            )
        )

    order = (
        np.argsort(
            score.ravel()
        )[::-1]
    )

    best = None

    for flat_idx in (
        order[
            :1500
        ]
    ):

        r, c = (
            np.unravel_index(

                flat_idx,

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

            offset=
                "center",

        )

        point = Point(
            px,
            py,
        )

        boundary_distance = float(

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

            stream_distance = float(

                point.distance(
                    stream_union
                )

            )

            if (
                stream_distance
                <
                10
            ):

                continue

        adjusted_score = float(
            score[
                r,
                c,
            ]
        )

        shear_distance = (
            nearest_distance(
                shear_local,
                px,
                py,
            )
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
                geomorph_local,
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
                (
                    float(
                        flow[
                            r,
                            c,
                        ]
                    )

                    if
                    np.isfinite(
                        flow[
                            r,
                            c,
                        ]
                    )

                    else

                    np.nan
                ),

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
                boundary_distance,

            "stream_distance":
                stream_distance,

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

    low_lon, low_lat = (
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

        "lowest": {

            "longitude":
                float(
                    low_lon
                ),

            "latitude":
                float(
                    low_lat
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
# 15. POND DESIGN
# ============================================================

def calculate_pond_geometry(

    bottom_breadth,

    water_depth,

    freeboard,

    side_slope,

    ratio,

):

    bottom_length = (
        bottom_breadth
        *
        ratio
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

    long_side = (

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

    short_side = (

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
        long_side

        +

        2
        *
        short_side

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

    available_water,

    field_area_ha_value,

    storage_fraction,

    maximum_land_fraction,

    water_depth,

    freeboard,

    side_slope,

    ratio,

):

    field_area_m2 = (

        field_area_ha_value
        *
        10000

    )

    required_storage = (

        available_water
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
        70
    ):

        mid = (
            low
            +
            high
        ) / 2

        geometry = (
            calculate_pond_geometry(

                mid,

                water_depth,

                freeboard,

                side_slope,

                ratio,

            )
        )

        if (
            geometry[
                "footprint"
            ]
            >
            maximum_footprint
        ):

            high = mid

        else:

            low = mid

    maximum_geometry = (
        calculate_pond_geometry(

            low,

            water_depth,

            freeboard,

            side_slope,

            ratio,

        )
    )

    target_storage = min(

        required_storage,

        maximum_geometry[
            "volume"
        ],

    )

    lower = 0.5

    upper = max(
        low,
        0.5,
    )

    for _ in range(
        70
    ):

        mid = (
            lower
            +
            upper
        ) / 2

        geometry = (
            calculate_pond_geometry(

                mid,

                water_depth,

                freeboard,

                side_slope,

                ratio,

            )
        )

        if (
            geometry[
                "volume"
            ]
            >
            target_storage
        ):

            upper = mid

        else:

            lower = mid

    result = (
        calculate_pond_geometry(

            lower,

            water_depth,

            freeboard,

            side_slope,

            ratio,

        )
    )

    result[
        "target_storage"
    ] = float(
        target_storage
    )

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
# 16. SEEPAGE / TERRAIN
# ============================================================

def classify_soil(
    texture
):

    text = (
        safe_text(
            texture
        )
        .lower()
    )

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


def classify_terrain(
    landform,
    slope
):

    text = (
        safe_text(
            landform
        )
        .lower()
    )

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


def classify_shear(
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

    soil_text, soil_score = (
        classify_soil(
            soil_texture
        )
    )

    terrain_text, terrain_score = (
        classify_terrain(
            landform,
            slope,
        )
    )

    shear_text, shear_score = (
        classify_shear(
            shear_distance
        )
    )

    lithology_text = (
        safe_text(
            lithology
        )
        .lower()
    )

    lithology_score = 1

    if any(

        term
        in lithology_text

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

    total = (

        soil_score

        +

        shear_score

        +

        lithology_score

        +

        lineament_score

    )

    if total >= 6:

        seepage = "HIGH"

    elif total >= 3:

        seepage = "MODERATE"

    else:

        seepage = "LOW"

    if seepage == "HIGH":

        lining = (
            "High GIS-screened seepage concern. "
            "Field permeability testing and engineered "
            "lining are likely required."
        )

    elif seepage == "MODERATE":

        lining = (
            "Moderate GIS-screened seepage concern. "
            "Check permeability and available clay. "
            "Compacted clay or geomembrane lining "
            "may be required."
        )

    else:

        lining = (
            "Lower GIS-screened seepage concern. "
            "Field permeability verification is still "
            "required before omitting lining."
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
# 17. FARMER INFORMATION
# ============================================================

with st.expander(
    "👨‍🌾 Farmer / Field Information",
    expanded=False,
):

    f1, f2 = st.columns(
        2
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
# 18. MAP
# ============================================================

st.markdown(
    "## 1️⃣ Locate and Draw Farmer Field"
)


st.markdown(
    """
<div class="gp-workflow">

<b>📍 Quick Field Workflow</b><br>

1. Use GPS or satellite imagery to locate the field.<br>
2. Zoom to the farm parcel.<br>
3. Select the polygon tool and trace the field.<br>
4. Close the polygon.<br>
5. GeoPond-DSS will calculate the field area and retrieve only the required GIS information.

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

    show=
        True,

    max_zoom=
        20,

).add_to(
    field_map
)


folium.TileLayer(

    "OpenStreetMap",

    name=
        "Road / Village Map",

    show=
        False,

).add_to(
    field_map
)


if boundary_wgs is not None:

    folium.GeoJson(

        boundary_wgs
        .__geo_interface__,

        name=
            "Ri Bhoi Boundary",

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
                    .25,

            },

    ).add_to(
        field_map
    )


Draw(

    export=
        False,

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
                    .25,

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


MeasureControl(

    position=
        "topleft",

    primary_length_unit=
        "meters",

    primary_area_unit=
        "sqmeters",

).add_to(
    field_map
)


MousePosition(

    position=
        "bottomleft",

    prefix=
        "Lat/Lon:",

    num_digits=
        5,

).add_to(
    field_map
)


Fullscreen(
    position=
        "topleft"
).add_to(
    field_map
)


folium.LayerControl(
    collapsed=True
).add_to(
    field_map
)


map_output = st_folium(

    field_map,

    height=
        550,

    use_container_width=
        True,

    key=
        "geopond_field_map",

    returned_objects=[
        "all_drawings"
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
# 19. LAND USE
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

    with st.spinner(
        "Reading LULC only for the selected field..."
    ):

        lulc_result = (
            get_field_lulc(
                field
            )
        )

    mapped_lulc = (
        lulc_result[
            "dominant"
        ]
    )


    st.success(
        f"✅ Field captured: "
        f"{field_area:.3f} ha"
    )


    st.markdown(
        "## 🌾 Land-Use Assessment"
    )


    c1, c2, c3 = st.columns(
        3
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
        f"{lulc_result['coverage']:.1f}%",
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
Field observation is authoritative for farm-level planning.
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

        "AGREEMENT"

        if agreement

        else

        "GIS / FIELD DIFFERENCE"

    )


    st.markdown(
        f"""
<div class="gis-card">

<div class="gis-card-title">
🗺 GIS Land-Use Evidence
</div>

<div class="gis-grid">

<div>
<span>Mapped 2022</span>
<b>{mapped_lulc}</b>
</div>

<div>
<span>Mapped share</span>
<b>{lulc_result['confidence']:.1f}%</b>
</div>

<div>
<span>Coverage</span>
<b>{lulc_result['coverage']:.1f}%</b>
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
        "View mapped LULC composition",
        expanded=False,
    ):

        if lulc_result[
            "composition"
        ]:

            st.dataframe(

                pd.DataFrame(
                    lulc_result[
                        "composition"
                    ]
                ),

                use_container_width=
                    True,

                hide_index=
                    True,

            )


    load_2025 = (
        st.checkbox(
            "Load optional 2025 projected LULC comparison",
            value=False,
        )
    )


    if load_2025:

        with st.spinner(
            "Reading projected 2025 LULC..."
        ):

            projected = (
                get_projected_2025(
                    field
                )
            )

        if projected:

            st.info(
                f"Projected 2025 class: "
                f"**{projected['dominant']}** "
                f"({projected['share']:.1f}% of mapped overlap)"
            )

        else:

            st.warning(
                "2025 projected LULC is unavailable for this field."
            )


# ============================================================
# 20. PLANNING SETTINGS
# ============================================================

if field:

    with st.expander(
        "⚙️ Preliminary Pond Planning Settings",
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


        ratio = (
            st.slider(

                "Bottom length : breadth ratio",

                1.0,

                2.0,

                1.5,

                0.1,

            )
        )


# ============================================================
# 21. RUN DSS
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
            "Analysing runoff and terrain..."
        ):

            area_ha = (
                field_area_ha(
                    field
                )
            )


            rainfall = (
                raster_mean(
                    RAINFALL_RASTER,
                    field,
                )
            )


            mean_runoff = (
                raster_mean(
                    MEAN_RUNOFF_RASTER,
                    field,
                )
            )


            dependable_runoff = (
                raster_mean(
                    DEPENDABLE_RUNOFF_RASTER,
                    field,
                )
            )


            cn = (
                raster_mean(
                    CN_RASTER,
                    field,
                )
            )


            runoff_coefficient = (
                raster_mean(
                    RUNOFF_COEFFICIENT_RASTER,
                    field,
                )
            )


            mean_volume = (

                mean_runoff
                *
                area_ha
                *
                10

            )


            dependable_volume = (

                dependable_runoff
                *
                area_ha
                *
                10

            )


            pond = (
                design_farm_pond(

                    dependable_volume,

                    area_ha,

                    storage_fraction_percent
                    /
                    100,

                    maximum_land_percent
                    /
                    100,

                    water_depth,

                    freeboard,

                    side_slope,

                    ratio,

                )
            )


        field_geom = (
            to_utm_geometry(
                field
            )
        )


        # Only local GIS data are read
        field_search = (
            field_geom
            .buffer(
                1000
            )
        )


        with st.spinner(
            "Reading only nearby GIS features..."
        ):


            streams_local = (

                read_bbox_vector(

                    str(
                        STREAMS_GPKG
                    ),

                    tuple(
                        field_search.bounds
                    ),

                )

                if STREAMS_GPKG.exists()

                else

                None

            )


            geomorph_local = (
                read_bbox_vector(

                    str(
                        GEOMORPHOLOGY_GPKG
                    ),

                    tuple(
                        field_geom.bounds
                    ),

                )
            )


            shear_search = (
                field_geom
                .buffer(
                    300
                )
            )


            shear_local_for_site = (
                read_bbox_vector(

                    str(
                        SHEAR_ZONE_GPKG
                    ),

                    tuple(
                        shear_search.bounds
                    ),

                )
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

                streams_local,

                geomorph_local,

                shear_local_for_site,

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
                "No suitable GIS candidate could be identified inside this field."
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


        # ====================================================
        # READ ONLY AROUND FINAL POND SITE
        # ====================================================

        candidate_search_100 = box(

            site[
                "x"
            ]
            -
            100,

            site[
                "y"
            ]
            -
            100,

            site[
                "x"
            ]
            +
            100,

            site[
                "y"
            ]
            +
            100,

        )


        with st.spinner(
            "Retrieving local soil and geology..."
        ):


            soil_local = (
                read_bbox_with_fallback(

                    SOIL_GPKG,

                    SOIL_SHP,

                    candidate_search_100.bounds,

                )
            )


            lithology_local = (
                read_bbox_vector(

                    str(
                        LITHOLOGY_GPKG
                    ),

                    tuple(
                        candidate_search_100.bounds
                    ),

                )
            )


            geomorph_site = (
                read_bbox_vector(

                    str(
                        GEOMORPHOLOGY_GPKG
                    ),

                    tuple(
                        candidate_search_100.bounds
                    ),

                )
            )


            # Actual nearest shear search
            shear_search_large = box(

                site[
                    "x"
                ]
                -
                15000,

                site[
                    "y"
                ]
                -
                15000,

                site[
                    "x"
                ]
                +
                15000,

                site[
                    "y"
                ]
                +
                15000,

            )


            shear_nearby = (
                read_bbox_vector(

                    str(
                        SHEAR_ZONE_GPKG
                    ),

                    tuple(
                        shear_search_large.bounds
                    ),

                )
            )


            # C-horizon local search
            c_search = box(

                site[
                    "x"
                ]
                -
                15000,

                site[
                    "y"
                ]
                -
                15000,

                site[
                    "x"
                ]
                +
                15000,

                site[
                    "y"
                ]
                +
                15000,

            )


            c_horizon_local = (
                read_bbox_vector(

                    str(
                        SOIL_C_HORIZON_GPKG
                    ),

                    tuple(
                        c_search.bounds
                    ),

                )
            )


        soil_record = (
            point_record(

                soil_local,

                site[
                    "x"
                ],

                site[
                    "y"
                ],

            )
        )


        geom_record = (
            point_record(

                geomorph_site,

                site[
                    "x"
                ],

                site[
                    "y"
                ],

            )
        )


        lith_record = (
            point_record(

                lithology_local,

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
                            "texture_class",
                        ],

                },

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
                        ],

                },

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
            nearest_distance(

                shear_nearby,

                site[
                    "x"
                ],

                site[
                    "y"
                ],

            )
        )


        c_record, c_distance = (
            nearest_record(

                c_horizon_local,

                site[
                    "x"
                ],

                site[
                    "y"
                ],

            )
        )


        c_info = (
            extract_record_fields(

                c_record,

                {

                    "sample_id":
                        [
                            "sample_id",
                            "sampleid",
                            "id",
                            "sample_no",
                        ],

                },

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
                area_ha,

            "current_lulc":
                current_lulc,

            "mapped_lulc":
                mapped_lulc,

            "lulc_confidence":
                lulc_result[
                    "confidence"
                ],

            "lulc_coverage":
                lulc_result[
                    "coverage"
                ],

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

            "c_horizon_info":
                c_info,

            "c_horizon_distance":
                c_distance,

            "screening":
                screening,

        }


        st.rerun()


# ============================================================
# 22. RESULTS
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

    pond = (
        assessment[
            "pond"
        ]
    )

    lowest = (
        assessment[
            "lowest"
        ]
    )

    screening = (
        assessment[
            "screening"
        ]
    )


    st.markdown(
        "## ✅ GeoPond-DSS Assessment"
    )


    st.markdown(
        f"""
<div class="gis-card">

<div class="gis-card-title">
📊 Decision Dashboard
</div>

<div class="gis-grid">

<div>
<span>Land use</span>
<b>{assessment['current_lulc']}</b>
</div>

<div>
<span>Terrain</span>
<b>{screening['terrain_constraint']}</b>
</div>

<div>
<span>Seepage</span>
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


    m1, m2, m3, m4 = (
        st.columns(
            4
        )
    )


    m1.metric(

        "Field Area",

        f"{assessment['field_area']:.3f} ha",

    )


    m2.metric(

        "Dependable Water",

        f"{assessment['dependable_volume']:,.0f} m³/year",

    )


    m3.metric(

        "Storage Target",

        f"{pond['target_storage']:,.0f} m³",

    )


    m4.metric(

        "Seepage",

        screening[
            "seepage_risk"
        ],

    )


    (
        tab_overview,
        tab_map,
        tab_hydrology,
        tab_design,
        tab_geology,
        tab_technical,
    ) = st.tabs(
        [

            "📊 Overview",

            "🗺 Map",

            "💧 Hydrology",

            "📐 Pond Design",

            "🪨 Geology",

            "🔬 Technical",

        ]
    )


    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab_overview:

        st.success(
            "Best GIS-screened candidate identified. "
            "The coordinate represents a candidate zone, "
            "not a final construction peg."
        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Latitude",

            f"{site['latitude']:.6f}",

        )


        c2.metric(

            "Longitude",

            f"{site['longitude']:.6f}",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Elevation",

            f"{site['elevation']:.1f} m",

        )


        c2.metric(

            "Slope",

            f"{site['slope']:.2f}°",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "TWI",

            f"{site['twi']:.2f}",

        )


        c2.metric(

            "Boundary Clearance",

            f"{site['boundary_distance']:.1f} m",

        )


        if site[
            "stream_distance"
        ] is not None:

            st.metric(

                "Distance to Mapped Stream",

                f"{site['stream_distance']:.1f} m",

            )


    # ========================================================
    # RESULT MAP
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
                "Esri",

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
                        .18,

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

            .iloc[
                0
            ]

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
                        .30,

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
                "Absolute lowest DEM location",

            icon=
                folium.Icon(
                    color="orange"
                ),

        ).add_to(
            result_map
        )


        Fullscreen(
            position="topleft"
        ).add_to(
            result_map
        )


        MeasureControl(
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
                570,

            use_container_width=
                True,

            key=
                "geopond_result_map",

            returned_objects=[],

        )


    # ========================================================
    # HYDROLOGY
    # ========================================================

    with tab_hydrology:


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Spatial Annual Rainfall",

            f"{assessment['rainfall']:.0f} mm/year",

        )


        c2.metric(

            "Mean Annual Runoff",

            f"{assessment['mean_runoff']:.0f} mm",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "75% Dependable Runoff",

            f"{assessment['dependable_runoff']:.0f} mm",

        )


        c2.metric(

            "75% Dependable Water",

            f"{assessment['dependable_volume']:,.0f} m³/year",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "CN-II",

            f"{assessment['cn']:.1f}",

        )


        c2.metric(

            "Runoff Coefficient",

            f"{assessment['runoff_coefficient']:.3f}",

        )


    # ========================================================
    # DESIGN
    # ========================================================

    with tab_design:


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Storage",

            f"{pond['target_storage']:,.0f} m³",

        )


        c2.metric(

            "Water Depth",

            f"{pond['water_depth']:.2f} m",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Bottom Length",

            f"{pond['bottom_length']:.1f} m",

        )


        c2.metric(

            "Bottom Breadth",

            f"{pond['bottom_breadth']:.1f} m",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Top Length",

            f"{pond['top_length']:.1f} m",

        )


        c2.metric(

            "Top Breadth",

            f"{pond['top_breadth']:.1f} m",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Excavation Depth",

            f"{pond['excavation_depth']:.2f} m",

        )


        c2.metric(

            "Freeboard",

            f"{pond['freeboard']:.2f} m",

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Footprint",

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


        c1, c2 = st.columns(
            2
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


        c1, c2 = st.columns(
            2
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


        c1, c2 = st.columns(
            2
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


        c1, c2 = st.columns(
            2
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

                ">15 km or unavailable"
            ),

        )


        c2.metric(

            "Shear Screening",

            screening[
                "shear_concern"
            ],

        )


        c1, c2 = st.columns(
            2
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

            "Seepage Risk",

            screening[
                "seepage_risk"
            ],

        )


        if (
            assessment[
                "c_horizon_distance"
            ]
            is not None
        ):

            st.write(
                "**Nearest C-horizon sample:**",
                f"{assessment['c_horizon_distance']:.0f} m",
            )


        st.info(
            screening[
                "lining_recommendation"
            ]
        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Lining Area",

            f"{pond['lining_area']:,.0f} m²",

        )


        c2.metric(

            "Procurement +10%",

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
            "**Planning land use:**",
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
            "**Mapped LULC share:**",
            f"{assessment['lulc_confidence']:.1f}%",
        )


        st.write(
            "**Mapped LULC coverage:**",
            f"{assessment['lulc_coverage']:.1f}%",
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
            (
                round(
                    site[
                        "flow_accumulation"
                    ],
                    2,
                )

                if
                np.isfinite(
                    site[
                        "flow_accumulation"
                    ]
                )

                else

                "No data"
            ),
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


    # ========================================================
    # CSV REPORT
    # ========================================================

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

        "Confirmed_Current_Land_Use":
            assessment[
                "current_lulc"
            ],

        "Mapped_2022_LULC":
            assessment[
                "mapped_lulc"
            ],

        "Mapped_2022_Class_Share_percent":
            assessment[
                "lulc_confidence"
            ],

        "Mapped_2022_Coverage_percent":
            assessment[
                "lulc_coverage"
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

        "Site_Score":
            site[
                "adjusted_score"
            ],

        "Soil_Texture":
            assessment[
                "soil_info"
            ][
                "texture"
            ],

        "Geomorphology":
            assessment[
                "geom_info"
            ][
                "landform"
            ],

        "Lithological_Group":
            assessment[
                "lith_info"
            ][
                "group"
            ],

        "Formation":
            assessment[
                "lith_info"
            ][
                "formation"
            ],

        "Lithology":
            assessment[
                "lith_info"
            ][
                "lithology"
            ],

        "Geological_Age":
            assessment[
                "lith_info"
            ][
                "age"
            ],

        "Nearest_Shear_Zone_m":
            assessment[
                "shear_distance"
            ],

        "Lineament_Density":
            assessment[
                "lineament_density"
            ],

        "Seepage_Risk":
            screening[
                "seepage_risk"
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

        "Field_Occupied_percent":
            pond[
                "occupied_percent"
            ],

        "Lining_Area_m2":
            pond[
                "lining_area"
            ],

        "Liner_Procurement_Area_m2":
            pond[
                "liner_purchase_area"
            ],

        "Lining_Recommendation":
            screening[
                "lining_recommendation"
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


    st.warning(
        "Engineering note: GeoPond-DSS provides GIS-based "
        "screening and preliminary planning. Final siting, "
        "pond orientation, excavation, embankment, inlet, "
        "outlet, spillway and lining require field survey, "
        "permeability testing and engineering verification."
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

    st.rerun()