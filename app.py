# ============================================================
# RI BHOI RAINWATER HARVESTING GIS DSS
# ICAR RESEARCH COMPLEX FOR NEH REGION
# UMIAM, MEGHALAYA
#
# FARMER FIELD
# -> WATER AVAILABILITY
# -> AUTOMATIC FARM-POND SITE
# -> PRELIMINARY POND DESIGN
# -> SEEPAGE / LINING GUIDANCE
# ============================================================

from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.transform import xy

import streamlit as st
import folium

from folium.plugins import (
    Draw,
    Fullscreen,
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
    page_title="ICAR Ri Bhoi RWH GIS DSS",
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


# ------------------------------------------------------------
# 10-FACTOR EXTRACTED RASTERS
# ------------------------------------------------------------

GEOLOGY_RASTER = (
    CACHE_DIR
    / "01_Geology.tif"
)

GEOMORPH_RASTER = (
    CACHE_DIR
    / "02_Geomorphology.tif"
)

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

LULC_RANK_RASTER = (
    CACHE_DIR
    / "07_LULC.tif"
)

LINEAMENT_DENSITY_RASTER = (
    CACHE_DIR
    / "08_LineamentDensity.tif"
)

SOIL_RASTER = (
    CACHE_DIR
    / "09_Soil.tif"
)

TWI_RASTER = (
    CACHE_DIR
    / "10_TWI.tif"
)


# ------------------------------------------------------------
# HYDROLOGY
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# VECTOR LAYERS
# ------------------------------------------------------------

BOUNDARY_GPKG = (
    CACHE_DIR
    / "boundary_utm46.gpkg"
)

STREAMS_GPKG = (
    CACHE_DIR
    / "streams_utm46.gpkg"
)

LULC_GPKG = (
    CACHE_DIR
    / "lulc_2021_10k_clean_utm46.gpkg"
)

MGNREGA_GPKG = (
    CACHE_DIR
    / "mgnrega_utm46.gpkg"
)

FINAL94_GPKG = (
    CACHE_DIR
    / "final94_utm46.gpkg"
)

PUBLISHED_GPKG = (
    CACHE_DIR
    / "published_sites_utm46.gpkg"
)


# Optional
AHP_RANK_STACK = (
    CACHE_DIR
    / "ahp_rank_stack.tif"
)


WORKING_CRS = "EPSG:32646"

MAP_CENTER = [
    25.88,
    91.88,
]

MAP_ZOOM = 10


# ============================================================
# 3. MOBILE-FRIENDLY CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 0.45rem;
    padding-left: 0.7rem;
    padding-right: 0.7rem;
    padding-bottom: 3rem;
}

.stButton > button,
.stDownloadButton > button {

    width: 100%;
    min-height: 54px;

    border-radius: 12px;

    font-size: 16px;
    font-weight: 650;
}

[data-testid="stMetric"] {

    background: #ffffff;

    border:
        1px solid #d8e5e8;

    border-radius: 12px;

    padding: 10px;
}

.card {

    background: white;

    border:
        1px solid #d8e5e8;

    border-radius: 12px;

    padding: 14px;

    margin: 8px 0;
}

.green-card {
    border-left:
        5px solid #159957;
}

.orange-card {
    border-left:
        5px solid #EF6C00;
}

.red-card {
    border-left:
        5px solid #C62828;
}


@media only screen and (max-width: 768px) {

    .block-container {

        padding-left: 0.25rem;

        padding-right: 0.25rem;
    }

    h1 {
        font-size:
            1.45rem !important;
    }

    h2 {
        font-size:
            1.20rem !important;
    }

    h3 {
        font-size:
            1.05rem !important;
    }

    .stButton > button,
    .stDownloadButton > button {

        min-height: 58px;
    }

    [data-testid="stMetricValue"] {

        font-size:
            1.15rem !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    """
<div style="
background:
linear-gradient(
120deg,
#064c44,
#087c6c
);
padding:20px;
border-radius:16px;
color:white;
text-align:center;
margin-bottom:14px;
">

<div style="
font-size:14px;
font-weight:650;
letter-spacing:0.4px;
">

INDIAN COUNCIL OF AGRICULTURAL RESEARCH

</div>

<div style="
font-size:26px;
font-weight:800;
margin-top:4px;
">

ICAR Research Complex for NEH Region

</div>

<div style="
font-size:17px;
margin-top:3px;
">

Umiam, Meghalaya

</div>

<hr style="
border:none;
border-top:
1px solid
rgba(255,255,255,0.35);
">

<div style="
font-size:23px;
font-weight:750;
">

💧 Rainwater Harvesting GIS Decision Support System

</div>

<div style="
font-size:15px;
margin-top:5px;
">

Farmer-Level Farm Pond Siting & Preliminary Design
<br>
Ri Bhoi District, Meghalaya

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 5. REQUIRED DATA CHECK
# ============================================================

REQUIRED_FILES = [

    GEOLOGY_RASTER,
    GEOMORPH_RASTER,

    SLOPE_RASTER,

    RAINFALL_RASTER,

    ELEVATION_RASTER,

    DRAINAGE_DENSITY_RASTER,

    LINEAMENT_DENSITY_RASTER,

    SOIL_RASTER,

    TWI_RASTER,

    FLOW_ACCUMULATION_RASTER,

    CN_RASTER,

    MEAN_RUNOFF_RASTER,

    DEPENDABLE_RUNOFF_RASTER,

    RUNOFF_COEFFICIENT_RASTER,

    BOUNDARY_GPKG,

    STREAMS_GPKG,
]


missing = [

    str(
        p.relative_to(
            BASE_DIR
        )
    )

    for p
    in REQUIRED_FILES

    if not p.exists()

]


if missing:

    st.error(
        "Required GIS files are missing."
    )

    st.write(
        "Copy the following files into `data/cache/`:"
    )

    st.code(
        "\n".join(
            missing
        )
    )

    st.stop()


# ============================================================
# 6. LOAD VECTOR DATA
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def read_vector(
    path_string
):

    path = Path(
        path_string
    )

    if not path.exists():

        return None

    try:

        gdf = gpd.read_file(
            path
        )

        if gdf.crs is None:

            gdf = (
                gdf
                .set_crs(
                    WORKING_CRS
                )
            )

        return gdf

    except Exception:

        return None


boundary_utm = read_vector(
    str(
        BOUNDARY_GPKG
    )
)

streams_utm = read_vector(
    str(
        STREAMS_GPKG
    )
)

lulc_utm = read_vector(
    str(
        LULC_GPKG
    )
)

mgnrega_utm = read_vector(
    str(
        MGNREGA_GPKG
    )
)

final94_utm = read_vector(
    str(
        FINAL94_GPKG
    )
)

published_utm = read_vector(
    str(
        PUBLISHED_GPKG
    )
)


boundary_wgs = None


if (
    boundary_utm is not None
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
# 7. SESSION STATE
# ============================================================

DEFAULT_STATE = {

    "field_geometry":
        None,

    "current_lulc":
        None,

    "assessment":
        None,
}


for key, value in (
    DEFAULT_STATE.items()
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
        "current_lulc"
    ] = None

    st.session_state[
        "assessment"
    ] = None


# ============================================================
# 8. GEOMETRY FUNCTIONS
# ============================================================

def get_last_polygon(
    map_data
):

    if not map_data:

        return None

    drawings = (
        map_data.get(
            "all_drawings"
        )
        or []
    )

    polygon = None


    for drawing in drawings:

        if not drawing:

            continue

        geom = (
            drawing.get(
                "geometry",
                {}
            )
        )

        if geom.get(
            "type"
        ) in [

            "Polygon",

            "MultiPolygon",

        ]:

            polygon = geom


    return polygon


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

    geom = to_utm_geometry(
        geometry_geojson
    )

    return float(
        geom.area
        /
        10000.0
    )


# ============================================================
# 9. RASTER STATISTICS
# ============================================================

def safe_raster_stats(

    raster_path,

    polygon_geojson,

    centroid_fallback=True,

):

    polygon = geometry_gdf(
        polygon_geojson
    )


    with rasterio.open(
        raster_path
    ) as src:

        if src.crs is None:

            return {

                "mean":
                    np.nan,

                "median":
                    np.nan,

                "min":
                    np.nan,

                "max":
                    np.nan,

            }


        geom = (

            polygon

            .to_crs(
                src.crs
            )

            .geometry
            .iloc[0]

        )


        values = np.array(
            []
        )


        try:

            clipped, _ = (
                rio_mask(

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


            if (
                src.nodata
                is not None
            ):

                values = values[
                    values
                    != src.nodata
                ]


        except Exception:

            values = np.array(
                []
            )


        # Small farmer-field fallback

        if (
            len(values) == 0
            and
            centroid_fallback
        ):

            c = geom.centroid

            try:

                sampled = next(

                    src.sample(

                        [
                            (
                                c.x,
                                c.y
                            )
                        ],

                        indexes=1,

                        masked=True,

                    )
                )


                values = (
                    np.asarray(
                        sampled
                    )
                    .reshape(-1)
                )


                values = values[
                    np.isfinite(
                        values
                    )
                ]


            except Exception:

                values = (
                    np.array(
                        []
                    )
                )


        if len(values) == 0:

            return {

                "mean":
                    np.nan,

                "median":
                    np.nan,

                "min":
                    np.nan,

                "max":
                    np.nan,

            }


        return {

            "mean":

                float(
                    np.mean(
                        values
                    )
                ),

            "median":

                float(
                    np.median(
                        values
                    )
                ),

            "min":

                float(
                    np.min(
                        values
                    )
                ),

            "max":

                float(
                    np.max(
                        values
                    )
                ),

        }


# ============================================================
# 10. SAMPLE RASTER AT POINT
# ============================================================

def sample_raster_at_xy(

    raster_path,

    x,

    y,

    point_crs=
        WORKING_CRS,

):

    with rasterio.open(
        raster_path
    ) as src:

        transformer = (
            Transformer
            .from_crs(

                point_crs,

                src.crs,

                always_xy=True,

            )
        )


        rx, ry = (
            transformer
            .transform(
                x,
                y
            )
        )


        try:

            value = next(

                src.sample(

                    [
                        (
                            rx,
                            ry
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


            if not np.isfinite(
                value
            ):

                return np.nan


            return value


        except Exception:

            return np.nan


# ============================================================
# 11. REFERENCE GRID
# ============================================================

def crop_reference_grid(
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


        data, transform = (
            rio_mask(

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
                    "float64"
                )
            )

            valid = (
                ~data.mask
            )

        else:

            array = (
                data
                .astype(
                    "float64"
                )
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


# ============================================================
# 12. RESAMPLE TO FIELD GRID
# ============================================================

def read_to_reference(

    raster_path,

    destination_shape,

    destination_transform,

    destination_crs,

):

    destination = (
        np.full(

            destination_shape,

            np.nan,

            dtype="float64",

        )
    )


    with rasterio.open(
        raster_path
    ) as src:

        source = (

            src.read(1)
            .astype(
                "float64"
            )

        )


        if src.nodata is not None:

            source[
                source
                ==
                src.nodata
            ] = np.nan


        reproject(

            source=source,

            destination=
                destination,

            src_transform=
                src.transform,

            src_crs=
                src.crs,

            dst_transform=
                destination_transform,

            dst_crs=
                destination_crs,

            src_nodata=
                np.nan,

            dst_nodata=
                np.nan,

            resampling=
                Resampling.bilinear,

        )


    return destination


# ============================================================
# 13. NORMALIZATION
# ============================================================

def normalize_array(

    array,

    valid_mask,

):

    values = array[
        valid_mask
    ]


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
        5
    )

    high = np.percentile(
        values,
        95
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


    result = (

        array
        -
        low

    ) / (

        high
        -
        low

    )


    return np.clip(
        result,
        0,
        1,
    )


# ============================================================
# 14. DETAILED LULC
# ============================================================

def mapped_lulc_summary(
    polygon_geojson
):

    if (
        lulc_utm is None
        or
        lulc_utm.empty
        or
        "LULC_2022"
        not in lulc_utm.columns
    ):

        return []


    selection = (

        geometry_gdf(
            polygon_geojson
        )

        .to_crs(
            lulc_utm.crs
        )

    )


    geom = (
        selection
        .geometry
        .iloc[0]
    )


    ids = list(

        lulc_utm
        .sindex
        .intersection(
            geom.bounds
        )

    )


    if not ids:

        return []


    candidate = (

        lulc_utm
        .iloc[
            ids
        ]
        .copy()

    )


    candidate = candidate[
        candidate.intersects(
            geom
        )
    ].copy()


    if candidate.empty:

        return []


    try:

        candidate[
            "geometry"
        ] = (
            candidate
            .geometry
            .make_valid()
        )

    except Exception:

        candidate[
            "geometry"
        ] = (
            candidate
            .geometry
            .buffer(0)
        )


    candidate[
        "clip_geom"
    ] = (

        candidate
        .geometry
        .intersection(
            geom
        )

    )


    candidate = candidate[

        ~candidate[
            "clip_geom"
        ].is_empty

    ].copy()


    if candidate.empty:

        return []


    candidate[
        "area_m2"
    ] = (

        candidate[
            "clip_geom"
        ]
        .area

    )


    grouped = (

        candidate

        .groupby(

            "LULC_2022",

            dropna=False,

        )[
            "area_m2"
        ]

        .sum()

        .reset_index()

    )


    total = float(

        grouped[
            "area_m2"
        ]
        .sum()

    )


    if total <= 0:

        return []


    grouped[
        "area_ha"
    ] = (

        grouped[
            "area_m2"
        ]

        /
        10000.0

    )


    grouped[
        "percent"
    ] = (

        grouped[
            "area_m2"
        ]

        /
        total

        *
        100.0

    )


    grouped = (
        grouped
        .sort_values(

            "percent",

            ascending=False,

        )
    )


    return [

        {

            "name":

                str(
                    row[
                        "LULC_2022"
                    ]
                ),

            "area_ha":

                float(
                    row[
                        "area_ha"
                    ]
                ),

            "percent":

                float(
                    row[
                        "percent"
                    ]
                ),

        }

        for _, row
        in grouped.iterrows()

    ]


# ============================================================
# 15. NEAREST DISTANCE
# ============================================================

def nearest_distance(

    polygon_geojson,

    gdf,

):

    if (
        gdf is None
        or
        gdf.empty
    ):

        return None


    field_geom = (

        geometry_gdf(
            polygon_geojson
        )

        .to_crs(
            gdf.crs
        )

        .geometry
        .iloc[0]

    )


    distances = (

        gdf
        .geometry
        .distance(
            field_geom
        )

    )


    if distances.empty:

        return None


    return float(
        distances.min()
    )


# ============================================================
# 16. AUTOMATIC FARM-POND SITE SELECTION
# ============================================================

def find_pond_site(

    field_geojson,

    minimum_boundary_clearance=
        10.0,

):

    (
        elevation,

        elevation_mask,

        transform,

        raster_crs,

    ) = crop_reference_grid(
        field_geojson
    )


    reference_shape = (
        elevation.shape
    )


    slope = read_to_reference(

        SLOPE_RASTER,

        reference_shape,

        transform,

        raster_crs,

    )


    twi = read_to_reference(

        TWI_RASTER,

        reference_shape,

        transform,

        raster_crs,

    )


    drainage_density = (
        read_to_reference(

            DRAINAGE_DENSITY_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
    )


    flow_accumulation = (
        read_to_reference(

            FLOW_ACCUMULATION_RASTER,

            reference_shape,

            transform,

            raster_crs,

        )
    )


    valid = (

        elevation_mask

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
            drainage_density
        )

    )


    if valid.sum() == 0:

        return None


    # --------------------------------------------------------
    # ABSOLUTE LOWEST POINT
    # --------------------------------------------------------

    elevation_work = np.where(

        valid,

        elevation,

        np.inf,

    )


    lowest_index = (

        np.unravel_index(

            np.argmin(
                elevation_work
            ),

            elevation_work.shape,

        )
    )


    lowest_x, lowest_y = xy(

        transform,

        lowest_index[0],

        lowest_index[1],

        offset="center",

    )


    # --------------------------------------------------------
    # NORMALIZED VARIABLES
    # --------------------------------------------------------

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
        drainage_density,
        valid,
    )


    # --------------------------------------------------------
    # BASE SCORE
    # --------------------------------------------------------

    score = (

        0.45
        *
        (
            1.0
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
            1.0
            -
            slope_norm
        )

        +

        0.10
        *
        drainage_norm

    )


    weight_sum = (
        0.95
    )


    # --------------------------------------------------------
    # FLOW ACCUMULATION
    # --------------------------------------------------------

    flow_valid = (

        valid

        &

        np.isfinite(
            flow_accumulation
        )

    )


    if flow_valid.sum() > 0:

        log_flow = np.log1p(

            np.maximum(
                flow_accumulation,
                0
            )

        )


        flow_norm = normalize_array(

            log_flow,

            flow_valid,

        )


        score = (

            score

            +

            0.05
            *
            flow_norm

        )


        weight_sum += (
            0.05
        )


    score = (

        score
        /
        weight_sum

    )


    # --------------------------------------------------------
    # FARM-POND SLOPE PREFERENCE
    # --------------------------------------------------------

    preferred = (

        valid

        &

        (
            slope
            <=
            8.0
        )

    )


    # fallback if no <= 8 degree cell

    if preferred.sum() == 0:

        preferred = (

            valid

            &

            (
                slope
                <=
                15.0
            )

        )


    if preferred.sum() == 0:

        preferred = valid


    score = np.where(

        preferred,

        score,

        -np.inf,

    )


    # --------------------------------------------------------
    # FIELD GEOMETRY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # LOCAL STREAM UNION
    # --------------------------------------------------------

    stream_union = None


    if (
        streams_utm is not None
        and
        not streams_utm.empty
    ):

        streams_here = (

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


        ids = list(

            streams_here
            .sindex
            .intersection(

                search_area
                .bounds

            )

        )


        if ids:

            subset = (
                streams_here
                .iloc[
                    ids
                ]
                .copy()
            )


            subset = subset[
                subset.intersects(
                    search_area
                )
            ]


            if not subset.empty:

                stream_union = (
                    unary_union(

                        list(
                            subset
                            .geometry
                        )

                    )
                )


    # --------------------------------------------------------
    # SEARCH HIGHEST-SCORING FEASIBLE PIXEL
    # --------------------------------------------------------

    flat_order = (

        np.argsort(
            score.ravel()
        )[::-1]

    )


    best = None


    for flat_index in (
        flat_order[:1000]
    ):

        row, col = (

            np.unravel_index(

                flat_index,

                score.shape,

            )

        )


        if not np.isfinite(
            score[
                row,
                col
            ]
        ):

            continue


        px, py = xy(

            transform,

            row,

            col,

            offset="center",

        )


        point = Point(
            px,
            py
        )


        boundary_distance = float(

            point.distance(

                field_geom
                .boundary

            )

        )


        stream_distance = None


        if stream_union is not None:

            stream_distance = float(

                point.distance(
                    stream_union
                )

            )


        # Candidate must have enough boundary clearance

        if (
            boundary_distance
            <
            minimum_boundary_clearance
        ):

            continue


        # For an off-stream farm pond avoid placing
        # the centre directly in a mapped stream

        if (
            stream_distance
            is not None
            and
            stream_distance
            <
            10.0
        ):

            continue


        best = {

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
                        row,
                        col
                    ]
                ),

            "slope":
                float(
                    slope[
                        row,
                        col
                    ]
                ),

            "twi":
                float(
                    twi[
                        row,
                        col
                    ]
                ),

            "drainage_density":
                float(
                    drainage_density[
                        row,
                        col
                    ]
                ),

            "flow_accumulation":

                float(
                    flow_accumulation[
                        row,
                        col
                    ]
                )

                if np.isfinite(
                    flow_accumulation[
                        row,
                        col
                    ]
                )

                else np.nan,

            "score":
                float(
                    score[
                        row,
                        col
                    ]
                ),

            "boundary_distance":
                boundary_distance,

            "stream_distance":
                stream_distance,

        }


        break


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if best is None:

        finite_indices = (

            np.flatnonzero(

                np.isfinite(
                    score.ravel()
                )

            )

        )


        if len(
            finite_indices
        ) == 0:

            return None


        best_flat = (

            finite_indices[

                np.argmax(

                    score.ravel()[
                        finite_indices
                    ]

                )

            ]

        )


        row, col = (

            np.unravel_index(

                best_flat,

                score.shape,

            )

        )


        px, py = xy(

            transform,

            row,

            col,

            offset="center",

        )


        point = Point(
            px,
            py
        )


        best = {

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
                        row,
                        col
                    ]
                ),

            "slope":
                float(
                    slope[
                        row,
                        col
                    ]
                ),

            "twi":
                float(
                    twi[
                        row,
                        col
                    ]
                ),

            "drainage_density":
                float(
                    drainage_density[
                        row,
                        col
                    ]
                ),

            "flow_accumulation":

                float(
                    flow_accumulation[
                        row,
                        col
                    ]
                )

                if np.isfinite(
                    flow_accumulation[
                        row,
                        col
                    ]
                )

                else np.nan,

            "score":
                float(
                    score[
                        row,
                        col
                    ]
                ),

            "boundary_distance":
                float(

                    point.distance(

                        field_geom
                        .boundary

                    )

                ),

            "stream_distance":

                float(
                    point.distance(
                        stream_union
                    )
                )

                if stream_union
                is not None

                else None,

        }


    # --------------------------------------------------------
    # CONVERT TO LAT/LONG
    # --------------------------------------------------------

    transformer = (

        Transformer
        .from_crs(

            raster_crs,

            "EPSG:4326",

            always_xy=True,

        )

    )


    longitude, latitude = (

        transformer.transform(

            best[
                "x"
            ],

            best[
                "y"
            ],

        )

    )


    low_lon, low_lat = (

        transformer.transform(

            lowest_x,

            lowest_y,

        )

    )


    best[
        "longitude"
    ] = float(
        longitude
    )


    best[
        "latitude"
    ] = float(
        latitude
    )


    lowest = {

        "x":
            float(
                lowest_x
            ),

        "y":
            float(
                lowest_y
            ),

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
                    lowest_index
                ]
            ),

    }


    return {

        "recommended":
            best,

        "absolute_lowest":
            lowest,

    }


# ============================================================
# 17. FARM-POND GEOMETRY
# ============================================================

def pond_geometry(

    bottom_width,

    water_depth,

    freeboard,

    side_slope,

    length_width_ratio,

):

    bottom_length = (

        length_width_ratio
        *
        bottom_width

    )


    water_top_width = (

        bottom_width

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


    excavation_depth = (

        water_depth
        +
        freeboard

    )


    excavation_top_width = (

        bottom_width

        +

        2
        *
        side_slope
        *
        excavation_depth

    )


    excavation_top_length = (

        bottom_length

        +

        2
        *
        side_slope
        *
        excavation_depth

    )


    bottom_area = (

        bottom_length
        *
        bottom_width

    )


    water_top_area = (

        water_top_length
        *
        water_top_width

    )


    volume = (

        water_depth
        /
        3.0

        *

        (

            bottom_area

            +

            water_top_area

            +

            math.sqrt(

                bottom_area
                *
                water_top_area

            )

        )

    )


    footprint_area = (

        excavation_top_length
        *
        excavation_top_width

    )


    slant = math.sqrt(

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
        2.0

        *
        slant

    )


    short_side_area = (

        (
            bottom_width
            +
            water_top_width
        )

        /
        2.0

        *
        slant

    )


    liner_wetted_area = (

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

        "bottom_width":
            float(
                bottom_width
            ),

        "bottom_length":
            float(
                bottom_length
            ),

        "water_top_width":
            float(
                water_top_width
            ),

        "water_top_length":
            float(
                water_top_length
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

        "excavation_top_width":
            float(
                excavation_top_width
            ),

        "excavation_top_length":
            float(
                excavation_top_length
            ),

        "volume":
            float(
                volume
            ),

        "footprint_area":
            float(
                footprint_area
            ),

        "liner_wetted_area":
            float(
                liner_wetted_area
            ),

        "liner_procurement_area":
            float(
                liner_wetted_area
                *
                1.10
            ),

    }


# ============================================================
# 18. DESIGN POND TO TARGET STORAGE
# ============================================================

def design_farm_pond(

    available_water_m3,

    field_area_ha_value,

    storage_fraction,

    maximum_field_fraction,

    water_depth,

    freeboard,

    side_slope,

    length_width_ratio,

):

    field_area_m2 = (

        field_area_ha_value
        *
        10000.0

    )


    requested_storage = (

        available_water_m3
        *
        storage_fraction

    )


    maximum_footprint = (

        field_area_m2
        *
        maximum_field_fraction

    )


    # --------------------------------------------------------
    # FIND MAXIMUM WIDTH ALLOWED BY LAND AREA
    # --------------------------------------------------------

    low = 0.5

    high = 150.0


    for _ in range(
        80
    ):

        middle = (
            low
            +
            high
        ) / 2.0


        geom = pond_geometry(

            middle,

            water_depth,

            freeboard,

            side_slope,

            length_width_ratio,

        )


        if (
            geom[
                "footprint_area"
            ]
            >
            maximum_footprint
        ):

            high = middle

        else:

            low = middle


    maximum_geom = pond_geometry(

        low,

        water_depth,

        freeboard,

        side_slope,

        length_width_ratio,

    )


    target_storage = min(

        requested_storage,

        maximum_geom[
            "volume"
        ],

    )


    # --------------------------------------------------------
    # SOLVE WIDTH FOR TARGET STORAGE
    # --------------------------------------------------------

    low_width = 0.5

    high_width = max(
        low,
        0.5,
    )


    for _ in range(
        80
    ):

        middle = (

            low_width

            +

            high_width

        ) / 2.0


        geom = pond_geometry(

            middle,

            water_depth,

            freeboard,

            side_slope,

            length_width_ratio,

        )


        if (
            geom[
                "volume"
            ]
            >
            target_storage
        ):

            high_width = middle

        else:

            low_width = middle


    result = pond_geometry(

        low_width,

        water_depth,

        freeboard,

        side_slope,

        length_width_ratio,

    )


    result[
        "requested_storage"
    ] = float(
        requested_storage
    )


    result[
        "target_storage"
    ] = float(
        target_storage
    )


    result[
        "maximum_storage_by_land"
    ] = float(

        maximum_geom[
            "volume"
        ]

    )


    result[
        "field_occupied_percent"
    ] = float(

        result[
            "footprint_area"
        ]

        /
        field_area_m2

        *
        100.0

    )


    result[
        "remaining_field_ha"
    ] = float(

        max(

            field_area_m2

            -

            result[
                "footprint_area"
            ],

            0,

        )

        /
        10000.0

    )


    return result


# ============================================================
# 19. THEMATIC RANK DESCRIPTIONS
# ============================================================

def soil_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    return {

        1:
            "Clay / Clayey / Clay loam",

        2:
            "Silt clay loam / Silt loam",

        3:
            "Sandy clay loam",

        4:
            "Miscellaneous soil",

        5:
            "Loam",

    }.get(

        int(
            round(
                rank
            )
        ),

        "Unknown",

    )


def geology_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    return {

        5:
            "Quaternary",

        3:
            (
                "Umsning Schist / "
                "Assam-Meghalaya Gneissic"
            ),

        2:
            (
                "Shillong Group / Kyrdem"
            ),

    }.get(

        int(
            round(
                rank
            )
        ),

        "Other / unmatched",

    )


def geomorph_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    return {

        5:
            "Valley",

        4:
            (
                "Pediment-Pediplain / "
                "River / Pond"
            ),

        3:
            (
                "Bench / Moderately "
                "dissected upper plateau"
            ),

        2:
            (
                "Highly or moderately "
                "dissected hills/plateau"
            ),

        1:
            (
                "Scarp / Ridge / "
                "Low dissected upper plateau"
            ),

    }.get(

        int(
            round(
                rank
            )
        ),

        "Unknown",

    )


# ============================================================
# 20. SEEPAGE SCREENING
# ============================================================

def seepage_assessment(

    soil_rank,

    geology_rank,

    geomorph_rank,

    lineament_density,

):

    score = 0

    reasons = []


    # --------------------------------------------------------
    # SOIL
    # Groundwater-oriented rank is reinterpreted for storage
    # --------------------------------------------------------

    if np.isfinite(
        soil_rank
    ):

        soil = int(
            round(
                soil_rank
            )
        )


        if soil == 1:

            reasons.append(

                "Clayey/clay-loam soil context suggests comparatively favourable water retention."

            )


        elif soil == 2:

            score += 1

            reasons.append(

                "Silty soil may permit moderate seepage and should be checked in the field."

            )


        elif soil == 3:

            score += 2

            reasons.append(

                "Sandy clay loam indicates increased seepage potential."

            )


        elif soil == 5:

            score += 2

            reasons.append(

                "Loam may permit greater seepage than clayey soil."

            )


        else:

            score += 1

            reasons.append(

                "Soil conditions require field permeability verification."

            )


    # --------------------------------------------------------
    # LINEAMENT DENSITY
    # Screening thresholds based on observed distribution
    # --------------------------------------------------------

    if np.isfinite(
        lineament_density
    ):

        if (
            lineament_density
            >=
            0.55
        ):

            score += 2

            reasons.append(

                "High lineament density indicates greater fracture-related seepage concern."

            )


        elif (
            lineament_density
            >=
            0.19
        ):

            score += 1

            reasons.append(

                "Moderate lineament density indicates some fracture-related seepage concern."

            )


        else:

            reasons.append(

                "Low local lineament density reduces fracture-related seepage concern."

            )


    # --------------------------------------------------------
    # GEOLOGY
    # --------------------------------------------------------

    if np.isfinite(
        geology_rank
    ):

        geology = int(
            round(
                geology_rank
            )
        )


        if geology == 5:

            score += 1

            reasons.append(

                "Quaternary material may be heterogeneous; permeability should be verified."

            )


        elif geology in [
            2,
            3,
        ]:

            reasons.append(

                "Bedrock setting requires checking weathering and fractures at excavation depth."

            )


    # --------------------------------------------------------
    # GEOMORPHOLOGY
    # --------------------------------------------------------

    if np.isfinite(
        geomorph_rank
    ):

        geomorph = int(
            round(
                geomorph_rank
            )
        )


        if geomorph in [
            1,
            2,
        ]:

            score += 1

            reasons.append(

                "Dissected/ridge-scarp terrain requires additional stability and seepage verification."

            )


    # --------------------------------------------------------
    # FINAL CLASS
    # --------------------------------------------------------

    if score >= 4:

        risk = "HIGH"

        lining = (

            "Impermeable lining is strongly recommended. "
            "Use a properly prepared and compacted subgrade "
            "with an appropriate geomembrane or approved "
            "pond-lining system after field permeability "
            "and geotechnical verification."

        )


    elif score >= 2:

        risk = "MODERATE"

        lining = (

            "Lining should be considered. "
            "A compacted clay blanket may be suitable where "
            "good clay is available. Use an approved geomembrane "
            "where reliable storage retention is essential."

        )


    else:

        risk = "LOW"

        lining = (

            "Compacted natural soil treatment may be adequate, "
            "but a field permeability test is still required "
            "before deciding that synthetic lining is unnecessary."

        )


    return {

        "risk":
            risk,

        "score":
            score,

        "reasons":
            reasons,

        "lining_recommendation":
            lining,

    }


# ============================================================
# 21. FARMER INFORMATION
# ============================================================

with st.expander(
    "👨‍🌾 Farmer Details"
):

    farmer_name = st.text_input(
        "Farmer Name"
    )

    village = st.text_input(
        "Village"
    )


# ============================================================
# 22. DRAW FARMER FIELD
# ============================================================

st.header(
    "1️⃣ Demarcate Farmer Field"
)


st.info(

    "Use the polygon tool ⬠ and trace the actual farmer field. "
    "On mobile, tap each corner and then tap the first point again "
    "to close the polygon."

)


field_map = folium.Map(

    location=
        MAP_CENTER,

    zoom_start=
        MAP_ZOOM,

    tiles=None,

    prefer_canvas=True,

    control_scale=True,

)


folium.TileLayer(

    tiles=(

        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}"

    ),

    attr="Esri",

    name="Satellite",

    show=True,

).add_to(
    field_map
)


folium.TileLayer(

    "OpenStreetMap",

    name="Street Map",

    show=False,

).add_to(
    field_map
)


# District boundary

if boundary_wgs is not None:

    folium.GeoJson(

        boundary_wgs.__geo_interface__,

        name=
            "Ri Bhoi District",

        style_function=lambda x: {

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


# Existing drawn field

if st.session_state[
    "field_geometry"
]:

    folium.GeoJson(

        st.session_state[
            "field_geometry"
        ],

        name="Farmer Field",

        style_function=lambda x: {

            "color":
                "#00C853",

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


Draw(

    export=False,

    position="topleft",

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
                    "#00C853",

                "weight":
                    5,

                "fillColor":
                    "#69F0AE",

                "fillOpacity":
                    0.30,

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


Fullscreen().add_to(
    field_map
)


folium.LayerControl(
    collapsed=True
).add_to(
    field_map
)


field_output = st_folium(

    field_map,

    height=520,

    use_container_width=True,

    key="farmer_field_map",

    returned_objects=[

        "all_drawings",

        "last_active_drawing",

    ],

)


new_field = get_last_polygon(
    field_output
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

    st.session_state[
        "current_lulc"
    ] = None

    st.rerun()


field = (
    st.session_state[
        "field_geometry"
    ]
)


# ============================================================
# 23. FIELD INFORMATION
# ============================================================

if field:

    area_ha = field_area_ha(
        field
    )


    lulc_summary = (
        mapped_lulc_summary(
            field
        )
    )


    mapped_lulc = (

        lulc_summary[0][
            "name"
        ]

        if lulc_summary

        else "Unknown"

    )


    st.success(

        f"✅ Farmer field captured: "
        f"{area_ha:.3f} ha"

    )


    c1, c2 = (
        st.columns(2)
    )


    c1.metric(

        "Field Area",

        f"{area_ha:.3f} ha",

    )


    c2.metric(

        "GIS-Mapped LULC",

        mapped_lulc,

    )


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


    if (
        st.session_state[
            "current_lulc"
        ]
        in LAND_USE_OPTIONS
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
        in LAND_USE_OPTIONS
    ):

        default_index = (

            LAND_USE_OPTIONS
            .index(
                mapped_lulc
            )

        )


    else:

        default_index = 0


    st.markdown(
        "### 🌾 Confirm Present Land Use"
    )


    current_lulc = (
        st.selectbox(

            "Current land use observed by farmer/field team",

            LAND_USE_OPTIONS,

            index=
                default_index,

        )
    )


    st.session_state[
        "current_lulc"
    ] = current_lulc


    if (
        current_lulc
        !=
        mapped_lulc
    ):

        st.warning(

            f"GIS-mapped LULC = **{mapped_lulc}**. "
            f"Present field observation = **{current_lulc}**. "
            "The DSS will use the confirmed present land use "
            "for planning while retaining the GIS class."

        )


    with st.expander(
        "GIS LULC composition inside farmer field"
    ):

        if lulc_summary:

            df_lulc = pd.DataFrame(
                lulc_summary
            )


            df_lulc[
                "area_ha"
            ] = (

                df_lulc[
                    "area_ha"
                ]
                .round(3)

            )


            df_lulc[
                "percent"
            ] = (

                df_lulc[
                    "percent"
                ]
                .round(1)

            )


            st.dataframe(

                df_lulc,

                use_container_width=True,

                hide_index=True,

            )


        else:

            st.info(
                "Detailed LULC layer unavailable."
            )


# ============================================================
# 24. DESIGN SETTINGS
# ============================================================

if field:

    st.markdown("---")


    st.header(
        "2️⃣ Farm Pond Planning Settings"
    )


    with st.expander(

        "Adjust preliminary design assumptions",

        expanded=False,

    ):

        storage_fraction_percent = (
            st.slider(

                "One-time storage as % of 75% dependable annual runoff",

                min_value=10,

                max_value=60,

                value=30,

                step=5,

            )
        )


        maximum_land_percent = (
            st.slider(

                "Maximum farmer land occupied by pond (%)",

                min_value=5,

                max_value=20,

                value=10,

                step=1,

            )
        )


        water_depth = (
            st.slider(

                "Planning water depth (m)",

                min_value=1.5,

                max_value=3.5,

                value=2.5,

                step=0.25,

            )
        )


        freeboard = (
            st.slider(

                "Freeboard (m)",

                min_value=0.30,

                max_value=0.75,

                value=0.50,

                step=0.05,

            )
        )


        side_slope = (
            st.slider(

                "Planning side slope (H:1V)",

                min_value=1.0,

                max_value=2.5,

                value=1.5,

                step=0.25,

            )
        )


        length_width_ratio = (
            st.slider(

                "Bottom length : breadth ratio",

                min_value=1.0,

                max_value=2.0,

                value=1.5,

                step=0.1,

            )
        )


    st.caption(

        "These are planning assumptions, not universal engineering standards. "
        "Final dimensions require field survey and engineering design."

    )


# ============================================================
# 25. RUN ASSESSMENT
# ============================================================

if field:

    st.markdown("---")


    st.header(
        "3️⃣ Automatic Pond Site & Preliminary Design"
    )


    if st.button(

        "💧 FIND FARM POND LOCATION & DESIGN",

        type="primary",

        use_container_width=True,

    ):

        with st.spinner(

            "Analysing rainfall, runoff, terrain, "
            "pond site and preliminary dimensions..."

        ):

            area_ha = (
                field_area_ha(
                    field
                )
            )


            rainfall = (
                safe_raster_stats(
                    RAINFALL_RASTER,
                    field,
                )["mean"]
            )


            mean_runoff = (
                safe_raster_stats(
                    MEAN_RUNOFF_RASTER,
                    field,
                )["mean"]
            )


            dependable_runoff = (
                safe_raster_stats(
                    DEPENDABLE_RUNOFF_RASTER,
                    field,
                )["mean"]
            )


            runoff_coefficient = (
                safe_raster_stats(
                    RUNOFF_COEFFICIENT_RASTER,
                    field,
                )["mean"]
            )


            cn = (
                safe_raster_stats(
                    CN_RASTER,
                    field,
                )["mean"]
            )


            mean_runoff_volume = (

                mean_runoff
                *
                area_ha
                *
                10

                if np.isfinite(
                    mean_runoff
                )

                else np.nan

            )


            dependable_volume = (

                dependable_runoff
                *
                area_ha
                *
                10

                if np.isfinite(
                    dependable_runoff
                )

                else np.nan

            )


            available_water = (

                dependable_volume

                if np.isfinite(
                    dependable_volume
                )

                else
                    mean_runoff_volume

            )


            if (
                not np.isfinite(
                    available_water
                )
                or
                available_water <= 0
            ):

                st.error(

                    "Runoff data are unavailable or invalid for this field."

                )

                st.stop()


            # --------------------------------------------
            # PRELIMINARY POND DESIGN
            # --------------------------------------------

            design = (
                design_farm_pond(

                    available_water_m3=
                        available_water,

                    field_area_ha_value=
                        area_ha,

                    storage_fraction=
                        storage_fraction_percent
                        /
                        100.0,

                    maximum_field_fraction=
                        maximum_land_percent
                        /
                        100.0,

                    water_depth=
                        water_depth,

                    freeboard=
                        freeboard,

                    side_slope=
                        side_slope,

                    length_width_ratio=
                        length_width_ratio,

                )
            )


            # Required boundary clearance
            # based partly on pond footprint

            half_diagonal = (

                0.5

                *

                math.sqrt(

                    design[
                        "excavation_top_length"
                    ] ** 2

                    +

                    design[
                        "excavation_top_width"
                    ] ** 2

                )

            )


            clearance = max(

                10.0,

                half_diagonal
                +
                2.0,

            )


            sites = (
                find_pond_site(

                    field,

                    minimum_boundary_clearance=
                        clearance,

                )
            )


            if sites is None:

                st.error(

                    "No valid terrain cells were found inside this field."

                )

                st.stop()


            site = (
                sites[
                    "recommended"
                ]
            )


            lowest = (
                sites[
                    "absolute_lowest"
                ]
            )


            # --------------------------------------------
            # THEMATIC VALUES AT PROPOSED SITE
            # --------------------------------------------

            geology_rank = (
                sample_raster_at_xy(

                    GEOLOGY_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            geomorph_rank = (
                sample_raster_at_xy(

                    GEOMORPH_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            soil_rank = (
                sample_raster_at_xy(

                    SOIL_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            lineament_density = (
                sample_raster_at_xy(

                    LINEAMENT_DENSITY_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            seepage = (
                seepage_assessment(

                    soil_rank,

                    geology_rank,

                    geomorph_rank,

                    lineament_density,

                )
            )


            st.session_state[
                "assessment"
            ] = {

                "farmer_name":
                    farmer_name,

                "village":
                    village,

                "field_area_ha":
                    area_ha,

                "mapped_lulc":
                    mapped_lulc,

                "current_lulc":
                    current_lulc,

                "rainfall":
                    rainfall,

                "mean_runoff":
                    mean_runoff,

                "dependable_runoff":
                    dependable_runoff,

                "runoff_coefficient":
                    runoff_coefficient,

                "cn":
                    cn,

                "mean_runoff_volume":
                    mean_runoff_volume,

                "dependable_volume":
                    dependable_volume,

                "design":
                    design,

                "site":
                    site,

                "lowest":
                    lowest,

                "geology_rank":
                    geology_rank,

                "geomorph_rank":
                    geomorph_rank,

                "soil_rank":
                    soil_rank,

                "lineament_density":
                    lineament_density,

                "seepage":
                    seepage,

                "nearest_stream":
                    nearest_distance(
                        field,
                        streams_utm,
                    ),

                "nearest_mgnrega":
                    nearest_distance(
                        field,
                        mgnrega_utm,
                    ),

                "nearest_final94":
                    nearest_distance(
                        field,
                        final94_utm,
                    ),

                "nearest_published":
                    nearest_distance(
                        field,
                        published_utm,
                    ),

            }


        st.rerun()


# ============================================================
# 26. DISPLAY RESULTS
# ============================================================

result = (
    st.session_state[
        "assessment"
    ]
)


if result:

    site = result[
        "site"
    ]


    lowest = result[
        "lowest"
    ]


    design = result[
        "design"
    ]


    seepage = result[
        "seepage"
    ]


    st.markdown("---")


    st.header(
        "✅ Farmer RWH Planning Result"
    )


    agriculture = any(

        name
        in
        result[
            "current_lulc"
        ].lower()

        for name in [

            "crop",

            "plantation",

            "shifting",

        ]

    )


    if agriculture:

        st.success(

            "**Primary screened structure: Farm Pond.** "
            "The site was selected using low elevation, "
            "high TWI, gentle slope, drainage context and "
            "flow accumulation with boundary and stream checks."

        )


    else:

        st.warning(

            "The confirmed current land use is not clearly agricultural. "
            "The farm-pond result should therefore be treated as a "
            "screening scenario requiring further justification."

        )


    # ========================================================
    # WATER AVAILABILITY
    # ========================================================

    st.subheader(
        "🌧️ Water Availability"
    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Field Area",

        f"{result['field_area_ha']:.3f} ha",

    )


    c2.metric(

        "Spatial Annual Rainfall",

        (
            f"{result['rainfall']:.0f} mm/year"

            if np.isfinite(
                result[
                    "rainfall"
                ]
            )

            else "No data"
        ),

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Mean Annual Runoff",

        (
            f"{result['mean_runoff']:.0f} mm"

            if np.isfinite(
                result[
                    "mean_runoff"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "75% Dependable Runoff",

        (
            f"{result['dependable_runoff']:.0f} mm"

            if np.isfinite(
                result[
                    "dependable_runoff"
                ]
            )

            else "No data"
        ),

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Annual Runoff Volume",

        (
            f"{result['mean_runoff_volume']:,.0f} m³/year"

            if np.isfinite(
                result[
                    "mean_runoff_volume"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "75% Dependable Volume",

        (
            f"{result['dependable_volume']:,.0f} m³/year"

            if np.isfinite(
                result[
                    "dependable_volume"
                ]
            )

            else "No data"
        ),

    )


    st.caption(

        "Rainfall retains the spatial information content of "
        "the original rainfall dataset. Resampling onto a 30-m "
        "GIS grid does not create true 30-m rainfall observations."

    )


    # ========================================================
    # SITE
    # ========================================================

    st.subheader(
        "📍 Recommended Farm Pond Location"
    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Latitude",

        f"{site['latitude']:.6f}",

    )


    c2.metric(

        "Longitude",

        f"{site['longitude']:.6f}",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Elevation",

        f"{site['elevation']:.1f} m",

    )


    c2.metric(

        "Local Slope",

        f"{site['slope']:.2f}°",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "TWI",

        f"{site['twi']:.2f}",

    )


    c2.metric(

        "Boundary Clearance",

        f"{site['boundary_distance']:.1f} m",

    )


    if (
        site[
            "stream_distance"
        ]
        is not None
    ):

        st.metric(

            "Distance from Proposed Pond Centre to Mapped Stream",

            f"{site['stream_distance']:.1f} m",

        )


    st.caption(

        "Red = recommended practical farm-pond centre. "
        "Orange = absolute lowest DEM pixel. "
        "They may differ when the absolute lowest pixel is "
        "too close to the field boundary or mapped drainage."

    )


    # ========================================================
    # RESULT MAP
    # ========================================================

    result_map = folium.Map(

        location=[

            site[
                "latitude"
            ],

            site[
                "longitude"
            ],

        ],

        zoom_start=17,

        tiles=None,

        control_scale=True,

    )


    folium.TileLayer(

        tiles=(

            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/"
            "tile/{z}/{y}/{x}"

        ),

        attr="Esri",

        name="Satellite",

        show=True,

    ).add_to(
        result_map
    )


    folium.TileLayer(

        "OpenStreetMap",

        name="Street Map",

        show=False,

    ).add_to(
        result_map
    )


    # Farmer field

    folium.GeoJson(

        field,

        name="Farmer Field",

        style_function=lambda x: {

            "color":
                "#00C853",

            "weight":
                5,

            "fillColor":
                "#69F0AE",

            "fillOpacity":
                0.20,

        },

    ).add_to(
        result_map
    )


    # --------------------------------------------------------
    # APPROXIMATE POND FOOTPRINT
    # --------------------------------------------------------

    pond_box_utm = box(

        site[
            "x"
        ]
        -
        design[
            "excavation_top_length"
        ]
        /
        2,

        site[
            "y"
        ]
        -
        design[
            "excavation_top_width"
        ]
        /
        2,

        site[
            "x"
        ]
        +
        design[
            "excavation_top_length"
        ]
        /
        2,

        site[
            "y"
        ]
        +
        design[
            "excavation_top_width"
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
            "Approximate Pond Footprint",

        style_function=lambda x: {

            "color":
                "#D50000",

            "weight":
                3,

            "fillColor":
                "#FF5252",

            "fillOpacity":
                0.28,

        },

        tooltip=
            "Approximate preliminary pond footprint",

    ).add_to(
        result_map
    )


    # Recommended centre

    folium.Marker(

        location=[

            site[
                "latitude"
            ],

            site[
                "longitude"
            ],

        ],

        tooltip=
            "Recommended Farm Pond Site",

        popup=(

            "<b>Recommended Farm Pond</b>"
            "<br>"

            f"Elevation: "
            f"{site['elevation']:.1f} m"

            "<br>"

            f"Slope: "
            f"{site['slope']:.2f}°"

            "<br>"

            f"TWI: "
            f"{site['twi']:.2f}"

            "<br>"

            f"Storage: "
            f"{design['target_storage']:.0f} m³"

        ),

        icon=folium.Icon(

            color="red",

            icon="tint",

        ),

    ).add_to(
        result_map
    )


    # Absolute lowest point

    folium.Marker(

        location=[

            lowest[
                "latitude"
            ],

            lowest[
                "longitude"
            ],

        ],

        tooltip=
            "Absolute Lowest DEM Point",

        popup=(

            "<b>Absolute Lowest Point</b>"
            "<br>"

            f"Elevation: "
            f"{lowest['elevation']:.1f} m"

        ),

        icon=folium.Icon(

            color="orange",

            icon="arrow-down",

        ),

    ).add_to(
        result_map
    )


    Fullscreen().add_to(
        result_map
    )


    folium.LayerControl(
        collapsed=True
    ).add_to(
        result_map
    )


    st_folium(

        result_map,

        height=520,

        use_container_width=True,

        key="result_site_map",

    )


    # ========================================================
    # DESIGN
    # ========================================================

    st.subheader(
        "📐 Preliminary Farm Pond Design"
    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Proposed Storage",

        f"{design['target_storage']:,.0f} m³",

    )


    c2.metric(

        "Water Depth",

        f"{design['water_depth']:.2f} m",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Bottom Length",

        f"{design['bottom_length']:.1f} m",

    )


    c2.metric(

        "Bottom Breadth",

        f"{design['bottom_width']:.1f} m",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Excavation Top Length",

        f"{design['excavation_top_length']:.1f} m",

    )


    c2.metric(

        "Excavation Top Breadth",

        f"{design['excavation_top_width']:.1f} m",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Excavation Depth",

        f"{design['excavation_depth']:.2f} m",

    )


    c2.metric(

        "Freeboard",

        f"{design['freeboard']:.2f} m",

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Pond Footprint",

        f"{design['footprint_area']:,.0f} m²",

    )


    c2.metric(

        "Farm Area Occupied",

        f"{design['field_occupied_percent']:.1f}%",

    )


    st.metric(

        "Remaining Farmer Land",

        f"{design['remaining_field_ha']:.3f} ha",

    )


    if (
        design[
            "requested_storage"
        ]
        >
        design[
            "target_storage"
        ]
        +
        0.01
    ):

        st.warning(

            "Requested storage exceeded the selected maximum "
            "land-occupation limit. The DSS therefore reduced "
            "the preliminary pond capacity."

        )


    # ========================================================
    # SEEPAGE
    # ========================================================

    st.subheader(
        "🪨 Seepage & Pond Lining Screening"
    )


    if seepage[
        "risk"
    ] == "HIGH":

        st.error(
            "Seepage Risk: HIGH"
        )


    elif seepage[
        "risk"
    ] == "MODERATE":

        st.warning(
            "Seepage Risk: MODERATE"
        )


    else:

        st.success(
            "Seepage Risk: LOW"
        )


    c1, c2 = st.columns(2)


    c1.metric(

        "Soil Context",

        soil_rank_description(
            result[
                "soil_rank"
            ]
        ),

    )


    c2.metric(

        "Geology Context",

        geology_rank_description(
            result[
                "geology_rank"
            ]
        ),

    )


    st.write(

        "**Geomorphology:**",

        geomorph_rank_description(
            result[
                "geomorph_rank"
            ]
        ),

    )


    st.write(

        "**Lineament Density:**",

        (
            f"{result['lineament_density']:.3f}"

            if np.isfinite(
                result[
                    "lineament_density"
                ]
            )

            else "No data"
        ),

    )


    st.markdown(
        "#### Why this seepage class?"
    )


    for reason in seepage[
        "reasons"
    ]:

        st.write(
            "•",
            reason
        )


    st.markdown(
        "#### Lining Recommendation"
    )


    st.info(

        seepage[
            "lining_recommendation"
        ]

    )


    c1, c2 = st.columns(2)


    c1.metric(

        "Estimated Wetted Lining Area",

        f"{design['liner_wetted_area']:,.0f} m²",

    )


    c2.metric(

        "Liner Procurement Area (+10%)",

        f"{design['liner_procurement_area']:,.0f} m²",

    )


    # ========================================================
    # TECHNICAL INFORMATION
    # ========================================================

    with st.expander(
        "🔬 Technical Site Information"
    ):

        st.write(
            "**Farmer:**",
            result[
                "farmer_name"
            ]
            or
            "Not entered",
        )


        st.write(
            "**Village:**",
            result[
                "village"
            ]
            or
            "Not entered",
        )


        st.write(
            "**Current land use:**",
            result[
                "current_lulc"
            ],
        )


        st.write(
            "**GIS-mapped LULC:**",
            result[
                "mapped_lulc"
            ],
        )


        st.write(

            "**CN-II:**",

            (
                round(
                    result[
                        "cn"
                    ],
                    1,
                )

                if np.isfinite(
                    result[
                        "cn"
                    ]
                )

                else "No data"
            ),

        )


        st.write(

            "**Runoff coefficient:**",

            (
                round(
                    result[
                        "runoff_coefficient"
                    ],
                    3,
                )

                if np.isfinite(
                    result[
                        "runoff_coefficient"
                    ]
                )

                else "No data"
            ),

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

                if np.isfinite(
                    site[
                        "flow_accumulation"
                    ]
                )

                else "No data"
            ),

        )


        st.write(

            "**Nearest mapped stream to farmer field:**",

            (
                f"{result['nearest_stream']:.0f} m"

                if result[
                    "nearest_stream"
                ]
                is not None

                else "No data"
            ),

        )


        st.write(

            "**Nearest MGNREGA RWH structure:**",

            (
                f"{result['nearest_mgnrega']:.0f} m"

                if result[
                    "nearest_mgnrega"
                ]
                is not None

                else "Layer unavailable"
            ),

        )


        st.write(

            "**Nearest Final-94 candidate:**",

            (
                f"{result['nearest_final94']:.0f} m"

                if result[
                    "nearest_final94"
                ]
                is not None

                else "Layer unavailable"
            ),

        )


        st.write(

            "**Nearest published RWH site:**",

            (
                f"{result['nearest_published']:.0f} m"

                if result[
                    "nearest_published"
                ]
                is not None

                else "Layer unavailable"
            ),

        )


    # ========================================================
    # MANAGEMENT GUIDANCE
    # ========================================================

    st.subheader(
        "🌱 Construction & Management Guidance"
    )


    st.markdown(
        """
<div class="card green-card">

<b>Farm Pond Management</b>
<br><br>

• Verify the proposed site by field survey before excavation.<br>
• Provide an inlet sediment/silt trap.<br>
• Do not obstruct the natural stream for an off-stream pond.<br>
• Provide safe overflow/spill disposal for excess monsoon runoff.<br>
• Stabilize bunds and exposed soil with suitable vegetation.<br>
• Desilt the inlet and pond periodically.<br>
• Protect geomembrane lining from puncture during installation and use.<br>
• Use stored water for protective/supplemental irrigation where appropriate.<br>
• Verify soil permeability before finalizing the lining system.<br>
• Confirm foundation and embankment stability before construction.<br>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # DOWNLOAD SUMMARY CSV
    # ========================================================

    summary = {

        "Farmer":
            result[
                "farmer_name"
            ],

        "Village":
            result[
                "village"
            ],

        "Field_Area_ha":
            result[
                "field_area_ha"
            ],

        "Current_LULC":
            result[
                "current_lulc"
            ],

        "GIS_LULC":
            result[
                "mapped_lulc"
            ],

        "Rainfall_mm_year":
            result[
                "rainfall"
            ],

        "Mean_Runoff_mm":
            result[
                "mean_runoff"
            ],

        "Dependable_Runoff_mm":
            result[
                "dependable_runoff"
            ],

        "Dependable_Volume_m3":
            result[
                "dependable_volume"
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
            design[
                "target_storage"
            ],

        "Bottom_Length_m":
            design[
                "bottom_length"
            ],

        "Bottom_Breadth_m":
            design[
                "bottom_width"
            ],

        "Water_Depth_m":
            design[
                "water_depth"
            ],

        "Freeboard_m":
            design[
                "freeboard"
            ],

        "Top_Length_m":
            design[
                "excavation_top_length"
            ],

        "Top_Breadth_m":
            design[
                "excavation_top_width"
            ],

        "Footprint_m2":
            design[
                "footprint_area"
            ],

        "Field_Occupied_percent":
            design[
                "field_occupied_percent"
            ],

        "Remaining_Field_ha":
            design[
                "remaining_field_ha"
            ],

        "Seepage_Risk":
            seepage[
                "risk"
            ],

        "Liner_Wetted_Area_m2":
            design[
                "liner_wetted_area"
            ],

        "Liner_Procurement_Area_m2":
            design[
                "liner_procurement_area"
            ],

    }


    csv_data = (

        pd.DataFrame(
            [
                summary
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

        "⬇️ Download Assessment CSV",

        data=
            csv_data,

        file_name=
            "RiBhoi_RWH_FarmPond_Assessment.csv",

        mime=
            "text/csv",

        use_container_width=True,

    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
This DSS provides preliminary GIS-based planning and screening.

The proposed coordinates identify a **30-m GIS candidate zone**, not a surveyed construction peg. Final pond location must be checked on the ground for micro-topography, ownership boundary, access, natural drainage, foundation condition, utilities and environmental constraints.

The pond dimensions are preliminary planning dimensions. Final excavation, embankment, inlet, outlet, spillway, freeboard, lining and structural design require field survey and engineering verification.
"""
    )


# ============================================================
# 27. RESET
# ============================================================

st.markdown("---")


if st.button(

    "🔄 Start New Farmer Assessment",

    use_container_width=True,

):

    reset_assessment()

    st.rerun()
