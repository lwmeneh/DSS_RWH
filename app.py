# ============================================================
# RI BHOI RAINWATER HARVESTING GIS DSS
# ICAR RESEARCH COMPLEX FOR NEH REGION
# UMIAM, MEGHALAYA
#
# FARMER FIELD -> AUTOMATIC FARM-POND SITE ->
# WATER AVAILABILITY -> PRELIMINARY DESIGN ->
# SEEPAGE/LINING GUIDANCE
# ============================================================

from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.transform import xy

import streamlit as st
import folium

from shapely.geometry import (
    shape,
    mapping,
    Point,
)

from pyproj import Transformer

from folium.plugins import (
    Draw,
    Fullscreen,
    MeasureControl,
)

from streamlit_folium import st_folium


# ============================================================
# 1. PAGE SETTINGS
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
CACHE_DIR = BASE_DIR / "data" / "cache"


# ----------------------------
# 10-factor extracted rasters
# ----------------------------

GEOLOGY_RASTER = CACHE_DIR / "01_Geology.tif"
GEOMORPH_RASTER = CACHE_DIR / "02_Geomorphology.tif"
SLOPE_RASTER = CACHE_DIR / "03_Slope.tif"
RAINFALL_RASTER = CACHE_DIR / "04_Rainfall.tif"
ELEVATION_RASTER = CACHE_DIR / "05_Elevation.tif"
DRAINAGE_DENSITY_RASTER = CACHE_DIR / "06_DrainageDensity.tif"
LULC_RANK_RASTER = CACHE_DIR / "07_LULC.tif"
LINEAMENT_DENSITY_RASTER = CACHE_DIR / "08_LineamentDensity.tif"
SOIL_RASTER = CACHE_DIR / "09_Soil.tif"
TWI_RASTER = CACHE_DIR / "10_TWI.tif"


# ----------------------------
# Hydrology
# ----------------------------

FLOW_ACCUMULATION_RASTER = (
    CACHE_DIR / "flow_accumulation.tif"
)

CN_RASTER = (
    CACHE_DIR / "CN_II.tif"
)

MEAN_RUNOFF_RASTER = (
    CACHE_DIR / "mean_runoff_mm.tif"
)

DEPENDABLE_RUNOFF_RASTER = (
    CACHE_DIR / "dependable_runoff_mm.tif"
)

RUNOFF_COEFFICIENT_RASTER = (
    CACHE_DIR / "runoff_coefficient.tif"
)


# ----------------------------
# Vector layers
# ----------------------------

BOUNDARY_GPKG = (
    CACHE_DIR / "boundary_utm46.gpkg"
)

STREAMS_GPKG = (
    CACHE_DIR / "streams_utm46.gpkg"
)

LULC_GPKG = (
    CACHE_DIR
    / "lulc_2021_10k_clean_utm46.gpkg"
)

MGNREGA_GPKG = (
    CACHE_DIR / "mgnrega_utm46.gpkg"
)

FINAL94_GPKG = (
    CACHE_DIR / "final94_utm46.gpkg"
)

PUBLISHED_GPKG = (
    CACHE_DIR / "published_sites_utm46.gpkg"
)


WORKING_CRS = "EPSG:32646"

MAP_CENTER = [
    25.88,
    91.88,
]

MAP_ZOOM = 10


# ============================================================
# 3. MOBILE CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 0.4rem;
    padding-left: 0.65rem;
    padding-right: 0.65rem;
    padding-bottom: 3rem;
}

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 650;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d8e5e8;
    border-radius: 12px;
    padding: 10px;
}

.card {
    background: white;
    border: 1px solid #d8e5e8;
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
}

.green-card {
    border-left: 5px solid #159957;
}

.blue-card {
    border-left: 5px solid #1976D2;
}

.orange-card {
    border-left: 5px solid #EF6C00;
}

.red-card {
    border-left: 5px solid #C62828;
}

@media only screen and (max-width: 768px) {

    .block-container {
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }

    h1 {
        font-size: 1.45rem !important;
    }

    h2 {
        font-size: 1.20rem !important;
    }

    h3 {
        font-size: 1.05rem !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 60px;
        font-size: 17px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.18rem !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. INSTITUTIONAL HEADER
# ============================================================

st.markdown(
    """
<div style="
background:linear-gradient(120deg,#064c44,#087c6c);
padding:20px;
border-radius:16px;
color:white;
text-align:center;
margin-bottom:14px;
">

<div style="
font-size:14px;
font-weight:600;
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
border-top:1px solid rgba(255,255,255,0.35);
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
    str(p.relative_to(BASE_DIR))
    for p in REQUIRED_FILES
    if not p.exists()
]


if missing:

    st.error(
        "Required GIS files are missing."
    )

    st.code(
        "\n".join(missing)
    )

    st.stop()


# ============================================================
# 6. LOAD VECTOR DATA
# ============================================================

@st.cache_resource
def read_vector(path):

    if not path.exists():
        return None

    try:

        gdf = gpd.read_file(
            path
        )

        if gdf.crs is None:
            gdf = gdf.set_crs(
                WORKING_CRS
            )

        return gdf

    except Exception:

        return None


boundary_utm = read_vector(
    BOUNDARY_GPKG
)

streams_utm = read_vector(
    STREAMS_GPKG
)

lulc_utm = read_vector(
    LULC_GPKG
)

mgnrega_utm = read_vector(
    MGNREGA_GPKG
)

final94_utm = read_vector(
    FINAL94_GPKG
)

published_utm = read_vector(
    PUBLISHED_GPKG
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

STATE_DEFAULTS = {

    "field_geometry":
        None,

    "current_lulc":
        None,

    "assessment":
        None,
}


for key, value in STATE_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# 8. GEOMETRY HELPERS
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

        geom = drawing.get(
            "geometry",
            {}
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

        {"id": [1]},

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

        .geometry.iloc[0]

    )


def field_area_ha(
    geometry_geojson
):

    geom = to_utm_geometry(
        geometry_geojson
    )

    return float(
        geom.area
        / 10000.0
    )


def field_center_wgs84(
    geometry_geojson
):

    geom = to_utm_geometry(
        geometry_geojson
    )

    c = geom.centroid

    gs = (

        gpd.GeoSeries(
            [c],
            crs=WORKING_CRS,
        )

        .to_crs(
            "EPSG:4326"
        )

    )

    p = gs.iloc[0]

    return [
        float(p.y),
        float(p.x),
    ]


# ============================================================
# 9. SAFE RASTER STATISTICS
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
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
            }

        polygon_raster = (
            polygon.to_crs(
                src.crs
            )
        )

        geom = (
            polygon_raster
            .geometry.iloc[0]
        )

        values = np.array([])

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
                    != src.nodata
                ]


        except Exception:

            values = np.array([])


        # Small-field fallback

        if (
            len(values) == 0
            and
            centroid_fallback
        ):

            centroid = (
                geom.centroid
            )

            try:

                sampled = next(

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

                values = np.asarray(
                    sampled
                ).reshape(-1)

                values = values[
                    np.isfinite(
                        values
                    )
                ]

            except Exception:

                values = np.array([])


        if len(values) == 0:

            return {
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan,
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
# 10. POINT RASTER SAMPLING
# ============================================================

def sample_raster_at_xy(
    raster_path,
    x,
    y,
    point_crs=WORKING_CRS,
):

    with rasterio.open(
        raster_path
    ) as src:

        transformer = Transformer.from_crs(
            point_crs,
            src.crs,
            always_xy=True,
        )

        rx, ry = transformer.transform(
            x,
            y,
        )

        try:

            value = next(

                src.sample(
                    [(rx, ry)],
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
# 11. LULC FROM DETAILED VECTOR
# ============================================================

def mapped_lulc_summary(
    polygon_geojson
):

    if (
        lulc_utm is None
        or
        lulc_utm.empty
    ):

        return []


    selected = (
        geometry_gdf(
            polygon_geojson
        )
        .to_crs(
            lulc_utm.crs
        )
    )


    geom = (
        selected
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


    candidates = (
        lulc_utm
        .iloc[ids]
        .copy()
    )


    candidates = candidates[
        candidates.intersects(
            geom
        )
    ].copy()


    if candidates.empty:
        return []


    try:

        candidates[
            "geometry"
        ] = (
            candidates
            .geometry
            .make_valid()
        )

    except Exception:

        candidates[
            "geometry"
        ] = (
            candidates
            .geometry
            .buffer(0)
        )


    candidates[
        "clip_geom"
    ] = (
        candidates
        .geometry
        .intersection(
            geom
        )
    )


    candidates = candidates[
        ~candidates[
            "clip_geom"
        ].is_empty
    ].copy()


    if candidates.empty:
        return []


    candidates[
        "area_m2"
    ] = candidates[
        "clip_geom"
    ].area


    if (
        "LULC_2022"
        not in
        candidates.columns
    ):

        return []


    grouped = (

        candidates

        .groupby(
            "LULC_2022"
        )["area_m2"]

        .sum()

        .reset_index()

    )


    total = float(
        grouped[
            "area_m2"
        ].sum()
    )


    if total <= 0:
        return []


    grouped[
        "area_ha"
    ] = (
        grouped[
            "area_m2"
        ]
        / 10000.0
    )


    grouped[
        "percent"
    ] = (
        grouped[
            "area_m2"
        ]
        / total
        * 100.0
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
# 12. NEAREST VECTOR DISTANCE
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

        .geometry.iloc[0]
    )


    dist = (
        gdf
        .geometry
        .distance(
            field_geom
        )
    )


    if dist.empty:

        return None


    return float(
        dist.min()
    )


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


    lo = np.percentile(
        values,
        5
    )

    hi = np.percentile(
        values,
        95
    )


    if (
        not np.isfinite(lo)
        or
        not np.isfinite(hi)
        or
        hi <= lo
    ):

        return np.zeros_like(
            array,
            dtype=float,
        )


    normalized = (
        array - lo
    ) / (
        hi - lo
    )


    return np.clip(
        normalized,
        0,
        1,
    )


# ============================================================
# 14. READ FIELD ARRAY
# ============================================================

def crop_raster_to_field(
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

            .geometry.iloc[0]

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

            array = (
                data
                .filled(
                    np.nan
                )
                .astype(float)
            )

            mask = (
                ~data.mask
            )

        else:

            array = (
                data
                .astype(float)
            )

            mask = np.isfinite(
                array
            )


        return (
            array,
            mask,
            transform,
            src.crs,
        )


# ============================================================
# 15. AUTOMATIC POND SITE SELECTION
# ============================================================

def find_pond_sites(
    field_geojson,
    minimum_boundary_clearance=10.0,
):

    elev, elev_mask, transform, raster_crs = (
        crop_raster_to_field(
            ELEVATION_RASTER,
            field_geojson,
        )
    )


    slope, slope_mask, _, _ = (
        crop_raster_to_field(
            SLOPE_RASTER,
            field_geojson,
        )
    )


    twi, twi_mask, _, _ = (
        crop_raster_to_field(
            TWI_RASTER,
            field_geojson,
        )
    )


    dd, dd_mask, _, _ = (
        crop_raster_to_field(
            DRAINAGE_DENSITY_RASTER,
            field_geojson,
        )
    )


    # Flow accumulation may not have identical crop shape.
    # Use only if aligned.

    flowacc = None
    flow_mask = None

    try:

        fa, fm, _, _ = (
            crop_raster_to_field(
                FLOW_ACCUMULATION_RASTER,
                field_geojson,
            )
        )

        if fa.shape == elev.shape:

            flowacc = fa
            flow_mask = fm

    except Exception:

        pass


    valid = (
        elev_mask
        &
        slope_mask
        &
        twi_mask
        &
        dd_mask
        &
        np.isfinite(elev)
        &
        np.isfinite(slope)
        &
        np.isfinite(twi)
        &
        np.isfinite(dd)
    )


    if valid.sum() == 0:

        return None


    # ----------------------------------------
    # Absolute lowest DEM point
    # ----------------------------------------

    elev_work = np.where(
        valid,
        elev,
        np.inf,
    )

    low_index = np.unravel_index(
        np.argmin(
            elev_work
        ),
        elev_work.shape,
    )


    low_x, low_y = xy(
        transform,
        low_index[0],
        low_index[1],
        offset="center",
    )


    # ----------------------------------------
    # Suitability score
    # ----------------------------------------

    elev_norm = normalize_array(
        elev,
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

    dd_norm = normalize_array(
        dd,
        valid,
    )


    score = (

        0.45
        *
        (
            1.0
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
            1.0
            -
            slope_norm
        )

        +

        0.10
        *
        dd_norm

    )


    weight_sum = 0.95


    if (
        flowacc is not None
        and
        flow_mask is not None
    ):

        flow_valid = (
            valid
            &
            flow_mask
            &
            np.isfinite(
                flowacc
            )
        )


        # log transform due to highly skewed accumulation

        log_flow = np.log1p(
            np.maximum(
                flowacc,
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

        weight_sum += 0.05


    score = score / weight_sum


    # Preferred pond slopes

    preferred = (
        valid
        &
        (
            slope
            <= 8.0
        )
    )


    if preferred.sum() == 0:

        preferred = (
            valid
            &
            (
                slope
                <= 15.0
            )
        )


    if preferred.sum() == 0:

        preferred = valid


    score = np.where(
        preferred,
        score,
        -np.inf,
    )


    # ----------------------------------------
    # Field geometry in raster CRS
    # ----------------------------------------

    field_geom = (

        geometry_gdf(
            field_geojson
        )

        .to_crs(
            raster_crs
        )

        .geometry.iloc[0]

    )


    # ----------------------------------------
    # Nearby streams in same CRS
    # ----------------------------------------

    stream_subset = None

    if (
        streams_utm is not None
        and
        not streams_utm.empty
    ):

        stream_subset = (
            streams_utm
            .to_crs(
                raster_crs
            )
        )


    # ----------------------------------------
    # Ranked candidate pixels
    # ----------------------------------------

    flat_order = np.argsort(
        score.ravel()
    )[::-1]


    best = None


    for flat_idx in flat_order[:500]:

        r, c = np.unravel_index(
            flat_idx,
            score.shape,
        )


        if not np.isfinite(
            score[r, c]
        ):

            continue


        px, py = xy(
            transform,
            r,
            c,
            offset="center",
        )


        pt = Point(
            px,
            py,
        )


        boundary_distance = (
            pt.distance(
                field_geom.boundary
            )
        )


        # Avoid proposed off-stream farm pond
        # directly over mapped stream.

        stream_distance = None

        if (
            stream_subset is not None
            and
            not stream_subset.empty
        ):

            nearby = (

                stream_subset
                .geometry
                .distance(
                    pt
                )

            )

            stream_distance = float(
                nearby.min()
            )


        if (
            boundary_distance
            >=
            minimum_boundary_clearance
        ):

            if (
                stream_distance is None
                or
                stream_distance >= 10.0
            ):

                best = {

                    "row":
                        int(r),

                    "col":
                        int(c),

                    "x":
                        float(px),

                    "y":
                        float(py),

                    "elevation":
                        float(
                            elev[r, c]
                        ),

                    "slope":
                        float(
                            slope[r, c]
                        ),

                    "twi":
                        float(
                            twi[r, c]
                        ),

                    "drainage_density":
                        float(
                            dd[r, c]
                        ),

                    "score":
                        float(
                            score[r, c]
                        ),

                    "boundary_distance":
                        float(
                            boundary_distance
                        ),

                    "stream_distance":
                        stream_distance,

                }

                if flowacc is not None:

                    value = flowacc[
                        r,
                        c
                    ]

                    if np.isfinite(
                        value
                    ):

                        best[
                            "flow_accumulation"
                        ] = float(
                            value
                        )

                break


    # fallback if no point satisfied clearance

    if best is None:

        flat_idx = flat_order[0]

        r, c = np.unravel_index(
            flat_idx,
            score.shape,
        )

        px, py = xy(
            transform,
            r,
            c,
            offset="center",
        )

        best = {

            "row":
                int(r),

            "col":
                int(c),

            "x":
                float(px),

            "y":
                float(py),

            "elevation":
                float(
                    elev[r, c]
                ),

            "slope":
                float(
                    slope[r, c]
                ),

            "twi":
                float(
                    twi[r, c]
                ),

            "drainage_density":
                float(
                    dd[r, c]
                ),

            "score":
                float(
                    score[r, c]
                ),

            "boundary_distance":
                float(
                    Point(
                        px,
                        py
                    )
                    .distance(
                        field_geom.boundary
                    )
                ),

            "stream_distance":
                None,
        }


    transformer = Transformer.from_crs(
        raster_crs,
        "EPSG:4326",
        always_xy=True,
    )


    lon, lat = transformer.transform(
        best[
            "x"
        ],
        best[
            "y"
        ],
    )


    low_lon, low_lat = transformer.transform(
        low_x,
        low_y,
    )


    best[
        "longitude"
    ] = float(lon)

    best[
        "latitude"
    ] = float(lat)


    lowest = {

        "x":
            float(
                low_x
            ),

        "y":
            float(
                low_y
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
                elev[
                    low_index
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
# 16. POND VOLUME EQUATIONS
# ============================================================

def pond_geometry_from_bottom_width(
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
        / 3.0

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


    excavation_footprint = (
        excavation_top_length
        *
        excavation_top_width
    )


    # Wetted side slant length

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
        / 2.0

        *
        slant

    )


    short_side_area = (

        (
            bottom_width
            +
            water_top_width
        )
        / 2.0

        *
        slant

    )


    wetted_lining_area = (

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
            bottom_width,

        "bottom_length":
            bottom_length,

        "water_top_width":
            water_top_width,

        "water_top_length":
            water_top_length,

        "water_depth":
            water_depth,

        "freeboard":
            freeboard,

        "excavation_depth":
            excavation_depth,

        "excavation_top_width":
            excavation_top_width,

        "excavation_top_length":
            excavation_top_length,

        "bottom_area":
            bottom_area,

        "volume":
            volume,

        "footprint_area":
            excavation_footprint,

        "liner_wetted_area":
            wetted_lining_area,

        "liner_procurement_area":
            wetted_lining_area
            *
            1.10,
    }


# ============================================================
# 17. PRELIMINARY FARM POND DESIGN
# ============================================================

def design_farm_pond(
    available_water_m3,
    field_area_ha_value,
    storage_fraction,
    maximum_field_fraction,
    water_depth,
    freeboard=0.5,
    side_slope=1.5,
    length_width_ratio=1.5,
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


    max_footprint = (

        field_area_m2
        *
        maximum_field_fraction
    )


    # ----------------------------------------
    # Maximum pond fitting footprint constraint
    # ----------------------------------------

    lo = 0.5
    hi = 100.0


    for _ in range(80):

        mid = (
            lo + hi
        ) / 2.0


        geom = (
            pond_geometry_from_bottom_width(

                mid,
                water_depth,
                freeboard,
                side_slope,
                length_width_ratio,

            )
        )


        if (
            geom[
                "footprint_area"
            ]
            >
            max_footprint
        ):

            hi = mid

        else:

            lo = mid


    max_geom = (
        pond_geometry_from_bottom_width(

            lo,
            water_depth,
            freeboard,
            side_slope,
            length_width_ratio,

        )
    )


    maximum_storage = (
        max_geom[
            "volume"
        ]
    )


    target_storage = min(

        requested_storage,

        maximum_storage,

    )


    # ----------------------------------------
    # Solve bottom width for target volume
    # ----------------------------------------

    low_b = 0.5
    high_b = max(
        lo,
        1.0,
    )


    for _ in range(80):

        mid_b = (
            low_b
            +
            high_b
        ) / 2.0


        geom = (
            pond_geometry_from_bottom_width(

                mid_b,
                water_depth,
                freeboard,
                side_slope,
                length_width_ratio,

            )
        )


        if (
            geom[
                "volume"
            ]
            >
            target_storage
        ):

            high_b = mid_b

        else:

            low_b = mid_b


    result = (
        pond_geometry_from_bottom_width(

            low_b,
            water_depth,
            freeboard,
            side_slope,
            length_width_ratio,

        )
    )


    result[
        "requested_storage"
    ] = requested_storage


    result[
        "target_storage"
    ] = target_storage


    result[
        "maximum_storage_by_land"
    ] = maximum_storage


    result[
        "maximum_allowed_footprint"
    ] = max_footprint


    result[
        "field_area_m2"
    ] = field_area_m2


    result[
        "field_occupied_percent"
    ] = (

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
    ] = (

        (
            field_area_m2

            -

            result[
                "footprint_area"
            ]
        )

        /
        10000.0
    )


    return result


# ============================================================
# 18. RANK LOOKUPS
# ============================================================

def soil_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):
        return "Unknown"

    rank = int(
        round(rank)
    )

    mapping_dict = {

        1:
            "Clay / Clayey / Clay loam",

        2:
            "Silt clay loam / Silt loam",

        3:
            "Sandy clay loam",

        4:
            "Miscellaneous soil class",

        5:
            "Loam",
    }

    return mapping_dict.get(
        rank,
        "Unknown"
    )


def geology_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):
        return "Unknown"

    rank = int(
        round(rank)
    )

    mapping_dict = {

        5:
            "Quaternary",

        3:
            (
                "Umsning Schist / "
                "Assam-Meghalaya Gneissic"
            ),

        2:
            (
                "Shillong Group / "
                "Kyrdem"
            ),
    }

    return mapping_dict.get(
        rank,
        "Unmatched/other"
    )


def geomorph_rank_description(
    rank
):

    if not np.isfinite(
        rank
    ):
        return "Unknown"

    rank = int(
        round(rank)
    )

    mapping_dict = {

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
                "Highly/moderately dissected "
                "hills or plateau"
            ),

        1:
            (
                "Scarp / Ridge / "
                "Low dissected upper plateau"
            ),
    }

    return mapping_dict.get(
        rank,
        "Unknown"
    )


# ============================================================
# 19. SEEPAGE SCREENING
# ============================================================

def seepage_assessment(
    soil_rank,
    geology_rank,
    geomorph_rank,
    lineament_density,
):

    score = 0
    reasons = []


    # ----------------------------------------
    # Soil
    # Groundwater ranks are reinterpreted for
    # storage retention.
    # ----------------------------------------

    if np.isfinite(
        soil_rank
    ):

        sr = int(
            round(
                soil_rank
            )
        )


        if sr == 1:

            reasons.append(
                "Clayey/clay-loam soil rank suggests relatively favourable water retention."
            )


        elif sr == 2:

            score += 1

            reasons.append(
                "Silty soil class may permit moderate seepage depending on field condition."
            )


        elif sr == 3:

            score += 2

            reasons.append(
                "Sandy clay loam indicates increased seepage potential."
            )


        elif sr == 5:

            score += 2

            reasons.append(
                "Loam groundwater rank indicates potentially greater permeability than clayey soil."
            )


        else:

            score += 1

            reasons.append(
                "Soil class requires field permeability verification."
            )


    # ----------------------------------------
    # Lineament density
    # Approximate thresholds guided by observed
    # stack distribution.
    # ----------------------------------------

    if np.isfinite(
        lineament_density
    ):

        if lineament_density >= 0.55:

            score += 2

            reasons.append(
                "High lineament density indicates elevated fracture-related seepage risk."
            )

        elif lineament_density >= 0.19:

            score += 1

            reasons.append(
                "Moderate lineament density indicates some fracture-related seepage risk."
            )

        else:

            reasons.append(
                "Low local lineament density reduces fracture-related seepage concern."
            )


    # ----------------------------------------
    # Geology
    # Do not over-interpret groundwater rank.
    # ----------------------------------------

    if np.isfinite(
        geology_rank
    ):

        gr = int(
            round(
                geology_rank
            )
        )

        if gr == 5:

            score += 1

            reasons.append(
                "Quaternary material can be heterogeneous; permeability must be verified in the field."
            )

        elif gr in [2, 3]:

            reasons.append(
                "Bedrock geological setting requires checking weathering and fractures at excavation depth."
            )


    if score >= 4:

        risk = "HIGH"

        lining = (
            "Impermeable lining is strongly recommended. "
            "Consider a properly prepared compacted subgrade "
            "with an appropriate geomembrane/approved pond-lining "
            "system after field permeability and geotechnical verification."
        )


    elif score >= 2:

        risk = "MODERATE"

        lining = (
            "Lining should be considered. A compacted clay blanket "
            "may be adequate where suitable clay is available; "
            "use an approved geomembrane where reliable retention "
            "is essential or field permeability is high."
        )


    else:

        risk = "LOW"

        lining = (
            "Natural/compacted soil treatment may be adequate, "
            "but a field permeability test is still required before "
            "deciding that synthetic lining is unnecessary."
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
# 20. RESET
# ============================================================

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
# 22. STEP 1 — DRAW FIELD
# ============================================================

st.header(
    "1️⃣ Demarcate Farmer Field"
)


st.info(
    "Use the polygon tool ⬠ and tap the actual corners "
    "of the farmer's land. Close the polygon by tapping "
    "the first point again."
)


field_map = folium.Map(

    location=MAP_CENTER,

    zoom_start=MAP_ZOOM,

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


if boundary_wgs is not None:

    folium.GeoJson(

        boundary_wgs.__geo_interface__,

        name="Ri Bhoi District",

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


MeasureControl(

    primary_length_unit="meters",

    primary_area_unit="hectares",

).add_to(
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


if new_field is not None:

    if (
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


field = st.session_state[
    "field_geometry"
]


# ============================================================
# 23. FIELD DETAILS
# ============================================================

if field:

    area_ha_value = field_area_ha(
        field
    )


    lulc_summary = mapped_lulc_summary(
        field
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
        f"{area_ha_value:.3f} ha"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Field Area",
        f"{area_ha_value:.3f} ha",
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

        default_lulc_index = (
            LAND_USE_OPTIONS.index(
                st.session_state[
                    "current_lulc"
                ]
            )
        )

    elif mapped_lulc in LAND_USE_OPTIONS:

        default_lulc_index = (
            LAND_USE_OPTIONS.index(
                mapped_lulc
            )
        )

    else:

        default_lulc_index = 0


    st.markdown(
        "### 🌾 Confirm Present Land Use"
    )


    current_lulc = st.selectbox(

        "Current land use observed by farmer/field team",

        LAND_USE_OPTIONS,

        index=default_lulc_index,

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
            f"GIS layer indicates **{mapped_lulc}**, "
            f"but present land use is confirmed as "
            f"**{current_lulc}**. The DSS will use the "
            "confirmed present land use for structure planning."
        )


    with st.expander(
        "GIS LULC composition inside farmer field"
    ):

        st.dataframe(
            pd.DataFrame(
                lulc_summary
            ),
            use_container_width=True,
            hide_index=True,
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

        storage_fraction_percent = st.slider(

            "One-time storage as % of 75% dependable annual runoff",

            min_value=10,

            max_value=60,

            value=30,

            step=5,

        )


        maximum_land_percent = st.slider(

            "Maximum farmer land occupied by pond (%)",

            min_value=5,

            max_value=20,

            value=10,

            step=1,

        )


        water_depth = st.slider(

            "Planning water depth (m)",

            min_value=1.5,

            max_value=3.5,

            value=2.5,

            step=0.25,

        )


        freeboard = st.slider(

            "Freeboard (m)",

            min_value=0.3,

            max_value=0.75,

            value=0.5,

            step=0.05,

        )


        side_slope = st.slider(

            "Planning side slope (H:1V)",

            min_value=1.0,

            max_value=2.5,

            value=1.5,

            step=0.25,

        )


        length_width_ratio = st.slider(

            "Bottom length : breadth ratio",

            min_value=1.0,

            max_value=2.0,

            value=1.5,

            step=0.1,

        )


# ============================================================
# 25. RUN ASSESSMENT
# ============================================================

if field:

    st.markdown("---")

    st.header(
        "3️⃣ Automatic Pond Site & Design"
    )


    if st.button(

        "💧 FIND FARM POND LOCATION & DESIGN",

        type="primary",

        use_container_width=True,

    ):

        area_ha_value = field_area_ha(
            field
        )


        # ------------------------------------
        # Hydrology
        # ------------------------------------

        rainfall = safe_raster_stats(
            RAINFALL_RASTER,
            field,
        )["mean"]


        mean_runoff = safe_raster_stats(
            MEAN_RUNOFF_RASTER,
            field,
        )["mean"]


        dependable_runoff = safe_raster_stats(
            DEPENDABLE_RUNOFF_RASTER,
            field,
        )["mean"]


        runoff_coeff = safe_raster_stats(
            RUNOFF_COEFFICIENT_RASTER,
            field,
        )["mean"]


        cn = safe_raster_stats(
            CN_RASTER,
            field,
        )["mean"]


        mean_runoff_volume = (

            mean_runoff
            *
            area_ha_value
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
            area_ha_value
            *
            10

            if np.isfinite(
                dependable_runoff
            )

            else np.nan

        )


        available_for_design = (

            dependable_volume

            if np.isfinite(
                dependable_volume
            )

            else mean_runoff_volume

        )


        if not np.isfinite(
            available_for_design
        ):

            st.error(
                "Runoff data are not available for this field."
            )

            st.stop()


        # ------------------------------------
        # Pond design
        # ------------------------------------

        pond_design = design_farm_pond(

            available_for_design,

            area_ha_value,

            storage_fraction=
                (
                    storage_fraction_percent
                    /
                    100.0
                ),

            maximum_field_fraction=
                (
                    maximum_land_percent
                    /
                    100.0
                ),

            water_depth=
                water_depth,

            freeboard=
                freeboard,

            side_slope=
                side_slope,

            length_width_ratio=
                length_width_ratio,

        )


        # ------------------------------------
        # Boundary clearance based partly
        # on proposed footprint.
        # ------------------------------------

        half_diagonal = (

            0.5

            *

            math.sqrt(

                pond_design[
                    "excavation_top_length"
                ] ** 2

                +

                pond_design[
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


        sites = find_pond_sites(

            field,

            minimum_boundary_clearance=
                clearance,

        )


        if sites is None:

            st.error(
                "No valid DEM cells were found inside the selected field."
            )

            st.stop()


        site = sites[
            "recommended"
        ]


        lowest = sites[
            "absolute_lowest"
        ]


        # ------------------------------------
        # Site-specific thematic values
        # ------------------------------------

        geology_rank = sample_raster_at_xy(

            GEOLOGY_RASTER,

            site[
                "x"
            ],

            site[
                "y"
            ],

        )


        geomorph_rank = sample_raster_at_xy(

            GEOMORPH_RASTER,

            site[
                "x"
            ],

            site[
                "y"
            ],

        )


        soil_rank = sample_raster_at_xy(

            SOIL_RASTER,

            site[
                "x"
            ],

            site[
                "y"
            ],

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


        # ------------------------------------
        # Seepage
        # ------------------------------------

        seepage = seepage_assessment(

            soil_rank,

            geology_rank,

            geomorph_rank,

            lineament_density,

        )


        # ------------------------------------
        # Nearby RWH data
        # ------------------------------------

        nearest_stream = nearest_distance(
            field,
            streams_utm,
        )


        nearest_mgnrega = nearest_distance(
            field,
            mgnrega_utm,
        )


        nearest_final94 = nearest_distance(
            field,
            final94_utm,
        )


        nearest_published = nearest_distance(
            field,
            published_utm,
        )


        # ------------------------------------
        # Store assessment
        # ------------------------------------

        st.session_state[
            "assessment"
        ] = {

            "field_area_ha":
                area_ha_value,

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
                runoff_coeff,

            "cn":
                cn,

            "mean_runoff_volume":
                mean_runoff_volume,

            "dependable_volume":
                dependable_volume,

            "pond_design":
                pond_design,

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
                nearest_stream,

            "nearest_mgnrega":
                nearest_mgnrega,

            "nearest_final94":
                nearest_final94,

            "nearest_published":
                nearest_published,

        }


        st.rerun()


# ============================================================
# 26. RESULTS
# ============================================================

result = st.session_state[
    "assessment"
]


if result:

    site = result[
        "site"
    ]

    lowest = result[
        "lowest"
    ]

    design = result[
        "pond_design"
    ]

    seepage = result[
        "seepage"
    ]


    st.markdown("---")

    st.header(
        "✅ Farmer RWH Planning Result"
    )


    # ========================================================
    # STRUCTURE
    # ========================================================

    if (
        "crop"
        in result[
            "current_lulc"
        ].lower()

        or

        "plantation"
        in result[
            "current_lulc"
        ].lower()
    ):

        st.success(
            """
### 🏗️ Primary Structure
**Farm Pond**

The structure is proposed within the farmer's field at the
lowest practical runoff-convergence zone identified from
elevation, slope, TWI, drainage context and flow accumulation.
"""
        )

    else:

        st.warning(
            """
The selected land use is not clearly agricultural.
The farm-pond design below should therefore be treated as
a screening result and verified before implementation.
"""
        )


    # ========================================================
    # FIELD WATER
    # ========================================================

    st.subheader(
        "🌧️ Water Availability"
    )


    c1, c2 = st.columns(
        2
    )


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


    c1, c2 = st.columns(
        2
    )


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


    c1, c2 = st.columns(
        2
    )


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


    # ========================================================
    # SITE LOCATION
    # ========================================================

    st.subheader(
        "📍 Recommended Farm Pond Location"
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
        "Local Slope",
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


    st.caption(
        "The red point is the recommended practical pond centre. "
        "The yellow point is the absolute lowest DEM pixel. "
        "They may differ if the absolute lowest location is too close "
        "to the farm boundary, stream, or has less favourable terrain."
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


    folium.Marker(

        location=[

            site[
                "latitude"
            ],

            site[
                "longitude"
            ],

        ],

        tooltip="Recommended Farm Pond Site",

        popup=(

            f"<b>Recommended Farm Pond</b><br>"
            f"Elevation: {site['elevation']:.1f} m<br>"
            f"Slope: {site['slope']:.2f}°<br>"
            f"TWI: {site['twi']:.2f}"

        ),

        icon=folium.Icon(
            color="red",
            icon="tint",
        ),

    ).add_to(
        result_map
    )


    folium.Marker(

        location=[

            lowest[
                "latitude"
            ],

            lowest[
                "longitude"
            ],

        ],

        tooltip="Absolute Lowest DEM Point",

        popup=(

            "<b>Absolute Lowest Point</b><br>"
            f"Elevation: {lowest['elevation']:.1f} m"

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
    # POND DESIGN
    # ========================================================

    st.subheader(
        "📐 Preliminary Farm Pond Design"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Proposed Storage",
        f"{design['target_storage']:,.0f} m³",
    )


    c2.metric(
        "Water Depth",
        f"{design['water_depth']:.2f} m",
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Bottom Length",
        f"{design['bottom_length']:.1f} m",
    )


    c2.metric(
        "Bottom Breadth",
        f"{design['bottom_width']:.1f} m",
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Top Length",
        f"{design['excavation_top_length']:.1f} m",
    )


    c2.metric(
        "Top Breadth",
        f"{design['excavation_top_width']:.1f} m",
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Excavation Depth",
        f"{design['excavation_depth']:.2f} m",
    )


    c2.metric(
        "Freeboard",
        f"{design['freeboard']:.2f} m",
    )


    c1, c2 = st.columns(
        2
    )


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
    ):

        st.warning(
            "The desired storage volume would occupy more than "
            "the permitted percentage of the farmer's land. "
            "The DSS therefore reduced the pond capacity to a "
            "preliminary size that fits the selected land-use limit."
        )


    # ========================================================
    # SEEPAGE
    # ========================================================

    st.subheader(
        "🪨 Seepage & Pond Lining Assessment"
    )


    risk = seepage[
        "risk"
    ]


    if risk == "HIGH":

        st.error(
            f"Seepage Risk: {risk}"
        )

    elif risk == "MODERATE":

        st.warning(
            f"Seepage Risk: {risk}"
        )

    else:

        st.success(
            f"Seepage Risk: {risk}"
        )


    c1, c2 = st.columns(
        2
    )


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
        "**Lineament density at proposed pond:**",
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


    c1, c2 = st.columns(
        2
    )


    c1.metric(
        "Estimated Wetted Lining Area",
        f"{design['liner_wetted_area']:,.0f} m²",
    )


    c2.metric(
        "Liner Procurement Area (+10%)",
        f"{design['liner_procurement_area']:,.0f} m²",
    )


    # ========================================================
    # TECHNICAL CONTEXT
    # ========================================================

    with st.expander(
        "🔬 Technical Site Information"
    ):

        st.write(
            "Current land use:",
            result[
                "current_lulc"
            ],
        )

        st.write(
            "GIS-mapped LULC:",
            result[
                "mapped_lulc"
            ],
        )

        st.write(
            "CN-II:",
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
            "Runoff coefficient:",
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
            "Drainage density:",
            round(
                site[
                    "drainage_density"
                ],
                3,
            ),
        )

        if (
            "flow_accumulation"
            in site
        ):

            st.write(
                "Flow accumulation:",
                round(
                    site[
                        "flow_accumulation"
                    ],
                    2,
                ),
            )

        st.write(
            "Distance from farmer field to mapped stream:",
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
            "Nearest MGNREGA RWH structure:",
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
            "Nearest Final-94 candidate:",
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
            "Nearest published RWH site:",
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
    # MANAGEMENT
    # ========================================================

    st.subheader(
        "🌱 Construction & Management Guidance"
    )


    st.markdown(
        """
<div class="card green-card">

<b>Farm Pond Management</b><br><br>

• Confirm the proposed point by field survey before excavation.<br>
• Provide a sediment/silt trap at the runoff inlet.<br>
• Do not block a natural stream for an off-stream farm pond.<br>
• Provide a safe overflow/spill arrangement for excess monsoon water.<br>
• Stabilize exposed pond bunds with suitable vegetation.<br>
• Desilt the inlet and pond periodically.<br>
• Protect any geomembrane from puncture during installation and operation.<br>
• Use stored water primarily for protective/supplemental irrigation where appropriate.<br>
• Verify soil permeability and foundation conditions before finalizing lining.<br>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
This DSS provides preliminary GIS-based planning and screening.

The proposed coordinates identify a 30-m GIS candidate zone rather
than a surveyed construction peg. Final pond location must be checked
on the ground for micro-topography, ownership boundary, access,
foundation conditions, natural drainage, utilities and environmental
constraints.

The calculated pond dimensions are preliminary planning dimensions.
Final embankment, inlet, outlet, spillway, excavation, lining,
freeboard and structural dimensions require engineering design and
field verification.

Rainfall values retain the information content of the original
rainfall dataset even when stored on a 30-m analysis grid.
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
