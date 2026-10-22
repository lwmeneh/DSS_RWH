from pathlib import Path
import base64
import math

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
# 1. PAGE CONFIG
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


# ============================================================
# 3. RASTER FILES
# ============================================================

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
# 4. VECTOR FILES
# ============================================================

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


WORKING_CRS = "EPSG:32646"

MAP_CENTER = [
    25.88,
    91.88,
]

MAP_ZOOM = 12


# ============================================================
# 5. MOBILE-FIRST CSS
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

html,
body,
[class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

h1,
h2,
h3 {
    color: #143e35;
}


/* Buttons */

.stButton > button,
.stDownloadButton > button {

    width: 100%;
    min-height: 54px;

    border-radius: 12px;

    font-size: 16px;
    font-weight: 650;
}


/* Metrics */

[data-testid="stMetric"] {

    background: #ffffff;

    border:
        1px solid #d5e5df;

    border-radius: 14px;

    padding: 11px;

    box-shadow:
        0 1px 4px
        rgba(0,0,0,0.05);
}


/* Main header */

.geopond-hero {

    background:
        linear-gradient(
            135deg,
            #07513f 0%,
            #08735c 55%,
            #10956f 100%
        );

    border-radius: 18px;

    padding: 18px 22px;

    color: white;

    margin-bottom: 18px;

    box-shadow:
        0 5px 18px
        rgba(0,0,0,0.14);
}


.geopond-row {

    display: flex;

    align-items: center;

    gap: 18px;
}


.geopond-logo {

    flex:
        0 0 auto;
}


.geopond-logo img {

    width: 100px;

    max-width: 100px;

    background: white;

    border-radius: 13px;

    padding: 7px;
}


.geopond-icar {

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 0.4px;

    opacity: 0.95;
}


.geopond-institute {

    font-size: 17px;

    font-weight: 700;

    margin-top: 2px;
}


.geopond-name {

    font-size: 31px;

    font-weight: 850;

    line-height: 1.05;

    margin-top: 9px;
}


.geopond-fullname {

    font-size: 16px;

    font-weight: 500;

    line-height: 1.45;

    margin-top: 5px;
}


.geopond-badge {

    display: inline-block;

    margin-top: 10px;

    padding:
        6px 12px;

    border-radius:
        999px;

    background:
        rgba(
            255,
            255,
            255,
            0.16
        );

    font-size: 12px;

    font-weight: 650;
}


/* GPS farmer guide */

.gps-card {

    background:
        #edf7ff;

    border:
        1px solid #cfe5f6;

    border-left:
        5px solid #1373b5;

    border-radius:
        13px;

    padding:
        13px 15px;

    margin:
        8px 0 12px 0;

    color:
        #164a72;

    line-height:
        1.55;
}


.result-card {

    background: white;

    border:
        1px solid #d5e5df;

    border-left:
        5px solid #08735c;

    border-radius: 13px;

    padding: 14px;

    margin:
        8px 0;
}


/* Mobile */

@media only screen and (max-width: 768px) {

    .block-container {

        padding-left:
            0.25rem;

        padding-right:
            0.25rem;
    }


    .geopond-hero {

        padding:
            15px 12px;

        border-radius:
            13px;
    }


    .geopond-row {

        flex-direction:
            column;

        text-align:
            center;

        gap:
            8px;
    }


    .geopond-logo img {

        width:
            78px;

        max-width:
            78px;
    }


    .geopond-icar {

        font-size:
            11px;
    }


    .geopond-institute {

        font-size:
            14px;
    }


    .geopond-name {

        font-size:
            25px;
    }


    .geopond-fullname {

        font-size:
            14px;
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

        min-height:
            58px;

        font-size:
            16px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 6. IMAGE TO BASE64
# ============================================================

def image_to_base64(
    image_path
):

    try:

        if not image_path.exists():

            return None


        with open(
            image_path,
            "rb"
        ) as image_file:

            return (

                base64
                .b64encode(
                    image_file.read()
                )
                .decode(
                    "utf-8"
                )

            )

    except Exception:

        return None


# ============================================================
# 7. PROFESSIONAL HEADER
# ============================================================

logo_base64 = (
    image_to_base64(
        ICAR_LOGO
    )
)


if logo_base64:

    header_html = f"""<div class="geopond-hero">
<div class="geopond-row">
<div class="geopond-logo">
<img src="data:image/png;base64,{logo_base64}" alt="ICAR Logo">
</div>
<div>
<div class="geopond-icar">
INDIAN COUNCIL OF AGRICULTURAL RESEARCH
</div>
<div class="geopond-institute">
ICAR Research Complex for NEH Region, Umiam, Meghalaya
</div>
<div class="geopond-name">
GeoPond-DSS
</div>
<div class="geopond-fullname">
<b>Geo</b>spatial <b>Pond</b> Siting and Preliminary Design Decision Support System
</div>
<div class="geopond-badge">
Ri Bhoi District • Meghalaya • Mobile GPS Enabled
</div>
</div>
</div>
</div>"""

else:

    header_html = """<div class="geopond-hero">
<div class="geopond-row">
<div style="font-size:52px;">
🌾
</div>
<div>
<div class="geopond-icar">
INDIAN COUNCIL OF AGRICULTURAL RESEARCH
</div>
<div class="geopond-institute">
ICAR Research Complex for NEH Region, Umiam, Meghalaya
</div>
<div class="geopond-name">
GeoPond-DSS
</div>
<div class="geopond-fullname">
<b>Geo</b>spatial <b>Pond</b> Siting and Preliminary Design Decision Support System
</div>
<div class="geopond-badge">
Ri Bhoi District • Meghalaya • Mobile GPS Enabled
</div>
</div>
</div>
</div>"""


st.markdown(
    header_html,
    unsafe_allow_html=True,
)


# ============================================================
# 8. REQUIRED FILE CHECK
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


missing_files = [

    str(
        file_path.relative_to(
            BASE_DIR
        )
    )

    for file_path
    in REQUIRED_FILES

    if not file_path.exists()

]


if missing_files:

    st.error(
        "GeoPond-DSS GIS database is incomplete."
    )

    st.write(
        "Copy the following files into the repository:"
    )

    st.code(
        "\n".join(
            missing_files
        )
    )

    st.stop()


# ============================================================
# 9. VECTOR LOADER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_vector(
    path_string
):

    path = Path(
        path_string
    )


    if not path.exists():

        return None


    try:

        data = gpd.read_file(
            path
        )


        if data.crs is None:

            data = data.set_crs(
                WORKING_CRS
            )


        return data


    except Exception:

        return None


boundary_utm = load_vector(
    str(
        BOUNDARY_GPKG
    )
)

streams_utm = load_vector(
    str(
        STREAMS_GPKG
    )
)

lulc_utm = load_vector(
    str(
        LULC_GPKG
    )
)

mgnrega_utm = load_vector(
    str(
        MGNREGA_GPKG
    )
)

final94_utm = load_vector(
    str(
        FINAL94_GPKG
    )
)

published_utm = load_vector(
    str(
        PUBLISHED_GPKG
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
# 10. SESSION STATE
# ============================================================

STATE_DEFAULTS = {

    "field_geometry":
        None,

    "assessment":
        None,

    "current_lulc":
        None,
}


for key, value in (
    STATE_DEFAULTS.items()
):

    if (
        key
        not in
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
# 11. GEOMETRY HELPERS
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

    geometry = (
        to_utm_geometry(
            geometry_geojson
        )
    )


    return float(

        geometry.area
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

        or []

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


        if (
            geometry.get(
                "type"
            )

            in [

                "Polygon",

                "MultiPolygon",

            ]
        ):

            polygon = geometry


    return polygon


# ============================================================
# 12. RASTER MEAN
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
    ) as source:

        geometry = (

            polygon

            .to_crs(
                source.crs
            )

            .geometry
            .iloc[0]

        )


        values = np.array(
            [],
            dtype=float,
        )


        try:

            clipped, _ = (
                rio_mask(

                    source,

                    [
                        mapping(
                            geometry
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
                source.nodata
                is not None
            ):

                values = values[
                    values
                    !=
                    source.nodata
                ]


        except Exception:

            values = np.array(
                [],
                dtype=float,
            )


        # ----------------------------------------
        # Small-field / coarse-raster fallback
        # ----------------------------------------

        if len(values) == 0:

            centroid = (
                geometry.centroid
            )


            try:

                sampled = next(

                    source.sample(

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


                sampled = np.asarray(
                    sampled,
                    dtype=float,
                ).reshape(-1)


                sampled = sampled[
                    np.isfinite(
                        sampled
                    )
                ]


                if len(
                    sampled
                ) > 0:

                    values = sampled


            except Exception:

                pass


        if len(values) == 0:

            return np.nan


        return float(
            np.mean(
                values
            )
        )


# ============================================================
# 13. SAMPLE RASTER AT POND POINT
# ============================================================

def sample_raster(
    raster_path,
    x,
    y,
):

    with rasterio.open(
        raster_path
    ) as source:

        transformer = (
            Transformer.from_crs(

                WORKING_CRS,

                source.crs,

                always_xy=True,

            )
        )


        raster_x, raster_y = (
            transformer.transform(
                x,
                y
            )
        )


        try:

            value = next(

                source.sample(

                    [
                        (
                            raster_x,
                            raster_y,
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
# 14. FIELD REFERENCE GRID
# ============================================================

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
    ) as source:

        geometry = (

            polygon

            .to_crs(
                source.crs
            )

            .geometry
            .iloc[0]

        )


        data, transform = (
            rio_mask(

                source,

                [
                    mapping(
                        geometry
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
                    float
                )

            )


            valid = (
                ~np.ma.getmaskarray(
                    data
                )
            )


        else:

            array = (
                data.astype(
                    float
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

            source.crs,

        )


# ============================================================
# 15. ALIGN RASTER TO ELEVATION GRID
# ============================================================

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
    ) as source:

        source_array = (

            source
            .read(1)
            .astype(
                np.float64
            )

        )


        if (
            source.nodata
            is not None
        ):

            source_array[
                source_array
                ==
                source.nodata
            ] = np.nan


        reproject(

            source=
                source_array,

            destination=
                destination,

            src_transform=
                source.transform,

            src_crs=
                source.crs,

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


# ============================================================
# 16. NORMALIZATION
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


    low = float(
        np.percentile(
            values,
            5
        )
    )


    high = float(
        np.percentile(
            values,
            95
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


    normalized = (

        array
        -
        low

    ) / (

        high
        -
        low

    )


    return np.clip(

        normalized,

        0.0,

        1.0,

    )


# ============================================================
# 17. DETAILED LULC
# ============================================================

def mapped_lulc_summary(
    field_geojson
):

    if (

        lulc_utm
        is None

        or

        lulc_utm.empty

        or

        "LULC_2022"
        not in
        lulc_utm.columns

    ):

        return []


    field_geometry = (

        geometry_gdf(
            field_geojson
        )

        .to_crs(
            lulc_utm.crs
        )

        .geometry
        .iloc[0]

    )


    candidate_ids = list(

        lulc_utm
        .sindex
        .intersection(
            field_geometry.bounds
        )

    )


    if not candidate_ids:

        return []


    candidates = (

        lulc_utm

        .iloc[
            candidate_ids
        ]

        .copy()

    )


    candidates = candidates[

        candidates.intersects(
            field_geometry
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
        "clipped_geometry"
    ] = (

        candidates
        .geometry
        .intersection(
            field_geometry
        )

    )


    candidates[
        "area_m2"
    ] = (

        candidates[
            "clipped_geometry"
        ]
        .area

    )


    grouped = (

        candidates

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
        "Area (ha)"
    ] = (

        grouped[
            "area_m2"
        ]

        /
        10000.0

    )


    grouped[
        "Percent"
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
            "Percent",
            ascending=False,
        )
    )


    return [

        {

            "LULC":

                str(
                    row[
                        "LULC_2022"
                    ]
                ),

            "Area (ha)":

                round(
                    float(
                        row[
                            "Area (ha)"
                        ]
                    ),
                    3,
                ),

            "Percent":

                round(
                    float(
                        row[
                            "Percent"
                        ]
                    ),
                    1,
                ),

        }

        for _, row
        in grouped.iterrows()

    ]


# ============================================================
# 18. NEAREST DISTANCE
# ============================================================

def nearest_distance(
    field_geojson,
    vector_layer,
):

    if (

        vector_layer
        is None

        or

        vector_layer.empty

    ):

        return None


    geometry = (

        geometry_gdf(
            field_geojson
        )

        .to_crs(
            vector_layer.crs
        )

        .geometry
        .iloc[0]

    )


    distances = (

        vector_layer
        .geometry
        .distance(
            geometry
        )

    )


    if distances.empty:

        return None


    return float(
        distances.min()
    )


# ============================================================
# 19. AUTOMATIC POND SITE SELECTION
# ============================================================

def find_pond_site(
    field_geojson,
    boundary_clearance=10.0,
):

    (
        elevation,
        elevation_valid,
        transform,
        raster_crs,
    ) = field_reference_grid(
        field_geojson
    )


    reference_shape = (
        elevation.shape
    )


    slope = raster_to_reference(

        SLOPE_RASTER,

        reference_shape,

        transform,

        raster_crs,

    )


    twi = raster_to_reference(

        TWI_RASTER,

        reference_shape,

        transform,

        raster_crs,

    )


    drainage_density = raster_to_reference(

        DRAINAGE_DENSITY_RASTER,

        reference_shape,

        transform,

        raster_crs,

    )


    flow_accumulation = raster_to_reference(

        FLOW_ACCUMULATION_RASTER,

        reference_shape,

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
            drainage_density
        )

    )


    if np.count_nonzero(
        valid
    ) == 0:

        return None


    # --------------------------------------------------------
    # Absolute lowest DEM point
    # --------------------------------------------------------

    elevation_search = (
        np.where(
            valid,
            elevation,
            np.inf,
        )
    )


    lowest_index = (
        np.unravel_index(

            np.argmin(
                elevation_search
            ),

            elevation_search.shape,

        )
    )


    lowest_x, lowest_y = xy(

        transform,

        lowest_index[0],

        lowest_index[1],

        offset="center",

    )


    # --------------------------------------------------------
    # Normalize criteria
    # --------------------------------------------------------

    elevation_norm = (
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
            drainage_density,
            valid,
        )
    )


    # --------------------------------------------------------
    # GeoPond site score
    #
    # Lower elevation        45%
    # Higher TWI             25%
    # Gentler slope          15%
    # Drainage density       10%
    # Flow accumulation       5%
    # --------------------------------------------------------

    site_score = (

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


    flow_valid = (

        valid

        &

        np.isfinite(
            flow_accumulation
        )

    )


    if np.count_nonzero(
        flow_valid
    ) > 0:

        flow_log = np.log1p(

            np.maximum(
                flow_accumulation,
                0.0,
            )

        )


        flow_norm = (
            normalize_array(
                flow_log,
                flow_valid,
            )
        )


        site_score = (

            site_score

            +

            0.05
            *
            flow_norm

        )


    # --------------------------------------------------------
    # Preferred terrain
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


    if np.count_nonzero(
        preferred
    ) == 0:

        preferred = (

            valid

            &

            (
                slope
                <=
                15.0
            )

        )


    if np.count_nonzero(
        preferred
    ) == 0:

        preferred = valid


    site_score = np.where(

        preferred,

        site_score,

        -np.inf,

    )


    # --------------------------------------------------------
    # Farmer field
    # --------------------------------------------------------

    farmer_field = (

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
    # Nearby stream geometry
    # --------------------------------------------------------

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


        search_zone = (
            farmer_field
            .buffer(
                1000.0
            )
        )


        candidate_ids = list(

            streams_local
            .sindex
            .intersection(
                search_zone.bounds
            )

        )


        if candidate_ids:

            local_streams = (

                streams_local

                .iloc[
                    candidate_ids
                ]

                .copy()

            )


            local_streams = local_streams[

                local_streams
                .intersects(
                    search_zone
                )

            ]


            if not local_streams.empty:

                stream_union = (
                    unary_union(

                        local_streams
                        .geometry
                        .tolist()

                    )
                )


    # --------------------------------------------------------
    # Highest-scoring feasible point
    # --------------------------------------------------------

    ordering = (

        np.argsort(
            site_score.ravel()
        )[::-1]

    )


    best = None


    for flat_index in (
        ordering[:1500]
    ):

        row, column = (
            np.unravel_index(

                flat_index,

                site_score.shape,

            )
        )


        current_score = (
            site_score[
                row,
                column
            ]
        )


        if not np.isfinite(
            current_score
        ):

            continue


        point_x, point_y = xy(

            transform,

            row,

            column,

            offset="center",

        )


        point = Point(
            point_x,
            point_y,
        )


        distance_boundary = float(

            point.distance(
                farmer_field.boundary
            )

        )


        if (

            distance_boundary
            <
            boundary_clearance

        ):

            continue


        distance_stream = None


        if (
            stream_union
            is not None
        ):

            distance_stream = float(

                point.distance(
                    stream_union
                )

            )


            if (
                distance_stream
                <
                10.0
            ):

                continue


        best = {

            "x":
                float(
                    point_x
                ),

            "y":
                float(
                    point_y
                ),

            "elevation":
                float(
                    elevation[
                        row,
                        column
                    ]
                ),

            "slope":
                float(
                    slope[
                        row,
                        column
                    ]
                ),

            "twi":
                float(
                    twi[
                        row,
                        column
                    ]
                ),

            "drainage_density":
                float(
                    drainage_density[
                        row,
                        column
                    ]
                ),

            "flow_accumulation":

                (
                    float(
                        flow_accumulation[
                            row,
                            column
                        ]
                    )

                    if np.isfinite(
                        flow_accumulation[
                            row,
                            column
                        ]
                    )

                    else
                    np.nan
                ),

            "site_score":
                float(
                    current_score
                ),

            "boundary_distance":
                distance_boundary,

            "stream_distance":
                distance_stream,

        }


        break


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if best is None:

        finite_indices = (
            np.flatnonzero(

                np.isfinite(
                    site_score.ravel()
                )

            )
        )


        if len(
            finite_indices
        ) == 0:

            return None


        flat_index = finite_indices[

            np.argmax(

                site_score
                .ravel()[
                    finite_indices
                ]

            )

        ]


        row, column = (
            np.unravel_index(

                flat_index,

                site_score.shape,

            )
        )


        point_x, point_y = xy(

            transform,

            row,

            column,

            offset="center",

        )


        point = Point(
            point_x,
            point_y,
        )


        best = {

            "x":
                float(
                    point_x
                ),

            "y":
                float(
                    point_y
                ),

            "elevation":
                float(
                    elevation[
                        row,
                        column
                    ]
                ),

            "slope":
                float(
                    slope[
                        row,
                        column
                    ]
                ),

            "twi":
                float(
                    twi[
                        row,
                        column
                    ]
                ),

            "drainage_density":
                float(
                    drainage_density[
                        row,
                        column
                    ]
                ),

            "flow_accumulation":

                (
                    float(
                        flow_accumulation[
                            row,
                            column
                        ]
                    )

                    if np.isfinite(
                        flow_accumulation[
                            row,
                            column
                        ]
                    )

                    else
                    np.nan
                ),

            "site_score":
                float(
                    site_score[
                        row,
                        column
                    ]
                ),

            "boundary_distance":
                float(

                    point.distance(
                        farmer_field.boundary
                    )

                ),

            "stream_distance":

                (
                    float(
                        point.distance(
                            stream_union
                        )
                    )

                    if (
                        stream_union
                        is not None
                    )

                    else
                    None
                ),

        }


    # --------------------------------------------------------
    # Convert to latitude / longitude
    # --------------------------------------------------------

    transformer = (
        Transformer.from_crs(

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


    lowest_lon, lowest_lat = (
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


    return {

        "recommended":
            best,

        "lowest": {

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
                        lowest_index
                    ]
                ),

        },

    }


# ============================================================
# 20. POND GEOMETRY
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

        2.0
        *
        side_slope
        *
        water_depth

    )


    water_top_length = (

        bottom_length

        +

        2.0
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


    water_top_area = (

        water_top_length
        *
        water_top_breadth

    )


    storage_volume = (

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


    excavation_depth = (

        water_depth
        +
        freeboard

    )


    excavation_top_breadth = (

        bottom_breadth

        +

        2.0
        *
        side_slope
        *
        excavation_depth

    )


    excavation_top_length = (

        bottom_length

        +

        2.0
        *
        side_slope
        *
        excavation_depth

    )


    footprint = (

        excavation_top_length
        *
        excavation_top_breadth

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
        2.0

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
        2.0

        *
        side_slant

    )


    lining_area = (

        bottom_area

        +

        2.0
        *
        long_side_area

        +

        2.0
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

        "water_top_length":
            float(
                water_top_length
            ),

        "water_top_breadth":
            float(
                water_top_breadth
            ),

        "top_length":
            float(
                excavation_top_length
            ),

        "top_breadth":
            float(
                excavation_top_breadth
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
                storage_volume
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


# ============================================================
# 21. SIZE FARM POND
# ============================================================

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
        maximum_land_fraction

    )


    # --------------------------------------------------------
    # Find maximum size allowed by field-area constraint
    # --------------------------------------------------------

    low = 0.5

    high = 120.0


    for _ in range(
        80
    ):

        middle = (

            low
            +
            high

        ) / 2.0


        geometry = (
            calculate_pond_geometry(

                middle,

                water_depth,

                freeboard,

                side_slope,

                length_breadth_ratio,

            )
        )


        if (

            geometry[
                "footprint"
            ]

            >
            maximum_footprint

        ):

            high = middle

        else:

            low = middle


    maximum_geometry = (
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

        maximum_geometry[
            "volume"
        ],

    )


    # --------------------------------------------------------
    # Solve breadth for target volume
    # --------------------------------------------------------

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

        ) / 2.0


        geometry = (
            calculate_pond_geometry(

                middle,

                water_depth,

                freeboard,

                side_slope,

                length_breadth_ratio,

            )
        )


        if (

            geometry[
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

        maximum_geometry[
            "volume"
        ]

    )


    result[
        "occupied_percent"
    ] = float(

        result[
            "footprint"
        ]

        /
        field_area_m2

        *
        100.0

    )


    result[
        "remaining_area_ha"
    ] = float(

        max(

            field_area_m2

            -

            result[
                "footprint"
            ],

            0.0,

        )

        /
        10000.0

    )


    return result


# ============================================================
# 22. THEMATIC DESCRIPTIONS
# ============================================================

def soil_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    lookup = {

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

    }


    return lookup.get(

        int(
            round(
                rank
            )
        ),

        "Unknown",

    )


def geology_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    lookup = {

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

    }


    return lookup.get(

        int(
            round(
                rank
            )
        ),

        "Other / unmatched geological class",

    )


def geomorphology_description(
    rank
):

    if not np.isfinite(
        rank
    ):

        return "Unknown"


    lookup = {

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
                "Moderately/highly "
                "dissected terrain"
            ),

        1:
            (
                "Scarp / Ridge / "
                "Low dissected upper plateau"
            ),

    }


    return lookup.get(

        int(
            round(
                rank
            )
        ),

        "Unknown",

    )


# ============================================================
# 23. SEEPAGE SCREENING
# ============================================================

def assess_seepage(
    soil_rank,
    geology_rank,
    geomorphology_rank,
    lineament_density,
):

    risk_score = 0

    reasons = []


    # Soil interpreted for pond retention,
    # not groundwater-potential ranking

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

                "Clayey or clay-loam soil context is comparatively favourable for pond-water retention."

            )


        elif soil == 2:

            risk_score += 1

            reasons.append(

                "Silty soil may allow moderate seepage and should be checked by a field permeability test."

            )


        elif soil == 3:

            risk_score += 2

            reasons.append(

                "Sandy clay loam indicates increased seepage potential."

            )


        elif soil == 5:

            risk_score += 2

            reasons.append(

                "Loam may permit greater seepage than clay-rich material."

            )


        else:

            risk_score += 1

            reasons.append(

                "Soil class requires field permeability verification."

            )


    if np.isfinite(
        lineament_density
    ):

        if (

            lineament_density
            >=
            0.55

        ):

            risk_score += 2

            reasons.append(

                "High lineament density indicates elevated fracture-related seepage concern."

            )


        elif (

            lineament_density
            >=
            0.19

        ):

            risk_score += 1

            reasons.append(

                "Moderate lineament density indicates some fracture-related seepage concern."

            )


        else:

            reasons.append(

                "Low local lineament density reduces fracture-related seepage concern."

            )


    if np.isfinite(
        geology_rank
    ):

        geology = int(
            round(
                geology_rank
            )
        )


        if geology == 5:

            risk_score += 1

            reasons.append(

                "Quaternary material can have variable permeability and should be checked at excavation depth."

            )


        elif geology in [
            2,
            3,
        ]:

            reasons.append(

                "Bedrock setting requires field inspection for weathering, joints and fractures."

            )


    if np.isfinite(
        geomorphology_rank
    ):

        geomorphology = int(
            round(
                geomorphology_rank
            )
        )


        if geomorphology in [
            1,
            2,
        ]:

            risk_score += 1

            reasons.append(

                "Dissected, ridge or scarp terrain requires additional stability and seepage verification."

            )


    if risk_score >= 4:

        risk = "HIGH"

        lining = (

            "Impermeable lining is strongly recommended. "
            "Use a properly prepared and compacted subgrade "
            "with a suitable geomembrane or approved pond-lining "
            "system after field permeability and geotechnical verification."

        )


    elif risk_score >= 2:

        risk = "MODERATE"

        lining = (

            "Lining should be considered. "
            "A compacted clay blanket may be suitable where good "
            "clay material is available. A suitable geomembrane "
            "should be considered where dependable water retention is critical."

        )


    else:

        risk = "LOW"

        lining = (

            "Compacted natural soil treatment may be adequate, "
            "but a field permeability test is still required before "
            "deciding that synthetic lining is unnecessary."

        )


    return {

        "risk":
            risk,

        "risk_score":
            risk_score,

        "reasons":
            reasons,

        "lining":
            lining,

    }


# ============================================================
# 24. FARMER INFORMATION
# ============================================================

with st.expander(
    "👨‍🌾 Farmer / Field Information",
    expanded=False,
):

    col1, col2 = st.columns(
        2
    )


    farmer_name = (
        col1.text_input(
            "Farmer Name"
        )
    )


    village = (
        col2.text_input(
            "Village / Locality"
        )
    )


# ============================================================
# 25. STEP 1 — GPS + FIELD DELINEATION
# ============================================================

st.header(
    "1️⃣ Locate and Draw Farmer Field"
)


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


# ============================================================
# 26. FIELD MAP
# ============================================================

field_map = folium.Map(

    location=
        MAP_CENTER,

    zoom_start=
        MAP_ZOOM,

    tiles=None,

    control_scale=True,

    prefer_canvas=True,

)


# ------------------------------------------------------------
# Satellite imagery = DEFAULT
# ------------------------------------------------------------

folium.TileLayer(

    tiles=(

        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}"

    ),

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


# ------------------------------------------------------------
# OpenStreetMap = OPTIONAL
# ------------------------------------------------------------

folium.TileLayer(

    tiles=
        "OpenStreetMap",

    name=
        "Road / Village Map",

    overlay=
        False,

    control=
        True,

    show=
        False,

).add_to(
    field_map
)


# ------------------------------------------------------------
# Ri Bhoi boundary
# ------------------------------------------------------------

if (
    boundary_wgs
    is not None
):

    folium.GeoJson(

        boundary_wgs.__geo_interface__,

        name=
            "Ri Bhoi District Boundary",

        style_function=lambda feature: {

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


# ------------------------------------------------------------
# Mobile GPS button
# ------------------------------------------------------------

LocateControl(

    auto_start=False,

    position="topleft",

    flyTo=True,

    keepCurrentZoomLevel=False,

    strings={

        "title":
            "Use my GPS location",

        "popup":
            "You are approximately here",

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


# ------------------------------------------------------------
# Previously drawn farmer field
# ------------------------------------------------------------

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

        style_function=lambda feature: {

            "color":
                "#00C853",

            "weight":
                5,

            "fillColor":
                "#69F0AE",

            "fillOpacity":
                0.30,

        },

    ).add_to(
        field_map
    )


# ------------------------------------------------------------
# Polygon drawing
# ------------------------------------------------------------

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


Fullscreen(
    position="topleft"
).add_to(
    field_map
)


folium.LayerControl(

    position="topright",

    collapsed=True,

).add_to(
    field_map
)


map_output = st_folium(

    field_map,

    height=570,

    use_container_width=True,

    key="geopond_field_map",

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

    new_field
    is not None

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


st.caption(

    "📱 If GPS does not work, allow Location permission for the browser "
    "and tap the GPS button again. Use Satellite Imagery and zoom to "
    "approximately level 17–20 for field-boundary identification."

)


field = (
    st.session_state[
        "field_geometry"
    ]
)


# ============================================================
# 27. FIELD INFORMATION + PRESENT LAND USE
# ============================================================

if field:

    field_area = (
        field_area_ha(
            field
        )
    )


    lulc_composition = (
        mapped_lulc_summary(
            field
        )
    )


    mapped_lulc = (

        lulc_composition[0][
            "LULC"
        ]

        if lulc_composition

        else "Unknown"

    )


    st.success(

        f"✅ Farmer field captured successfully: "
        f"{field_area:.3f} ha"

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Field Area",

        f"{field_area:.3f} ha",

    )


    col2.metric(

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

        in
        LAND_USE_OPTIONS

    ):

        default_lulc_index = (

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

        default_lulc_index = (

            LAND_USE_OPTIONS
            .index(
                mapped_lulc
            )

        )


    else:

        default_lulc_index = 0


    current_lulc = (
        st.selectbox(

            "🌾 Confirm Present Land Use",

            LAND_USE_OPTIONS,

            index=
                default_lulc_index,

            help=(

                "If GIS LULC differs from current field condition, "
                "select the present land use observed by the farmer "
                "or field investigator."

            ),

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

            f"GIS layer indicates **{mapped_lulc}**, "
            f"whereas current land use is confirmed as "
            f"**{current_lulc}**. GeoPond-DSS will retain the GIS "
            "class but use the confirmed present land use for planning."

        )


    if lulc_composition:

        with st.expander(
            "🗺️ LULC composition inside field"
        ):

            st.dataframe(

                pd.DataFrame(
                    lulc_composition
                ),

                use_container_width=True,

                hide_index=True,

            )


# ============================================================
# 28. STEP 2 — PLANNING SETTINGS
# ============================================================

if field:

    st.markdown("---")


    st.header(
        "2️⃣ Preliminary Pond Planning Settings"
    )


    st.caption(

        "These values are planning assumptions for GIS-based "
        "screening and may be adjusted before analysis."

    )


    with st.expander(
        "⚙️ Adjust planning assumptions",
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

                "Maximum field area occupied by pond (%)",

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

                "Side slope (H:1V)",

                min_value=1.0,

                max_value=2.5,

                value=1.5,

                step=0.25,

            )
        )


        length_breadth_ratio = (
            st.slider(

                "Bottom length : breadth ratio",

                min_value=1.0,

                max_value=2.0,

                value=1.5,

                step=0.1,

            )
        )


# ============================================================
# 29. STEP 3 — RUN DSS
# ============================================================

if field:

    st.markdown("---")


    st.header(
        "3️⃣ Run GeoPond-DSS"
    )


    if st.button(

        "💧 FIND BEST POND SITE & PRELIMINARY DESIGN",

        type="primary",

        use_container_width=True,

    ):

        with st.spinner(

            "GeoPond-DSS is analysing rainfall, runoff, terrain, "
            "drainage, pond site and preliminary design..."

        ):

            field_area = (
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


            mean_runoff_volume = (

                mean_runoff

                *
                field_area

                *
                10.0

                if np.isfinite(
                    mean_runoff
                )

                else
                np.nan

            )


            dependable_volume = (

                dependable_runoff

                *
                field_area

                *
                10.0

                if np.isfinite(
                    dependable_runoff
                )

                else
                np.nan

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

                available_water
                <=
                0

            ):

                st.error(

                    "Valid runoff data could not be obtained for this field."

                )

                st.stop()


            pond_design = (
                design_farm_pond(

                    available_water_m3=
                        available_water,

                    field_area_ha_value=
                        field_area,

                    storage_fraction=
                        storage_fraction_percent
                        /
                        100.0,

                    maximum_land_fraction=
                        maximum_land_percent
                        /
                        100.0,

                    water_depth=
                        water_depth,

                    freeboard=
                        freeboard,

                    side_slope=
                        side_slope,

                    length_breadth_ratio=
                        length_breadth_ratio,

                )
            )


            pond_half_diagonal = (

                0.5

                *

                math.sqrt(

                    pond_design[
                        "top_length"
                    ] ** 2

                    +

                    pond_design[
                        "top_breadth"
                    ] ** 2

                )

            )


            boundary_clearance = max(

                10.0,

                pond_half_diagonal
                +
                2.0,

            )


            site_result = (
                find_pond_site(

                    field,

                    boundary_clearance=
                        boundary_clearance,

                )
            )


            if (
                site_result
                is None
            ):

                st.error(

                    "GeoPond-DSS could not identify a valid terrain cell "
                    "inside the selected field."

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


            soil_rank = (
                sample_raster(

                    SOIL_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            geology_rank = (
                sample_raster(

                    GEOLOGY_RASTER,

                    site[
                        "x"
                    ],

                    site[
                        "y"
                    ],

                )
            )


            geomorphology_rank = (
                sample_raster(

                    GEOMORPH_RASTER,

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


            seepage = (
                assess_seepage(

                    soil_rank,

                    geology_rank,

                    geomorphology_rank,

                    lineament_density,

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

                "mean_runoff_volume":
                    mean_runoff_volume,

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
                    pond_design,

                "soil_rank":
                    soil_rank,

                "geology_rank":
                    geology_rank,

                "geomorphology_rank":
                    geomorphology_rank,

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
# 30. RESULTS
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


    seepage = (
        assessment[
            "seepage"
        ]
    )


    st.markdown("---")


    st.header(
        "✅ GeoPond-DSS Assessment Result"
    )


    agricultural_land = any(

        keyword
        in
        assessment[
            "current_lulc"
        ].lower()

        for keyword in [

            "crop",

            "plantation",

            "shifting",

        ]

    )


    if agricultural_land:

        st.success(

            "### Recommended intervention: Farm Pond\n"
            "The proposed pond centre represents the best screened "
            "GIS candidate within the farmer field using terrain, "
            "wetness, drainage and runoff-convergence information."

        )


    else:

        st.warning(

            "The confirmed land use is not clearly agricultural. "
            "The farm-pond result should therefore be treated as "
            "a preliminary scenario requiring additional field justification."

        )


    # ========================================================
    # WATER AVAILABILITY
    # ========================================================

    st.subheader(
        "🌧️ Water Availability"
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Field Area",

        f"{assessment['field_area']:.3f} ha",

    )


    col2.metric(

        "Spatial Annual Rainfall",

        (
            f"{assessment['rainfall']:.0f} mm/year"

            if np.isfinite(
                assessment[
                    "rainfall"
                ]
            )

            else
            "No data"
        ),

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Mean Annual Runoff",

        (
            f"{assessment['mean_runoff']:.0f} mm"

            if np.isfinite(
                assessment[
                    "mean_runoff"
                ]
            )

            else
            "No data"
        ),

    )


    col2.metric(

        "75% Dependable Runoff",

        (
            f"{assessment['dependable_runoff']:.0f} mm"

            if np.isfinite(
                assessment[
                    "dependable_runoff"
                ]
            )

            else
            "No data"
        ),

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Mean Annual Runoff Volume",

        (
            f"{assessment['mean_runoff_volume']:,.0f} m³/year"

            if np.isfinite(
                assessment[
                    "mean_runoff_volume"
                ]
            )

            else
            "No data"
        ),

    )


    col2.metric(

        "75% Dependable Water",

        (
            f"{assessment['dependable_volume']:,.0f} m³/year"

            if np.isfinite(
                assessment[
                    "dependable_volume"
                ]
            )

            else
            "No data"
        ),

    )


    st.caption(

        "Rainfall values retain the information content of the "
        "original rainfall dataset. Resampling to the GIS grid "
        "does not create true 30-m rainfall observations."

    )


    # ========================================================
    # LOCATION
    # ========================================================

    st.subheader(
        "📍 Recommended Pond Location"
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Latitude",

        f"{site['latitude']:.6f}",

    )


    col2.metric(

        "Longitude",

        f"{site['longitude']:.6f}",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Elevation",

        f"{site['elevation']:.1f} m",

    )


    col2.metric(

        "Local Slope",

        f"{site['slope']:.2f}°",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "TWI",

        f"{site['twi']:.2f}",

    )


    col2.metric(

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

            "Distance to Mapped Stream",

            f"{site['stream_distance']:.1f} m",

        )


    st.caption(

        "🔴 Red marker = recommended practical pond centre. "
        "🟠 Orange marker = absolute lowest DEM cell. "
        "The recommended location may differ because GeoPond-DSS "
        "also considers slope, wetness, runoff convergence, "
        "field-boundary clearance and drainage."

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

        zoom_start=18,

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

        attr=
            "Esri World Imagery",

        name=
            "Satellite Imagery",

        overlay=False,

        show=True,

        max_zoom=20,

    ).add_to(
        result_map
    )


    folium.TileLayer(

        "OpenStreetMap",

        name=
            "Road / Village Map",

        overlay=False,

        show=False,

    ).add_to(
        result_map
    )


    folium.GeoJson(

        field,

        name=
            "Farmer Field",

        style_function=lambda feature: {

            "color":
                "#00C853",

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


    pond_polygon_utm = box(

        site[
            "x"
        ]
        -
        pond[
            "top_length"
        ]
        /
        2.0,


        site[
            "y"
        ]
        -
        pond[
            "top_breadth"
        ]
        /
        2.0,


        site[
            "x"
        ]
        +
        pond[
            "top_length"
        ]
        /
        2.0,


        site[
            "y"
        ]
        +
        pond[
            "top_breadth"
        ]
        /
        2.0,

    )


    pond_polygon_wgs = (

        gpd.GeoSeries(

            [
                pond_polygon_utm
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
            pond_polygon_wgs
        ),

        name=
            "Preliminary Pond Footprint",

        style_function=lambda feature: {

            "color":
                "#D50000",

            "weight":
                3,

            "fillColor":
                "#FF5252",

            "fillOpacity":
                0.30,

        },

        tooltip=
            "Preliminary GeoPond-DSS pond footprint",

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

        tooltip=
            "Recommended GeoPond Site",

        popup=(

            "<b>GeoPond-DSS Recommended Site</b>"

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
            f"{pond['target_storage']:.0f} m³"

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

        tooltip=
            "Absolute Lowest DEM Point",

        popup=(

            "<b>Absolute Lowest DEM Point</b>"

            "<br>"

            f"Elevation: "
            f"{lowest['elevation']:.1f} m"

        ),

        icon=folium.Icon(
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


    folium.LayerControl(
        collapsed=True
    ).add_to(
        result_map
    )


    st_folium(

        result_map,

        height=570,

        use_container_width=True,

        key=
            "geopond_result_map",

    )


    # ========================================================
    # POND DESIGN
    # ========================================================

    st.subheader(
        "📐 Preliminary Pond Design"
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Proposed Storage",

        f"{pond['target_storage']:,.0f} m³",

    )


    col2.metric(

        "Water Depth",

        f"{pond['water_depth']:.2f} m",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Bottom Length",

        f"{pond['bottom_length']:.1f} m",

    )


    col2.metric(

        "Bottom Breadth",

        f"{pond['bottom_breadth']:.1f} m",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Top Length",

        f"{pond['top_length']:.1f} m",

    )


    col2.metric(

        "Top Breadth",

        f"{pond['top_breadth']:.1f} m",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Excavation Depth",

        f"{pond['excavation_depth']:.2f} m",

    )


    col2.metric(

        "Freeboard",

        f"{pond['freeboard']:.2f} m",

    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Pond Footprint",

        f"{pond['footprint']:,.0f} m²",

    )


    col2.metric(

        "Field Area Occupied",

        f"{pond['occupied_percent']:.1f}%",

    )


    st.metric(

        "Remaining Cultivable / Available Land",

        f"{pond['remaining_area_ha']:.3f} ha",

    )


    if (

        pond[
            "requested_storage"
        ]

        >
        pond[
            "target_storage"
        ]

        +
        0.01

    ):

        st.warning(

            "The requested one-time storage would require more land "
            "than the selected maximum field-area allowance. "
            "GeoPond-DSS has therefore reduced the preliminary "
            "pond capacity to fit the land constraint."

        )


    # ========================================================
    # SEEPAGE AND LINING
    # ========================================================

    st.subheader(
        "🪨 Seepage & Pond-Lining Screening"
    )


    if (
        seepage[
            "risk"
        ]
        ==
        "HIGH"
    ):

        st.error(
            "Seepage Risk: HIGH"
        )


    elif (
        seepage[
            "risk"
        ]
        ==
        "MODERATE"
    ):

        st.warning(
            "Seepage Risk: MODERATE"
        )


    else:

        st.success(
            "Seepage Risk: LOW"
        )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Soil Context",

        soil_description(
            assessment[
                "soil_rank"
            ]
        ),

    )


    col2.metric(

        "Geology Context",

        geology_description(
            assessment[
                "geology_rank"
            ]
        ),

    )


    st.write(

        "**Geomorphology:**",

        geomorphology_description(
            assessment[
                "geomorphology_rank"
            ]
        ),

    )


    st.write(

        "**Lineament Density:**",

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


    st.markdown(
        "**Screening basis:**"
    )


    for reason in (
        seepage[
            "reasons"
        ]
    ):

        st.write(
            f"• {reason}"
        )


    st.info(
        seepage[
            "lining"
        ]
    )


    col1, col2 = st.columns(
        2
    )


    col1.metric(

        "Estimated Wetted Lining Area",

        f"{pond['lining_area']:,.0f} m²",

    )


    col2.metric(

        "Indicative Procurement Area (+10%)",

        f"{pond['liner_purchase_area']:,.0f} m²",

    )


    # ========================================================
    # TECHNICAL SITE INFORMATION
    # ========================================================

    with st.expander(
        "🔬 Technical Site Information"
    ):

        st.write(
            "**Farmer:**",
            assessment[
                "farmer"
            ]
            or
            "Not entered",
        )


        st.write(
            "**Village / Locality:**",
            assessment[
                "village"
            ]
            or
            "Not entered",
        )


        st.write(
            "**Current land use:**",
            assessment[
                "current_lulc"
            ],
        )


        st.write(
            "**GIS-mapped LULC:**",
            assessment[
                "mapped_lulc"
            ],
        )


        st.write(

            "**CN-II:**",

            (
                round(
                    assessment[
                        "cn"
                    ],
                    1,
                )

                if np.isfinite(
                    assessment[
                        "cn"
                    ]
                )

                else
                "No data"
            ),

        )


        st.write(

            "**Mean runoff coefficient:**",

            (
                round(
                    assessment[
                        "runoff_coefficient"
                    ],
                    3,
                )

                if np.isfinite(
                    assessment[
                        "runoff_coefficient"
                    ]
                )

                else
                "No data"
            ),

        )


        st.write(

            "**Drainage density at pond site:**",

            round(
                site[
                    "drainage_density"
                ],
                3,
            ),

        )


        st.write(

            "**Flow accumulation at pond site:**",

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

                else
                "No data"
            ),

        )


        st.write(

            "**GeoPond site score:**",

            round(
                site[
                    "site_score"
                ],
                3,
            ),

        )


        st.write(

            "**Nearest mapped stream to field:**",

            (
                f"{assessment['nearest_stream']:.0f} m"

                if (
                    assessment[
                        "nearest_stream"
                    ]
                    is not None
                )

                else
                "No data"
            ),

        )


        st.write(

            "**Nearest MGNREGA RWH structure:**",

            (
                f"{assessment['nearest_mgnrega']:.0f} m"

                if (
                    assessment[
                        "nearest_mgnrega"
                    ]
                    is not None
                )

                else
                "Optional layer unavailable"
            ),

        )


        st.write(

            "**Nearest Final-94 RWH site:**",

            (
                f"{assessment['nearest_final94']:.0f} m"

                if (
                    assessment[
                        "nearest_final94"
                    ]
                    is not None
                )

                else
                "Optional layer unavailable"
            ),

        )


        st.write(

            "**Nearest published RWH site:**",

            (
                f"{assessment['nearest_published']:.0f} m"

                if (
                    assessment[
                        "nearest_published"
                    ]
                    is not None
                )

                else
                "Optional layer unavailable"
            ),

        )


    # ========================================================
    # MANAGEMENT GUIDANCE
    # ========================================================

    st.subheader(
        "🌱 Construction & Management Guidance"
    )


    st.markdown(
        """<div class="result-card">
<b>Farm Pond Planning Guidance</b><br><br>
• Confirm the proposed GIS site through field survey and local micro-topography.<br>
• Verify ownership and maintain adequate clearance from the farmer's boundary.<br>
• Provide an inlet sediment/silt trap before runoff enters the pond.<br>
• Do not obstruct a natural stream for an off-stream farm pond.<br>
• Provide a safe overflow/spill arrangement for excess monsoon runoff.<br>
• Stabilize exposed pond bunds and disturbed soil with suitable vegetation.<br>
• Periodically remove accumulated sediment from the inlet and pond.<br>
• Verify soil permeability before deciding whether lining is required.<br>
• Protect any geomembrane from puncture during installation and operation.<br>
• Use harvested water efficiently for protective or supplemental irrigation.<br>
</div>""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    output_record = {

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

        "GIS_LULC":
            assessment[
                "mapped_lulc"
            ],

        "Confirmed_Land_Use":
            assessment[
                "current_lulc"
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

        "Drainage_Density":
            site[
                "drainage_density"
            ],

        "Flow_Accumulation":
            site[
                "flow_accumulation"
            ],

        "GeoPond_Site_Score":
            site[
                "site_score"
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

        "Remaining_Land_ha":
            pond[
                "remaining_area_ha"
            ],

        "Soil_Context":
            soil_description(
                assessment[
                    "soil_rank"
                ]
            ),

        "Geology_Context":
            geology_description(
                assessment[
                    "geology_rank"
                ]
            ),

        "Geomorphology_Context":
            geomorphology_description(
                assessment[
                    "geomorphology_rank"
                ]
            ),

        "Lineament_Density":
            assessment[
                "lineament_density"
            ],

        "Seepage_Risk":
            seepage[
                "risk"
            ],

        "Wetted_Lining_Area_m2":
            pond[
                "lining_area"
            ],

        "Liner_Procurement_Area_m2":
            pond[
                "liner_purchase_area"
            ],

        "Lining_Recommendation":
            seepage[
                "lining"
            ],

    }


    csv_data = (

        pd.DataFrame(
            [
                output_record
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

        use_container_width=True,

    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
**Important engineering note:** GeoPond-DSS provides preliminary GIS-based planning and screening.

The proposed coordinates represent a **30-m GIS candidate zone**, not a surveyed construction peg. Final pond location must be verified for micro-topography, farmer ownership boundary, access, natural drainage, foundation conditions, utilities and environmental constraints.

The calculated pond dimensions are preliminary planning dimensions. Final excavation, embankment, inlet, outlet, spillway, freeboard, lining and structural design require field survey and engineering verification.
"""
    )


# ============================================================
# 31. RESET
# ============================================================

st.markdown("---")


if st.button(

    "🔄 Start New Farmer Assessment",

    use_container_width=True,

):

    reset_assessment()

    st.rerun()
