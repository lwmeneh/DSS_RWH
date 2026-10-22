from pathlib import Path
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

from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

from shapely.geometry import shape, mapping, Point, box
from shapely.ops import unary_union
from pyproj import Transformer


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="ICAR Ri Bhoi RWH GIS DSS",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "cache"

GEOLOGY_RASTER = CACHE_DIR / "01_Geology.tif"
GEOMORPH_RASTER = CACHE_DIR / "02_Geomorphology.tif"
SLOPE_RASTER = CACHE_DIR / "03_Slope.tif"
RAINFALL_RASTER = CACHE_DIR / "04_Rainfall.tif"
ELEVATION_RASTER = CACHE_DIR / "05_Elevation.tif"
DRAINAGE_DENSITY_RASTER = CACHE_DIR / "06_DrainageDensity.tif"
LINEAMENT_DENSITY_RASTER = CACHE_DIR / "08_LineamentDensity.tif"
SOIL_RASTER = CACHE_DIR / "09_Soil.tif"
TWI_RASTER = CACHE_DIR / "10_TWI.tif"

FLOW_ACCUMULATION_RASTER = CACHE_DIR / "flow_accumulation.tif"

CN_RASTER = CACHE_DIR / "CN_II.tif"
MEAN_RUNOFF_RASTER = CACHE_DIR / "mean_runoff_mm.tif"
DEPENDABLE_RUNOFF_RASTER = CACHE_DIR / "dependable_runoff_mm.tif"
RUNOFF_COEFF_RASTER = CACHE_DIR / "runoff_coefficient.tif"

BOUNDARY_GPKG = CACHE_DIR / "boundary_utm46.gpkg"
STREAMS_GPKG = CACHE_DIR / "streams_utm46.gpkg"

LULC_GPKG = CACHE_DIR / "lulc_2021_10k_clean_utm46.gpkg"
MGNREGA_GPKG = CACHE_DIR / "mgnrega_utm46.gpkg"
FINAL94_GPKG = CACHE_DIR / "final94_utm46.gpkg"
PUBLISHED_GPKG = CACHE_DIR / "published_sites_utm46.gpkg"

WORKING_CRS = "EPSG:32646"

MAP_CENTER = [25.88, 91.88]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 0.5rem;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
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
    background: white;
    border: 1px solid #d8e5e8;
    border-radius: 12px;
    padding: 10px;
}

.info-card {
    background: white;
    border: 1px solid #d8e5e8;
    border-left: 5px solid #159957;
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
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
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
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

<div style="font-size:14px;font-weight:650;">
INDIAN COUNCIL OF AGRICULTURAL RESEARCH
</div>

<div style="font-size:26px;font-weight:800;">
ICAR Research Complex for NEH Region
</div>

<div style="font-size:17px;">
Umiam, Meghalaya
</div>

<hr style="border:none;border-top:1px solid rgba(255,255,255,0.35);">

<div style="font-size:23px;font-weight:750;">
💧 Rainwater Harvesting GIS Decision Support System
</div>

<div style="font-size:15px;">
Automatic Farm Pond Siting and Preliminary Design
<br>
Ri Bhoi District, Meghalaya
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# REQUIRED FILES
# ============================================================

required_files = [

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
    RUNOFF_COEFF_RASTER,

    BOUNDARY_GPKG,
    STREAMS_GPKG,
]


missing = [

    str(
        p.relative_to(BASE_DIR)
    )

    for p in required_files

    if not p.exists()

]


if missing:

    st.error(
        "Required GIS files are missing."
    )

    st.write(
        "Copy these files into data/cache:"
    )

    st.code(
        "\n".join(missing)
    )

    st.stop()


# ============================================================
# LOAD VECTOR
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vector(path_string):

    path = Path(path_string)

    if not path.exists():

        return None

    try:

        gdf = gpd.read_file(path)

        if gdf.crs is None:

            gdf = gdf.set_crs(
                WORKING_CRS
            )

        return gdf

    except Exception:

        return None


boundary = load_vector(
    str(BOUNDARY_GPKG)
)

streams = load_vector(
    str(STREAMS_GPKG)
)

lulc = load_vector(
    str(LULC_GPKG)
)

mgnrega = load_vector(
    str(MGNREGA_GPKG)
)

final94 = load_vector(
    str(FINAL94_GPKG)
)

published = load_vector(
    str(PUBLISHED_GPKG)
)


boundary_wgs = None

if boundary is not None and not boundary.empty:

    boundary_wgs = boundary.to_crs(
        "EPSG:4326"
    )


# ============================================================
# STATE
# ============================================================

if "field_geometry" not in st.session_state:

    st.session_state.field_geometry = None


if "assessment" not in st.session_state:

    st.session_state.assessment = None


if "current_lulc" not in st.session_state:

    st.session_state.current_lulc = None


# ============================================================
# GEOMETRY
# ============================================================

def geometry_gdf(
    geometry,
    crs="EPSG:4326"
):

    return gpd.GeoDataFrame(

        {"id": [1]},

        geometry=[
            shape(geometry)
        ],

        crs=crs,
    )


def to_utm(
    geometry
):

    return (

        geometry_gdf(
            geometry
        )

        .to_crs(
            WORKING_CRS
        )

        .geometry.iloc[0]

    )


def area_ha(
    geometry
):

    return (
        to_utm(
            geometry
        ).area
        /
        10000.0
    )


def last_polygon(
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


        geom = drawing.get(
            "geometry",
            {}
        )


        if geom.get(
            "type"
        ) in [
            "Polygon",
            "MultiPolygon"
        ]:

            polygon = geom


    return polygon


# ============================================================
# RASTER STATS
# ============================================================

def raster_stats(
    raster_path,
    field_geometry
):

    polygon = geometry_gdf(
        field_geometry
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


        values = np.array([])


        try:

            data, _ = rio_mask(

                src,

                [mapping(geom)],

                crop=True,

                indexes=1,

                filled=False,

                all_touched=True,

            )


            if np.ma.isMaskedArray(data):

                values = (
                    data.compressed()
                )

            else:

                values = (
                    data.reshape(-1)
                )


            values = values[
                np.isfinite(values)
            ]


        except Exception:

            values = np.array([])


        # small-field fallback
        if len(values) == 0:

            c = geom.centroid

            try:

                sample = next(

                    src.sample(
                        [(c.x, c.y)],
                        indexes=1,
                        masked=True,
                    )

                )


                values = np.asarray(
                    sample
                ).reshape(-1)


                values = values[
                    np.isfinite(values)
                ]


            except Exception:

                pass


        if len(values) == 0:

            return np.nan


        return float(
            np.mean(values)
        )


# ============================================================
# POINT SAMPLE
# ============================================================

def sample_point(
    raster_path,
    x,
    y
):

    with rasterio.open(
        raster_path
    ) as src:

        transformer = (
            Transformer.from_crs(

                WORKING_CRS,
                src.crs,

                always_xy=True,

            )
        )


        rx, ry = transformer.transform(
            x,
            y
        )


        try:

            value = next(

                src.sample(
                    [(rx, ry)],
                    indexes=1,
                    masked=True,
                )

            )[0]


            if np.ma.is_masked(value):

                return np.nan


            value = float(value)


            return (
                value
                if np.isfinite(value)
                else np.nan
            )


        except Exception:

            return np.nan


# ============================================================
# REFERENCE GRID
# ============================================================

def field_reference_grid(
    field_geometry
):

    polygon = geometry_gdf(
        field_geometry
    )


    with rasterio.open(
        ELEVATION_RASTER
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

            [mapping(geom)],

            crop=True,

            indexes=1,

            filled=False,

            all_touched=True,

        )


        if np.ma.isMaskedArray(data):

            array = (

                data

                .filled(np.nan)

                .astype(float)

            )

            valid = ~data.mask

        else:

            array = data.astype(float)

            valid = np.isfinite(array)


        return (
            array,
            valid,
            transform,
            src.crs
        )


# ============================================================
# ALIGN RASTER
# ============================================================

def aligned_raster(
    raster_path,
    reference_shape,
    transform,
    crs
):

    destination = np.full(

        reference_shape,

        np.nan,

        dtype=float

    )


    with rasterio.open(
        raster_path
    ) as src:

        source = (
            src.read(1)
            .astype(float)
        )


        if src.nodata is not None:

            source[
                source == src.nodata
            ] = np.nan


        reproject(

            source=source,

            destination=destination,

            src_transform=src.transform,

            src_crs=src.crs,

            dst_transform=transform,

            dst_crs=crs,

            src_nodata=np.nan,

            dst_nodata=np.nan,

            resampling=
                Resampling.bilinear,

        )


    return destination


# ============================================================
# NORMALIZE
# ============================================================

def normalize(
    array,
    valid
):

    values = array[
        valid
    ]


    values = values[
        np.isfinite(values)
    ]


    if len(values) == 0:

        return np.zeros_like(
            array
        )


    low = np.percentile(
        values,
        5
    )


    high = np.percentile(
        values,
        95
    )


    if high <= low:

        return np.zeros_like(
            array
        )


    return np.clip(

        (
            array - low
        )
        /
        (
            high - low
        ),

        0,
        1

    )


# ============================================================
# LULC
# ============================================================

def get_lulc(
    field_geometry
):

    if (
        lulc is None
        or lulc.empty
        or "LULC_2022"
        not in lulc.columns
    ):

        return []


    geom = (

        geometry_gdf(
            field_geometry
        )

        .to_crs(
            lulc.crs
        )

        .geometry.iloc[0]

    )


    ids = list(

        lulc.sindex.intersection(
            geom.bounds
        )

    )


    if not ids:

        return []


    data = lulc.iloc[
        ids
    ].copy()


    data = data[
        data.intersects(geom)
    ].copy()


    if data.empty:

        return []


    data[
        "clip"
    ] = (

        data.geometry.intersection(
            geom
        )

    )


    data[
        "area"
    ] = (

        data["clip"].area

    )


    grouped = (

        data.groupby(
            "LULC_2022"
        )["area"]

        .sum()

        .sort_values(
            ascending=False
        )

    )


    total = grouped.sum()


    return [

        {
            "name": name,

            "area_ha":
                value
                /
                10000,

            "percent":
                value
                /
                total
                *
                100
        }

        for name, value
        in grouped.items()

    ]


# ============================================================
# NEAREST DISTANCE
# ============================================================

def nearest_distance(
    field_geometry,
    layer
):

    if layer is None or layer.empty:

        return None


    geom = (

        geometry_gdf(
            field_geometry
        )

        .to_crs(
            layer.crs
        )

        .geometry.iloc[0]

    )


    distances = (
        layer.geometry.distance(
            geom
        )
    )


    return float(
        distances.min()
    )


# ============================================================
# FARM POND SITE
# ============================================================

def find_pond_site(
    field_geometry,
    boundary_clearance=10
):

    elevation, mask, transform, crs = (
        field_reference_grid(
            field_geometry
        )
    )


    shape_ref = (
        elevation.shape
    )


    slope = aligned_raster(
        SLOPE_RASTER,
        shape_ref,
        transform,
        crs
    )


    twi = aligned_raster(
        TWI_RASTER,
        shape_ref,
        transform,
        crs
    )


    drainage = aligned_raster(
        DRAINAGE_DENSITY_RASTER,
        shape_ref,
        transform,
        crs
    )


    flow = aligned_raster(
        FLOW_ACCUMULATION_RASTER,
        shape_ref,
        transform,
        crs
    )


    valid = (

        mask

        & np.isfinite(
            elevation
        )

        & np.isfinite(
            slope
        )

        & np.isfinite(
            twi
        )

        & np.isfinite(
            drainage
        )

    )


    if valid.sum() == 0:

        return None


    # absolute minimum elevation
    temp = np.where(
        valid,
        elevation,
        np.inf
    )


    minimum_index = np.unravel_index(

        np.argmin(temp),

        temp.shape

    )


    min_x, min_y = xy(

        transform,

        minimum_index[0],

        minimum_index[1],

        offset="center"

    )


    # normalized criteria

    elev_norm = normalize(
        elevation,
        valid
    )


    slope_norm = normalize(
        slope,
        valid
    )


    twi_norm = normalize(
        twi,
        valid
    )


    drainage_norm = normalize(
        drainage,
        valid
    )


    score = (

        0.45
        *
        (
            1 - elev_norm
        )

        +

        0.25
        *
        twi_norm

        +

        0.15
        *
        (
            1 - slope_norm
        )

        +

        0.10
        *
        drainage_norm

    )


    valid_flow = (

        valid

        & np.isfinite(
            flow
        )

    )


    if valid_flow.sum() > 0:

        flow_norm = normalize(

            np.log1p(
                np.maximum(
                    flow,
                    0
                )
            ),

            valid_flow

        )


        score += (
            0.05
            *
            flow_norm
        )


    # prefer <= 8 degrees

    preferred = (

        valid

        & (
            slope <= 8
        )

    )


    if preferred.sum() == 0:

        preferred = (

            valid

            & (
                slope <= 15
            )

        )


    if preferred.sum() == 0:

        preferred = valid


    score = np.where(

        preferred,

        score,

        -np.inf

    )


    field_utm = (

        geometry_gdf(
            field_geometry
        )

        .to_crs(
            crs
        )

        .geometry.iloc[0]

    )


    # stream union

    stream_union = None


    if streams is not None:

        stream_local = streams.to_crs(
            crs
        )


        search_area = field_utm.buffer(
            1000
        )


        ids = list(

            stream_local
            .sindex
            .intersection(
                search_area.bounds
            )

        )


        if ids:

            subset = (
                stream_local
                .iloc[ids]
            )


            subset = subset[
                subset.intersects(
                    search_area
                )
            ]


            if not subset.empty:

                stream_union = (
                    unary_union(
                        subset.geometry.tolist()
                    )
                )


    ordering = (
        np.argsort(
            score.ravel()
        )[::-1]
    )


    best = None


    for index in ordering[:1000]:

        row, column = (
            np.unravel_index(
                index,
                score.shape
            )
        )


        if not np.isfinite(
            score[
                row,
                column
            ]
        ):

            continue


        px, py = xy(

            transform,

            row,
            column,

            offset="center"

        )


        point = Point(
            px,
            py
        )


        boundary_distance = (
            point.distance(
                field_utm.boundary
            )
        )


        stream_distance = None


        if stream_union is not None:

            stream_distance = (
                point.distance(
                    stream_union
                )
            )


        if (
            boundary_distance
            <
            boundary_clearance
        ):

            continue


        if (
            stream_distance
            is not None
            and
            stream_distance
            <
            10
        ):

            continue


        best = {

            "x": px,

            "y": py,

            "elevation":
                elevation[
                    row,
                    column
                ],

            "slope":
                slope[
                    row,
                    column
                ],

            "twi":
                twi[
                    row,
                    column
                ],

            "drainage_density":
                drainage[
                    row,
                    column
                ],

            "flow_accumulation":
                flow[
                    row,
                    column
                ],

            "score":
                score[
                    row,
                    column
                ],

            "boundary_distance":
                boundary_distance,

            "stream_distance":
                stream_distance,

        }


        break


    if best is None:

        return None


    transformer = (
        Transformer.from_crs(

            crs,

            "EPSG:4326",

            always_xy=True

        )
    )


    lon, lat = transformer.transform(

        best["x"],
        best["y"]

    )


    min_lon, min_lat = (
        transformer.transform(
            min_x,
            min_y
        )
    )


    best[
        "longitude"
    ] = lon


    best[
        "latitude"
    ] = lat


    return {

        "recommended":
            best,

        "lowest": {

            "longitude":
                min_lon,

            "latitude":
                min_lat,

            "elevation":
                elevation[
                    minimum_index
                ]

        }

    }


# ============================================================
# POND GEOMETRY
# ============================================================

def pond_geometry(
    breadth,
    water_depth,
    freeboard,
    side_slope,
    ratio
):

    length = (
        breadth
        *
        ratio
    )


    top_breadth = (

        breadth

        +

        2
        *
        side_slope
        *
        water_depth

    )


    top_length = (

        length

        +

        2
        *
        side_slope
        *
        water_depth

    )


    bottom_area = (
        length
        *
        breadth
    )


    top_area = (
        top_length
        *
        top_breadth
    )


    volume = (

        water_depth

        /
        3

        *

        (
            bottom_area
            +
            top_area
            +
            math.sqrt(
                bottom_area
                *
                top_area
            )
        )

    )


    total_depth = (
        water_depth
        +
        freeboard
    )


    excavation_length = (

        length

        +

        2
        *
        side_slope
        *
        total_depth

    )


    excavation_breadth = (

        breadth

        +

        2
        *
        side_slope
        *
        total_depth

    )


    footprint = (
        excavation_length
        *
        excavation_breadth
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


    long_side = (

        (
            length
            +
            top_length
        )

        /
        2

        *
        slant

    )


    short_side = (

        (
            breadth
            +
            top_breadth
        )

        /
        2

        *
        slant

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
            length,

        "bottom_breadth":
            breadth,

        "top_length":
            excavation_length,

        "top_breadth":
            excavation_breadth,

        "water_depth":
            water_depth,

        "freeboard":
            freeboard,

        "total_depth":
            total_depth,

        "volume":
            volume,

        "footprint":
            footprint,

        "lining_area":
            lining_area,

        "liner_purchase":
            lining_area
            *
            1.10

    }


# ============================================================
# SIZE POND
# ============================================================

def design_pond(
    water_available,
    farm_area,
    storage_fraction,
    maximum_land_fraction,
    water_depth,
    freeboard,
    side_slope,
    ratio
):

    field_m2 = (
        farm_area
        *
        10000
    )


    requested = (
        water_available
        *
        storage_fraction
    )


    maximum_area = (
        field_m2
        *
        maximum_land_fraction
    )


    low = 0.5
    high = 100


    for _ in range(80):

        middle = (
            low + high
        ) / 2


        geometry = pond_geometry(

            middle,

            water_depth,

            freeboard,

            side_slope,

            ratio

        )


        if (
            geometry[
                "footprint"
            ]
            >
            maximum_area
        ):

            high = middle

        else:

            low = middle


    maximum_geometry = (
        pond_geometry(

            low,

            water_depth,

            freeboard,

            side_slope,

            ratio

        )
    )


    target = min(

        requested,

        maximum_geometry[
            "volume"
        ]

    )


    lower = 0.5
    upper = low


    for _ in range(80):

        middle = (
            lower
            +
            upper
        ) / 2


        geometry = pond_geometry(

            middle,

            water_depth,

            freeboard,

            side_slope,

            ratio

        )


        if (
            geometry[
                "volume"
            ]
            >
            target
        ):

            upper = middle

        else:

            lower = middle


    result = pond_geometry(

        lower,

        water_depth,

        freeboard,

        side_slope,

        ratio

    )


    result[
        "target_storage"
    ] = target


    result[
        "requested_storage"
    ] = requested


    result[
        "occupied_percent"
    ] = (

        result[
            "footprint"
        ]

        /
        field_m2

        *
        100

    )


    result[
        "remaining_area_ha"
    ] = (

        (
            field_m2

            -
            result[
                "footprint"
            ]
        )

        /
        10000

    )


    return result


# ============================================================
# SOIL / GEOLOGY DESCRIPTION
# ============================================================

def soil_description(
    rank
):

    if not np.isfinite(rank):

        return "Unknown"


    return {

        1:
            "Clay / Clayey / Clay loam",

        2:
            "Silt clay loam / Silt loam",

        3:
            "Sandy clay loam",

        4:
            "Miscellaneous",

        5:
            "Loam"

    }.get(
        int(round(rank)),
        "Unknown"
    )


def geology_description(
    rank
):

    if not np.isfinite(rank):

        return "Unknown"


    return {

        5:
            "Quaternary",

        3:
            "Umsning Schist / Assam-Meghalaya Gneissic",

        2:
            "Shillong Group / Kyrdem"

    }.get(
        int(round(rank)),
        "Other"
    )


# ============================================================
# SEEPAGE
# ============================================================

def seepage_risk(
    soil_rank,
    geology_rank,
    lineament_density
):

    risk_score = 0

    reasons = []


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
                "Clayey soil indicates relatively favourable water retention."
            )


        elif soil == 2:

            risk_score += 1

            reasons.append(
                "Silty soil indicates moderate seepage potential."
            )


        elif soil == 3:

            risk_score += 2

            reasons.append(
                "Sandy clay loam may permit increased seepage."
            )


        elif soil == 5:

            risk_score += 2

            reasons.append(
                "Loam may permit greater seepage than clayey soil."
            )


    if np.isfinite(
        lineament_density
    ):

        if lineament_density >= 0.55:

            risk_score += 2

            reasons.append(
                "High lineament density increases fracture-related seepage risk."
            )


        elif lineament_density >= 0.19:

            risk_score += 1

            reasons.append(
                "Moderate lineament density indicates some fracture-related seepage risk."
            )


    if np.isfinite(
        geology_rank
    ):

        if int(
            round(
                geology_rank
            )
        ) == 5:

            risk_score += 1

            reasons.append(
                "Quaternary material may have variable permeability."
            )


    if risk_score >= 4:

        risk = "HIGH"

        recommendation = (
            "Impermeable lining is recommended after field permeability "
            "and geotechnical verification. Consider an appropriate "
            "geomembrane or engineered pond-lining system."
        )


    elif risk_score >= 2:

        risk = "MODERATE"

        recommendation = (
            "Lining should be considered. Compacted clay may be suitable "
            "where appropriate material is available; otherwise consider "
            "a suitable geomembrane."
        )


    else:

        risk = "LOW"

        recommendation = (
            "Compacted natural soil may be adequate, subject to a field "
            "permeability test before construction."
        )


    return (
        risk,
        reasons,
        recommendation
    )


# ============================================================
# FARMER
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
# DRAW MAP
# ============================================================

st.header(
    "1️⃣ Draw Farmer Field"
)


st.info(
    "Use the polygon tool ⬠ and trace the farmer's actual field boundary."
)


field_map = folium.Map(

    location=
        MAP_CENTER,

    zoom_start=10,

    tiles=None,

    control_scale=True

)


folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}"
    ),

    attr="Esri",

    name="Satellite"

).add_to(
    field_map
)


folium.TileLayer(

    "OpenStreetMap",

    name="Street"

).add_to(
    field_map
)


if boundary_wgs is not None:

    folium.GeoJson(

        boundary_wgs.__geo_interface__,

        style_function=lambda x: {

            "color":
                "yellow",

            "weight":
                2,

            "fillOpacity":
                0

        }

    ).add_to(
        field_map
    )


if st.session_state.field_geometry:

    folium.GeoJson(

        st.session_state.field_geometry,

        style_function=lambda x: {

            "color":
                "#00C853",

            "weight":
                5,

            "fillColor":
                "#69F0AE",

            "fillOpacity":
                0.25

        }

    ).add_to(
        field_map
    )


Draw(

    export=False,

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
                True

        }

    },

    edit_options={

        "edit":
            True,

        "remove":
            True

    }

).add_to(
    field_map
)


Fullscreen().add_to(
    field_map
)


map_output = st_folium(

    field_map,

    height=520,

    use_container_width=True,

    key="field_map",

    returned_objects=[
        "all_drawings"
    ]

)


new_field = last_polygon(
    map_output
)


if (
    new_field is not None
    and
    new_field
    !=
    st.session_state.field_geometry
):

    st.session_state.field_geometry = (
        new_field
    )

    st.session_state.assessment = (
        None
    )

    st.rerun()


field = (
    st.session_state.field_geometry
)


# ============================================================
# FIELD / LULC
# ============================================================

if field:

    farm_area = area_ha(
        field
    )


    lulc_data = get_lulc(
        field
    )


    mapped_lulc = (

        lulc_data[0]["name"]

        if lulc_data

        else "Unknown"

    )


    st.success(
        f"Field captured: {farm_area:.3f} ha"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Area",
        f"{farm_area:.3f} ha"
    )


    c2.metric(
        "GIS LULC",
        mapped_lulc
    )


    land_options = [

        "Crop land",

        "Plantation",

        "Shifting Cultivation",

        "Scrub land",

        "Forest Open",

        "Forest Moderately Dense",

        "Forest Dense",

        "Built up",

        "Waterbody",

        "River/Stream"

    ]


    default = (

        land_options.index(
            mapped_lulc
        )

        if mapped_lulc
        in land_options

        else 0

    )


    current_lulc = st.selectbox(

        "Confirm current land use",

        land_options,

        index=default

    )


    st.session_state.current_lulc = (
        current_lulc
    )


# ============================================================
# SETTINGS
# ============================================================

if field:

    st.header(
        "2️⃣ Pond Planning Settings"
    )


    storage_percent = st.slider(

        "Storage as % of dependable annual runoff",

        10,
        60,
        30,
        5

    )


    land_percent = st.slider(

        "Maximum field occupied by pond (%)",

        5,
        20,
        10

    )


    water_depth = st.slider(

        "Water depth (m)",

        1.5,
        3.5,
        2.5,
        0.25

    )


    freeboard = st.slider(

        "Freeboard (m)",

        0.3,
        0.75,
        0.5,
        0.05

    )


    side_slope = st.slider(

        "Side slope H:V",

        1.0,
        2.5,
        1.5,
        0.25

    )


# ============================================================
# ANALYSE
# ============================================================

if field:

    if st.button(

        "💧 FIND POND SITE AND DESIGN",

        type="primary"

    ):

        with st.spinner(
            "Analysing field..."
        ):

            farm_area = area_ha(
                field
            )


            rainfall = raster_stats(
                RAINFALL_RASTER,
                field
            )


            runoff = raster_stats(
                MEAN_RUNOFF_RASTER,
                field
            )


            dependable = raster_stats(
                DEPENDABLE_RUNOFF_RASTER,
                field
            )


            cn = raster_stats(
                CN_RASTER,
                field
            )


            runoff_coeff = raster_stats(
                RUNOFF_COEFF_RASTER,
                field
            )


            dependable_volume = (

                dependable

                *
                farm_area

                *
                10

            )


            design = design_pond(

                dependable_volume,

                farm_area,

                storage_percent
                /
                100,

                land_percent
                /
                100,

                water_depth,

                freeboard,

                side_slope,

                1.5

            )


            clearance = max(

                10,

                0.5

                *
                math.sqrt(

                    design["top_length"] ** 2

                    +

                    design["top_breadth"] ** 2

                )

                +
                2

            )


            site_result = find_pond_site(

                field,

                clearance

            )


            if site_result is None:

                st.error(
                    "No feasible GIS candidate site was found."
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


            soil_rank = sample_point(

                SOIL_RASTER,

                site["x"],

                site["y"]

            )


            geology_rank = sample_point(

                GEOLOGY_RASTER,

                site["x"],

                site["y"]

            )


            lineament = sample_point(

                LINEAMENT_DENSITY_RASTER,

                site["x"],

                site["y"]

            )


            risk, reasons, lining = (
                seepage_risk(

                    soil_rank,

                    geology_rank,

                    lineament

                )
            )


            st.session_state.assessment = {

                "farmer":
                    farmer_name,

                "village":
                    village,

                "area":
                    farm_area,

                "lulc":
                    current_lulc,

                "mapped_lulc":
                    mapped_lulc,

                "rainfall":
                    rainfall,

                "runoff":
                    runoff,

                "dependable":
                    dependable,

                "dependable_volume":
                    dependable_volume,

                "cn":
                    cn,

                "runoff_coeff":
                    runoff_coeff,

                "site":
                    site,

                "lowest":
                    lowest,

                "design":
                    design,

                "soil_rank":
                    soil_rank,

                "geology_rank":
                    geology_rank,

                "lineament":
                    lineament,

                "risk":
                    risk,

                "risk_reasons":
                    reasons,

                "lining":
                    lining

            }


        st.rerun()


# ============================================================
# RESULTS
# ============================================================

result = (
    st.session_state.assessment
)


if result:

    site = result[
        "site"
    ]


    design = result[
        "design"
    ]


    lowest = result[
        "lowest"
    ]


    st.header(
        "✅ DSS Result"
    )


    st.subheader(
        "🌧️ Water Availability"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Rainfall",
        f"{result['rainfall']:.0f} mm/year"
    )


    c2.metric(
        "Dependable Runoff",
        f"{result['dependable']:.0f} mm"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Field Area",
        f"{result['area']:.3f} ha"
    )


    c2.metric(
        "Dependable Water",
        f"{result['dependable_volume']:,.0f} m³/year"
    )


    st.subheader(
        "📍 Proposed Farm Pond Site"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Latitude",
        f"{site['latitude']:.6f}"
    )


    c2.metric(
        "Longitude",
        f"{site['longitude']:.6f}"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Elevation",
        f"{site['elevation']:.1f} m"
    )


    c2.metric(
        "Slope",
        f"{site['slope']:.2f}°"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "TWI",
        f"{site['twi']:.2f}"
    )


    c2.metric(
        "Boundary distance",
        f"{site['boundary_distance']:.1f} m"
    )


    # --------------------------------------------------------
    # RESULT MAP
    # --------------------------------------------------------

    map_result = folium.Map(

        location=[
            site["latitude"],
            site["longitude"]
        ],

        zoom_start=17,

        tiles=None

    )


    folium.TileLayer(

        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/"
            "tile/{z}/{y}/{x}"
        ),

        attr="Esri",

        name="Satellite"

    ).add_to(
        map_result
    )


    folium.GeoJson(

        field,

        style_function=lambda x: {

            "color":
                "lime",

            "weight":
                5,

            "fillOpacity":
                0.15

        }

    ).add_to(
        map_result
    )


    pond_rectangle = box(

        site["x"]
        -
        design["top_length"]
        /
        2,

        site["y"]
        -
        design["top_breadth"]
        /
        2,

        site["x"]
        +
        design["top_length"]
        /
        2,

        site["y"]
        +
        design["top_breadth"]
        /
        2

    )


    pond_wgs = (

        gpd.GeoSeries(

            [pond_rectangle],

            crs=WORKING_CRS

        )

        .to_crs(
            "EPSG:4326"
        )

        .iloc[0]

    )


    folium.GeoJson(

        mapping(
            pond_wgs
        ),

        style_function=lambda x: {

            "color":
                "red",

            "weight":
                3,

            "fillColor":
                "red",

            "fillOpacity":
                0.25

        }

    ).add_to(
        map_result
    )


    folium.Marker(

        [
            site["latitude"],
            site["longitude"]
        ],

        tooltip=
            "Recommended Pond Site",

        icon=folium.Icon(
            color="red",
            icon="tint"
        )

    ).add_to(
        map_result
    )


    folium.Marker(

        [
            lowest["latitude"],
            lowest["longitude"]
        ],

        tooltip=
            "Absolute Lowest DEM Point",

        icon=folium.Icon(
            color="orange"
        )

    ).add_to(
        map_result
    )


    Fullscreen().add_to(
        map_result
    )


    st_folium(

        map_result,

        height=520,

        use_container_width=True,

        key="result_map"

    )


    # --------------------------------------------------------
    # DESIGN
    # --------------------------------------------------------

    st.subheader(
        "📐 Preliminary Pond Design"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Storage Capacity",
        f"{design['target_storage']:,.0f} m³"
    )


    c2.metric(
        "Water Depth",
        f"{design['water_depth']:.2f} m"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Bottom Length",
        f"{design['bottom_length']:.1f} m"
    )


    c2.metric(
        "Bottom Breadth",
        f"{design['bottom_breadth']:.1f} m"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Top Length",
        f"{design['top_length']:.1f} m"
    )


    c2.metric(
        "Top Breadth",
        f"{design['top_breadth']:.1f} m"
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Footprint",
        f"{design['footprint']:,.0f} m²"
    )


    c2.metric(
        "Field Occupied",
        f"{design['occupied_percent']:.1f}%"
    )


    st.metric(
        "Remaining Land",
        f"{design['remaining_area_ha']:.3f} ha"
    )


    # --------------------------------------------------------
    # SEEPAGE
    # --------------------------------------------------------

    st.subheader(
        "🪨 Seepage and Lining"
    )


    if result["risk"] == "HIGH":

        st.error(
            "Seepage Risk: HIGH"
        )


    elif result["risk"] == "MODERATE":

        st.warning(
            "Seepage Risk: MODERATE"
        )


    else:

        st.success(
            "Seepage Risk: LOW"
        )


    st.write(
        "**Soil:**",
        soil_description(
            result["soil_rank"]
        )
    )


    st.write(
        "**Geology:**",
        geology_description(
            result["geology_rank"]
        )
    )


    for reason in (
        result["risk_reasons"]
    ):

        st.write(
            "•",
            reason
        )


    st.info(
        result["lining"]
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Wetted Lining Area",
        f"{design['lining_area']:,.0f} m²"
    )


    c2.metric(
        "Procurement Area (+10%)",
        f"{design['liner_purchase']:,.0f} m²"
    )


    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    output = {

        "Farmer":
            result["farmer"],

        "Village":
            result["village"],

        "Field_Area_ha":
            result["area"],

        "Current_Land_Use":
            result["lulc"],

        "Rainfall_mm":
            result["rainfall"],

        "Dependable_Runoff_mm":
            result["dependable"],

        "Dependable_Water_m3":
            result["dependable_volume"],

        "Pond_Latitude":
            site["latitude"],

        "Pond_Longitude":
            site["longitude"],

        "Elevation_m":
            site["elevation"],

        "Slope_degree":
            site["slope"],

        "Storage_m3":
            design["target_storage"],

        "Bottom_Length_m":
            design["bottom_length"],

        "Bottom_Breadth_m":
            design["bottom_breadth"],

        "Top_Length_m":
            design["top_length"],

        "Top_Breadth_m":
            design["top_breadth"],

        "Water_Depth_m":
            design["water_depth"],

        "Freeboard_m":
            design["freeboard"],

        "Footprint_m2":
            design["footprint"],

        "Seepage_Risk":
            result["risk"],

        "Lining_Area_m2":
            design["lining_area"]

    }


    csv = (

        pd.DataFrame(
            [output]
        )

        .to_csv(
            index=False
        )

        .encode(
            "utf-8"
        )

    )


    st.download_button(

        "⬇️ Download Farmer Assessment",

        data=csv,

        file_name=
            "RiBhoi_RWH_Assessment.csv",

        mime="text/csv",

        use_container_width=True

    )


    st.warning(
        """
This DSS provides preliminary GIS-based planning only.
The proposed location represents a 30-m GIS candidate zone.
Final location, pond capacity, embankment, spillway, inlet,
outlet, lining and structural design require field survey
and engineering verification.
"""
    )


# ============================================================
# RESET
# ============================================================

st.markdown("---")


if st.button(
    "🔄 Start New Assessment",
    use_container_width=True
):

    st.session_state.field_geometry = None

    st.session_state.assessment = None

    st.session_state.current_lulc = None

    st.rerun()
