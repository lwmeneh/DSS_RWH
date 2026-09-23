# ============================================================
# RI BHOI RAINWATER HARVESTING GIS DSS
# MOBILE-FIRST FARMER FIELD + WATERSHED ASSESSMENT
# ============================================================

import numpy as np
import geopandas as gpd
import streamlit as st
import folium

from shapely.geometry import shape
from folium.plugins import Draw, Fullscreen, MeasureControl
from streamlit_folium import st_folium

from config.settings import *
from database.db import initialize, save_assessment
from data.loader import load_vectors
from gis.spatial import (
    nearest_feature,
    zonal_stats,
    band_index_by_name,
)
from reports.pdf_report import make_pdf
from utils.ui import load_css, render_html


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_SHORT_NAME,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    min-height: 55px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 650;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce7eb;
    border-radius: 12px;
    padding: 10px;
}

.card {
    background: white;
    border: 1px solid #dce7eb;
    border-radius: 12px;
    padding: 13px;
    margin: 8px 0;
}

.card-green {
    border-left: 5px solid #159957;
}

.card-blue {
    border-left: 5px solid #138aa5;
}

.card-orange {
    border-left: 5px solid #ee9800;
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
        font-size: 1.22rem !important;
    }

    h3 {
        font-size: 1.08rem !important;
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

load_css()
initialize()


# ============================================================
# ADDITIONAL RASTER PATHS
# ============================================================

RAINFALL_RASTER = (
    CACHE_DIR
    / "mean_rainfall_mm.tif"
)

STREAM_ORDER_RASTER = (
    CACHE_DIR
    / "stream_order_30m.tif"
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required = [

    CORE_STACK,

    CN_RASTER,

    RAINFALL_RASTER,

    MEAN_RUNOFF_RASTER,

    DEPENDABLE_RUNOFF_RASTER,

    RUNOFF_COEFF_RASTER,

    STREAM_ORDER_RASTER,

    STREAMS_GPKG,

    BOUNDARY_GPKG,
]

missing = [
    str(path)
    for path in required
    if not path.exists()
]

if missing:

    st.error(
        "Required GIS cache is incomplete."
    )

    st.code(
        "\n".join(missing)
    )

    st.stop()


# ============================================================
# LOAD VECTOR DATA
# ============================================================

DATA = load_vectors()

streams_utm = DATA.get(
    "streams"
)

boundary_utm = DATA.get(
    "boundary"
)

mgnrega_utm = DATA.get(
    "mgnrega"
)

final94_utm = DATA.get(
    "final94"
)

published_utm = DATA.get(
    "published"
)

lulc_utm = DATA.get(
    "lulc"
)

boundary_wgs = (
    boundary_utm
    .to_crs(
        "EPSG:4326"
    )
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "field_geometry":
        None,

    "watershed_geometry":
        None,

    "result":
        None,
}

for key, value in defaults.items():

    st.session_state.setdefault(
        key,
        value,
    )


# ============================================================
# DRAWN POLYGON
# ============================================================

def get_last_polygon(
    map_data
):

    """
    Only polygons created with Leaflet Draw
    are accepted.

    Ordinary mobile taps/clicks are ignored.
    """

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

        geometry = drawing.get(
            "geometry",
            {}
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
# GEOMETRY
# ============================================================

def to_utm_geometry(
    geometry_geojson
):

    return (

        gpd.GeoDataFrame(

            geometry=[
                shape(
                    geometry_geojson
                )
            ],

            crs="EPSG:4326",

        )

        .to_crs(
            WORKING_CRS
        )

        .geometry
        .iloc[0]

    )


def calculate_area_ha(
    geometry_geojson
):

    geom = to_utm_geometry(
        geometry_geojson
    )

    return float(
        geom.area
        / 10000.0
    )


def geometry_center(
    geometry_geojson
):

    geom = to_utm_geometry(
        geometry_geojson
    )

    centroid = geom.centroid

    point = (

        gpd.GeoSeries(

            [centroid],

            crs=WORKING_CRS,

        )

        .to_crs(
            "EPSG:4326"
        )

        .iloc[0]

    )

    return [

        float(
            point.y
        ),

        float(
            point.x
        ),
    ]


# ============================================================
# LULC CLIPPING
# ============================================================

def clipped_lulc(
    polygon_geojson
):

    """
    Clip LULC strictly to farmer/watershed polygon.
    """

    if (

        lulc_utm is None

        or lulc_utm.empty

    ):

        return None


    selected = gpd.GeoDataFrame(

        {
            "selection_id":
                [1]
        },

        geometry=[

            shape(
                polygon_geojson
            )

        ],

        crs="EPSG:4326",

    )


    selected = selected.to_crs(
        WORKING_CRS
    )


    lulc = lulc_utm.copy()


    if lulc.crs != selected.crs:

        lulc = lulc.to_crs(
            selected.crs
        )


    selected_geom = (
        selected.geometry.iloc[0]
    )


    ids = list(

        lulc.sindex.intersection(

            selected_geom.bounds

        )

    )


    if not ids:

        return None


    candidate = (
        lulc
        .iloc[ids]
        .copy()
    )


    candidate = candidate[

        candidate.intersects(
            selected_geom
        )

    ].copy()


    if candidate.empty:

        return None


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


    clipped = gpd.overlay(

        candidate,

        selected,

        how="intersection",

        keep_geom_type=True,

        make_valid=True,

    )


    if clipped.empty:

        return None


    clipped = clipped[

        clipped.geometry.notna()

        &

        (~clipped.geometry.is_empty)

    ].copy()


    clipped[
        "area_m2"
    ] = (
        clipped.geometry.area
    )


    return clipped


# ============================================================
# LULC SUMMARY
# ============================================================

def lulc_summary(
    polygon_geojson
):

    clipped = clipped_lulc(
        polygon_geojson
    )


    if (

        clipped is None

        or clipped.empty

    ):

        return []


    if "LULC_2022" not in clipped.columns:

        return []


    summary = (

        clipped

        .groupby(
            "LULC_2022",
            dropna=False,
        )["area_m2"]

        .sum()

        .reset_index()

    )


    total = float(
        summary[
            "area_m2"
        ].sum()
    )


    if total <= 0:

        return []


    summary[
        "area_ha"
    ] = (

        summary[
            "area_m2"
        ]

        / 10000.0

    )


    summary[
        "percent"
    ] = (

        summary[
            "area_m2"
        ]

        / total

        * 100.0

    )


    summary = summary.sort_values(

        "percent",

        ascending=False,

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
        in summary.iterrows()

    ]


# ============================================================
# LULC DISPLAY
# ============================================================

def lulc_display_geojson(
    polygon_geojson
):

    clipped = clipped_lulc(
        polygon_geojson
    )

    if (

        clipped is None

        or clipped.empty

    ):

        return None


    clipped = clipped.to_crs(
        "EPSG:4326"
    )

    return (
        clipped
        .__geo_interface__
    )


# ============================================================
# NEAREST STREAM
# ============================================================

def nearest_stream_distance(
    polygon_geojson
):

    if (

        streams_utm is None

        or streams_utm.empty

    ):

        return None


    geom = to_utm_geometry(
        polygon_geojson
    )


    distance = (

        streams_utm
        .geometry
        .distance(
            geom
        )

    )


    if distance.empty:

        return None


    return float(
        distance.min()
    )


# ============================================================
# LOCAL STREAMS FOR MAP
# ============================================================

def local_streams(
    polygon_geojson,
    buffer_m=2500,
):

    if (

        streams_utm is None

        or streams_utm.empty

    ):

        return None


    geom = to_utm_geometry(
        polygon_geojson
    )


    search_area = (
        geom.buffer(
            buffer_m
        )
    )


    ids = list(

        streams_utm
        .sindex
        .intersection(
            search_area.bounds
        )

    )


    if not ids:

        return None


    subset = (
        streams_utm
        .iloc[ids]
        .copy()
    )


    subset = subset[

        subset.intersects(
            search_area
        )

    ].copy()


    if subset.empty:

        return None


    return (

        subset
        .to_crs(
            "EPSG:4326"
        )
        .__geo_interface__

    )


# ============================================================
# RWH RECOMMENDATION ENGINE
# ============================================================

def recommend_structures(

    lulc_name,

    area,

    slope,

    cn,

    runoff,

    dependable_runoff,

    distance_stream,

    stream_order,

):

    recommendations = []


    lname = (
        lulc_name
        or ""
    ).lower()


    agriculture = any(

        x in lname

        for x in [

            "crop",

            "plantation",

            "shifting cultivation",

        ]

    )


    # ========================================================
    # FARM POND
    # ========================================================

    if (

        agriculture

        and np.isfinite(
            slope
        )

        and slope <= 8

        and np.isfinite(
            runoff
        )

        and runoff >= 100

    ):

        recommendations.append({

            "structure":
                "Farm Pond",

            "priority":
                "High",

            "reason":
                "Agricultural land, suitable slope and available "
                "runoff support on-farm water storage.",

            "management":
                "Provide inlet silt trap, stabilize bunds with "
                "vegetation, maintain safe overflow and desilt "
                "before the monsoon. Use stored water primarily "
                "for protective irrigation.",

        })


    # ========================================================
    # OFF-STREAM FARM POND
    # ========================================================

    if (

        distance_stream is not None

        and distance_stream <= 200

        and np.isfinite(
            slope
        )

        and slope <= 10

    ):

        recommendations.append({

            "structure":
                "Off-stream Farm Pond with Controlled Diversion",

            "priority":
                "High",

            "reason":
                "The farmer field lies close to a mapped stream "
                "or drainage line.",

            "management":
                "Use a controlled diversion only after hydraulic "
                "verification. Provide sediment trapping and a safe "
                "overflow without obstructing the natural stream.",

        })


    # ========================================================
    # RECHARGE POND
    # ========================================================

    if (

        np.isfinite(
            slope
        )

        and slope <= 10

        and np.isfinite(
            cn
        )

        and cn <= 80

    ):

        recommendations.append({

            "structure":
                "Recharge / Percolation Pond",

            "priority":
                "Moderate",

            "reason":
                "Gentle terrain and relatively lower Curve Number "
                "support recharge-oriented treatment.",

            "management":
                "Maintain the infiltration bed, remove sediment "
                "periodically and protect recharge water from contamination.",

        })


    # ========================================================
    # CONTOUR TREATMENT
    # ========================================================

    if (

        np.isfinite(
            slope
        )

        and 8 < slope <= 20

    ):

        recommendations.append({

            "structure":
                "Contour Bund / Contour Trench",

            "priority":
                "Moderate",

            "reason":
                "Moderately sloping terrain requires distributed "
                "runoff interception.",

            "management":
                "Construct strictly along contour and stabilize "
                "with grass or suitable vegetative barriers.",

        })


    # ========================================================
    # STEEP TERRAIN
    # ========================================================

    if (

        np.isfinite(
            slope
        )

        and slope > 20

    ):

        recommendations.append({

            "structure":
                "Staggered Trench + Vegetative Barrier",

            "priority":
                "High",

            "reason":
                "Steep terrain needs distributed runoff and "
                "erosion management.",

            "management":
                "Avoid large excavations. Use staggered trenches, "
                "vegetation and erosion-control treatment.",

        })


    # ========================================================
    # CHECK DAM
    # ========================================================

    if (

        stream_order is not None

        and np.isfinite(
            stream_order
        )

        and stream_order >= 1

    ):

        recommendations.append({

            "structure":
                "Check Dam / Gully Control Structure",

            "priority":
                "Watershed-level screening",

            "reason":
                (
                    "A stream of order "
                    f"{int(round(stream_order))} "
                    "occurs within the selected watershed."
                ),

            "management":
                "Verify channel cross-section, streambed material, "
                "foundation stability, design discharge and spillway "
                "before construction. Inspect after major monsoon events.",

        })


    # ========================================================
    # BUILT-UP
    # ========================================================

    if "built" in lname:

        recommendations.append({

            "structure":
                "Rooftop Rainwater Harvesting + Recharge Pit",

            "priority":
                "High",

            "reason":
                "Built-up areas are more suitable for rooftop "
                "collection than agricultural pond excavation.",

            "management":
                "Provide first-flush arrangement, filtration and "
                "regular cleaning of roof, tank and recharge media.",

        })


    # ========================================================
    # FOREST
    # ========================================================

    if "forest" in lname:

        recommendations.append({

            "structure":
                "Vegetative / Low-disturbance Recharge Treatment",

            "priority":
                "Preferred",

            "reason":
                "Forest land should prioritize low-disturbance "
                "water conservation.",

            "management":
                "Protect vegetation, minimize excavation and use "
                "small distributed infiltration treatments.",

        })


    # ========================================================
    # SCRUB
    # ========================================================

    if "scrub" in lname:

        recommendations.append({

            "structure":
                "Staggered Trench / Vegetative Rehabilitation",

            "priority":
                "Moderate",

            "reason":
                "Scrub land can benefit from distributed runoff "
                "detention and vegetation restoration.",

            "management":
                "Combine trenches with revegetation and protect "
                "treated areas during establishment.",

        })


    # ========================================================
    # FALLBACK
    # ========================================================

    if not recommendations:

        recommendations.append({

            "structure":
                "In-situ Rainwater Conservation",

            "priority":
                "General",

            "reason":
                "Available GIS indicators favour distributed "
                "field-scale water conservation.",

            "management":
                "Use field bunding, vegetative barriers and "
                "local infiltration measures.",

        })


    return recommendations


# ============================================================
# RESET
# ============================================================

def clear_field():

    st.session_state[
        "field_geometry"
    ] = None

    st.session_state[
        "watershed_geometry"
    ] = None

    st.session_state[
        "result"
    ] = None


def clear_watershed():

    st.session_state[
        "watershed_geometry"
    ] = None

    st.session_state[
        "result"
    ] = None


# ============================================================
# HEADER
# ============================================================

render_html(
    "hero.html"
)


st.markdown(
    """
<div class="card card-blue">

<b>📱 Farmer Mobile GIS Assessment</b><br>

Draw the exact farmer field and watershed using the polygon tool.
Only the area inside the farmer-drawn polygons is analysed.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FARMER DETAILS
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
# STEP 1
# ============================================================

st.header(
    "1️⃣ Draw Farmer Field"
)


st.info(
    "Tap polygon tool ⬠ → tap each field corner → "
    "tap the first point again to close the polygon."
)


# ============================================================
# FIELD MAP
# ============================================================

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


# District outline only

folium.GeoJson(

    boundary_wgs.__geo_interface__,

    name="Ri Bhoi Boundary",

    style_function=lambda x: {

        "color": "#FFD600",

        "weight": 2,

        "fillOpacity": 0,

    },

).add_to(
    field_map
)


# Existing field

if st.session_state[
    "field_geometry"
]:

    folium.GeoJson(

        st.session_state[
            "field_geometry"
        ],

        name="Farmer Field",

        style_function=lambda x: {

            "color": "#00C853",

            "weight": 5,

            "fillColor": "#69F0AE",

            "fillOpacity": 0.25,

        },

    ).add_to(
        field_map
    )


    # LULC clipped inside farm

    field_lulc_layer = (
        lulc_display_geojson(

            st.session_state[
                "field_geometry"
            ]

        )
    )


    if field_lulc_layer:

        folium.GeoJson(

            field_lulc_layer,

            name="LULC inside Farm",

            style_function=lambda x: {

                "color": "#8A6D1D",

                "weight": 1,

                "fillOpacity": 0.20,

            },

            tooltip=folium.GeoJsonTooltip(

                fields=[
                    "LULC_2022"
                ],

                aliases=[
                    "LULC:"
                ],

            ),

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

            "repeatMode":
                False,

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

    key="field_map",

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
            "watershed_geometry"
        ] = None

        st.session_state[
            "result"
        ] = None

        st.rerun()


field = st.session_state[
    "field_geometry"
]


# ============================================================
# IMMEDIATE FIELD OUTPUT
# ============================================================

if field:

    field_area = (
        calculate_area_ha(
            field
        )
    )


    field_lulc = (
        lulc_summary(
            field
        )
    )


    dominant_lulc = (

        field_lulc[0][
            "name"
        ]

        if field_lulc

        else "No LULC data"

    )


    rainfall = zonal_stats(

        RAINFALL_RASTER,

        field,

    )["mean"]


    runoff = zonal_stats(

        MEAN_RUNOFF_RASTER,

        field,

    )["mean"]


    dependable = zonal_stats(

        DEPENDABLE_RUNOFF_RASTER,

        field,

    )["mean"]


    potential_water = (

        runoff

        * field_area

        * 10

        if np.isfinite(
            runoff
        )

        else np.nan

    )


    st.success(

        f"✅ Farmer field captured: "
        f"{field_area:.3f} ha"

    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Dominant LULC",

        dominant_lulc,

    )


    c2.metric(

        "Field Area",

        f"{field_area:.3f} ha",

    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Annual Rainfall",

        (
            f"{rainfall:.0f} mm"

            if np.isfinite(
                rainfall
            )

            else "No data"
        ),

    )


    c2.metric(

        "Annual Runoff",

        (
            f"{runoff:.0f} mm"

            if np.isfinite(
                runoff
            )

            else "No data"
        ),

    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "75% Dependable Runoff",

        (
            f"{dependable:.0f} mm"

            if np.isfinite(
                dependable
            )

            else "No data"
        ),

    )


    c2.metric(

        "Potential Water",

        (
            f"{potential_water:,.0f} m³/year"

            if np.isfinite(
                potential_water
            )

            else "No data"
        ),

    )


    with st.expander(
        "🌾 Field LULC Composition"
    ):

        st.dataframe(

            field_lulc,

            use_container_width=True,

            hide_index=True,

        )


    if st.button(

        "🗑 Redraw Farmer Field",

        use_container_width=True,

    ):

        clear_field()

        st.rerun()


else:

    st.info(
        "Draw and close the farmer field polygon."
    )


# ============================================================
# STEP 2 — WATERSHED
# ============================================================

if field:

    st.markdown("---")


    st.header(
        "2️⃣ Draw Watershed / Catchment"
    )


    st.info(
        "Draw the contributing watershed as an orange polygon."
    )


    watershed_map = folium.Map(

        location=geometry_center(
            field
        ),

        zoom_start=14,

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
        watershed_map
    )


    # Farmer field

    folium.GeoJson(

        field,

        name="Farmer Field",

        style_function=lambda x: {

            "color":
                "#00C853",

            "weight":
                4,

            "fillColor":
                "#69F0AE",

            "fillOpacity":
                0.20,

        },

    ).add_to(
        watershed_map
    )


    # Nearby streams

    reference_geometry = (

        st.session_state[
            "watershed_geometry"
        ]

        if st.session_state[
            "watershed_geometry"
        ]

        else field

    )


    stream_layer = local_streams(
        reference_geometry
    )


    if stream_layer:

        folium.GeoJson(

            stream_layer,

            name="Nearby Streams",

            style_function=lambda x: {

                "color":
                    "#1565C0",

                "weight":
                    2.5,

            },

        ).add_to(
            watershed_map
        )


    # Existing watershed

    if st.session_state[
        "watershed_geometry"
    ]:

        folium.GeoJson(

            st.session_state[
                "watershed_geometry"
            ],

            name="Watershed",

            style_function=lambda x: {

                "color":
                    "#FF6D00",

                "weight":
                    5,

                "fillColor":
                    "#FFB74D",

                "fillOpacity":
                    0.18,

            },

        ).add_to(
            watershed_map
        )


        watershed_lulc_layer = (
            lulc_display_geojson(

                st.session_state[
                    "watershed_geometry"
                ]

            )
        )


        if watershed_lulc_layer:

            folium.GeoJson(

                watershed_lulc_layer,

                name="Watershed LULC",

                style_function=lambda x: {

                    "color":
                        "#795548",

                    "weight":
                        1,

                    "fillOpacity":
                        0.10,

                },

                tooltip=folium.GeoJsonTooltip(

                    fields=[
                        "LULC_2022"
                    ],

                    aliases=[
                        "LULC:"
                    ],

                ),

            ).add_to(
                watershed_map
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
                        "#FF6D00",

                    "weight":
                        5,

                    "fillColor":
                        "#FFB74D",

                    "fillOpacity":
                        0.20,

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
        watershed_map
    )


    Fullscreen().add_to(
        watershed_map
    )


    MeasureControl(

        primary_length_unit="meters",

        primary_area_unit="hectares",

    ).add_to(
        watershed_map
    )


    folium.LayerControl(
        collapsed=True
    ).add_to(
        watershed_map
    )


    watershed_output = st_folium(

        watershed_map,

        height=520,

        use_container_width=True,

        key="watershed_map",

        returned_objects=[

            "all_drawings",

            "last_active_drawing",

        ],

    )


    new_watershed = (
        get_last_polygon(
            watershed_output
        )
    )


    if new_watershed is not None:

        if (

            new_watershed

            !=

            st.session_state[
                "watershed_geometry"
            ]

        ):

            st.session_state[
                "watershed_geometry"
            ] = new_watershed

            st.session_state[
                "result"
            ] = None

            st.rerun()


watershed = st.session_state[
    "watershed_geometry"
]


if field and watershed:

    st.success(

        "✅ Watershed captured: "
        f"{calculate_area_ha(watershed):.2f} ha"

    )


    with st.expander(
        "🌿 Watershed LULC"
    ):

        st.dataframe(

            lulc_summary(
                watershed
            ),

            use_container_width=True,

            hide_index=True,

        )


    if st.button(

        "🗑 Redraw Watershed",

        use_container_width=True,

    ):

        clear_watershed()

        st.rerun()


# ============================================================
# STEP 3 — ANALYSIS
# ============================================================

if field and watershed:

    st.markdown("---")


    st.header(
        "3️⃣ Complete RWH Assessment"
    )


    if st.button(

        "💧 RUN COMPLETE RWH ASSESSMENT",

        type="primary",

        use_container_width=True,

    ):

        field_area = (
            calculate_area_ha(
                field
            )
        )


        watershed_area = (
            calculate_area_ha(
                watershed
            )
        )


        # ====================================================
        # LULC
        # ====================================================

        field_lulc = (
            lulc_summary(
                field
            )
        )


        watershed_lulc = (
            lulc_summary(
                watershed
            )
        )


        dominant_lulc = (

            field_lulc[0][
                "name"
            ]

            if field_lulc

            else None

        )


        # ====================================================
        # FIELD RAINFALL
        # ====================================================

        field_rainfall = zonal_stats(

            RAINFALL_RASTER,

            field,

        )["mean"]


        # ====================================================
        # FIELD RUNOFF
        # ====================================================

        field_runoff = zonal_stats(

            MEAN_RUNOFF_RASTER,

            field,

        )["mean"]


        field_dependable = zonal_stats(

            DEPENDABLE_RUNOFF_RASTER,

            field,

        )["mean"]


        field_coefficient = zonal_stats(

            RUNOFF_COEFF_RASTER,

            field,

        )["mean"]


        field_cn = zonal_stats(

            CN_RASTER,

            field,

        )["mean"]


        # ====================================================
        # WATERSHED RAINFALL + RUNOFF
        # ====================================================

        watershed_rainfall = zonal_stats(

            RAINFALL_RASTER,

            watershed,

        )["mean"]


        watershed_runoff = zonal_stats(

            MEAN_RUNOFF_RASTER,

            watershed,

        )["mean"]


        watershed_dependable = zonal_stats(

            DEPENDABLE_RUNOFF_RASTER,

            watershed,

        )["mean"]


        # ====================================================
        # STREAM ORDER
        # ====================================================

        stream_order_data = zonal_stats(

            STREAM_ORDER_RASTER,

            watershed,

        )


        stream_order = (
            stream_order_data[
                "max"
            ]
        )


        # ====================================================
        # SLOPE
        # ====================================================

        field_slope = np.nan


        slope_band = (
            band_index_by_name(

                CORE_STACK,

                "slope",

            )
        )


        if slope_band:

            field_slope = zonal_stats(

                CORE_STACK,

                field,

                slope_band,

            )["mean"]


        # ====================================================
        # WATER VOLUMES
        # ====================================================

        field_water = (

            field_runoff

            * field_area

            * 10

            if np.isfinite(
                field_runoff
            )

            else np.nan

        )


        field_dependable_water = (

            field_dependable

            * field_area

            * 10

            if np.isfinite(
                field_dependable
            )

            else np.nan

        )


        watershed_water = (

            watershed_runoff

            * watershed_area

            * 10

            if np.isfinite(
                watershed_runoff
            )

            else np.nan

        )


        watershed_dependable_water = (

            watershed_dependable

            * watershed_area

            * 10

            if np.isfinite(
                watershed_dependable
            )

            else np.nan

        )


        # ====================================================
        # STREAM
        # ====================================================

        stream_distance = (
            nearest_stream_distance(
                field
            )
        )


        # ====================================================
        # STRUCTURES
        # ====================================================

        nearest_mgnrega = (

            nearest_feature(

                field,

                mgnrega_utm,

            )

            if mgnrega_utm is not None

            else None

        )


        nearest_final = (

            nearest_feature(

                field,

                final94_utm,

            )

            if final94_utm is not None

            else None

        )


        nearest_published = (

            nearest_feature(

                field,

                published_utm,

            )

            if published_utm is not None

            else None

        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        recommendations = (
            recommend_structures(

                dominant_lulc,

                field_area,

                field_slope,

                field_cn,

                field_runoff,

                field_dependable,

                stream_distance,

                stream_order,

            )
        )


        result = {

            "field_boundary_source":
                "Farmer drawn polygon",

            "field_area_ha":
                field_area,

            "watershed_area_ha":
                watershed_area,

            "dominant_lulc_name":
                dominant_lulc,

            "field_lulc_composition":
                field_lulc,

            "watershed_lulc_composition":
                watershed_lulc,

            "field_rainfall_mm":
                field_rainfall,

            "watershed_rainfall_mm":
                watershed_rainfall,

            "field_cn":
                field_cn,

            "field_slope":
                field_slope,

            "field_runoff_mm":
                field_runoff,

            "field_dependable_mm":
                field_dependable,

            "field_runoff_coeff":
                field_coefficient,

            "field_runoff_m3":
                field_water,

            "field_dependable_m3":
                field_dependable_water,

            "watershed_runoff_mm":
                watershed_runoff,

            "watershed_dependable_mm":
                watershed_dependable,

            "watershed_runoff_m3":
                watershed_water,

            "watershed_dependable_m3":
                watershed_dependable_water,

            "stream_order":
                stream_order,

            "field_stream_distance_m":
                stream_distance,

            "nearest_mgnrega":
                nearest_mgnrega,

            "nearest_final_site":
                nearest_final,

            "nearest_published":
                nearest_published,

            "farmer_recommendations":
                recommendations,

            "watershed_recommendations":
                [],
        }


        st.session_state[
            "result"
        ] = result


        # Database compatibility

        if farmer_name or village:

            try:

                db_result = (
                    result.copy()
                )


                db_result[
                    "farmer_recommendations"
                ] = [

                    (
                        item[
                            "structure"
                        ],

                        item[
                            "reason"
                        ]
                    )

                    for item
                    in recommendations

                ]


                save_assessment(

                    farmer_name,

                    village,

                    db_result,

                )


            except Exception:

                pass


        st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state[
    "result"
]:

    r = st.session_state[
        "result"
    ]


    recommendations = (
        r[
            "farmer_recommendations"
        ]
    )


    st.markdown("---")


    st.header(
        "✅ RWH Assessment Result"
    )


    # ========================================================
    # MAIN RECOMMENDATION
    # ========================================================

    if recommendations:

        primary = (
            recommendations[0]
        )


        st.success(
            f"""
### 🏗️ Recommended Structure

**{primary['structure']}**

Priority: **{primary['priority']}**

{primary['reason']}
"""
        )


        st.info(
            f"""
### 🌱 Management Recommendation

{primary['management']}
"""
        )


    # ========================================================
    # FIELD
    # ========================================================

    st.subheader(
        "🌾 Farmer Field"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Field Area",

        f"{r['field_area_ha']:.3f} ha",

    )


    c2.metric(

        "Dominant LULC",

        r[
            "dominant_lulc_name"
        ]

        or "No data",

    )


    # ========================================================
    # FIELD WATER
    # ========================================================

    st.subheader(
        "🌧️ Field Rainfall & Runoff"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Annual Rainfall",

        (
            f"{r['field_rainfall_mm']:.0f} mm"

            if np.isfinite(
                r[
                    "field_rainfall_mm"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "Annual Runoff",

        (
            f"{r['field_runoff_mm']:.0f} mm"

            if np.isfinite(
                r[
                    "field_runoff_mm"
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
            f"{r['field_runoff_m3']:,.0f} m³"

            if np.isfinite(
                r[
                    "field_runoff_m3"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "75% Dependable Water",

        (
            f"{r['field_dependable_m3']:,.0f} m³"

            if np.isfinite(
                r[
                    "field_dependable_m3"
                ]
            )

            else "No data"
        ),

    )


    # ========================================================
    # WATERSHED
    # ========================================================

    st.subheader(
        "🌊 Watershed Hydrology"
    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Watershed Area",

        f"{r['watershed_area_ha']:.2f} ha",

    )


    c2.metric(

        "Highest Stream Order",

        (
            str(
                int(
                    round(
                        r[
                            "stream_order"
                        ]
                    )
                )
            )

            if np.isfinite(
                r[
                    "stream_order"
                ]
            )

            else "No stream"
        ),

    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Watershed Rainfall",

        (
            f"{r['watershed_rainfall_mm']:.0f} mm"

            if np.isfinite(
                r[
                    "watershed_rainfall_mm"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "Watershed Runoff",

        (
            f"{r['watershed_runoff_mm']:.0f} mm"

            if np.isfinite(
                r[
                    "watershed_runoff_mm"
                ]
            )

            else "No data"
        ),

    )


    c1, c2 = st.columns(
        2
    )


    c1.metric(

        "Watershed Runoff Volume",

        (
            f"{r['watershed_runoff_m3']:,.0f} m³/year"

            if np.isfinite(
                r[
                    "watershed_runoff_m3"
                ]
            )

            else "No data"
        ),

    )


    c2.metric(

        "Dependable Watershed Water",

        (
            f"{r['watershed_dependable_m3']:,.0f} m³"

            if np.isfinite(
                r[
                    "watershed_dependable_m3"
                ]
            )

            else "No data"
        ),

    )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    with st.expander(
        "📊 Technical Site Details"
    ):

        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "CN-II",

            (
                f"{r['field_cn']:.1f}"

                if np.isfinite(
                    r[
                        "field_cn"
                    ]
                )

                else "No data"
            ),

        )


        c2.metric(

            "Mean Slope",

            (
                f"{r['field_slope']:.2f}%"

                if np.isfinite(
                    r[
                        "field_slope"
                    ]
                )

                else "No data"
            ),

        )


        c1, c2 = st.columns(
            2
        )


        c1.metric(

            "Runoff Coefficient",

            (
                f"{r['field_runoff_coeff']:.3f}"

                if np.isfinite(
                    r[
                        "field_runoff_coeff"
                    ]
                )

                else "No data"
            ),

        )


        c2.metric(

            "Distance to Stream",

            (
                f"{r['field_stream_distance_m']:.0f} m"

                if r[
                    "field_stream_distance_m"
                ]
                is not None

                else "No data"
            ),

        )


    # ========================================================
    # EXISTING / PROPOSED RWH STRUCTURES
    # ========================================================

    st.subheader(
        "📍 Nearby Existing / Proposed RWH Structures"
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(

        "MGNREGA",

        (
            f"{r['nearest_mgnrega']['distance_m']:.0f} m"

            if r[
                "nearest_mgnrega"
            ]

            else "None"
        ),

    )


    c2.metric(

        "Final-94 Site",

        (
            f"{r['nearest_final_site']['distance_m']:.0f} m"

            if r[
                "nearest_final_site"
            ]

            else "None"
        ),

    )


    c3.metric(

        "Published Site",

        (
            f"{r['nearest_published']['distance_m']:.0f} m"

            if r[
                "nearest_published"
            ]

            else "None"
        ),

    )


    # ========================================================
    # LULC DETAILS
    # ========================================================

    with st.expander(
        "🌿 Field & Watershed LULC"
    ):

        st.markdown(
            "#### Farmer Field"
        )


        st.dataframe(

            r[
                "field_lulc_composition"
            ],

            use_container_width=True,

            hide_index=True,

        )


        st.markdown(
            "#### Watershed"
        )


        st.dataframe(

            r[
                "watershed_lulc_composition"
            ],

            use_container_width=True,

            hide_index=True,

        )


    # ========================================================
    # ALTERNATIVE OPTIONS
    # ========================================================

    if len(
        recommendations
    ) > 1:

        st.subheader(
            "🏗️ Alternative RWH & Management Options"
        )


        for item in recommendations[1:]:

            st.markdown(
                f"""
<div class="card card-blue">

<b>{item['structure']}</b><br>

Priority:
{item['priority']}

<br><br>

<b>Why:</b><br>
{item['reason']}

<br><br>

<b>Management:</b><br>
{item['management']}

</div>
""",
                unsafe_allow_html=True,
            )


    # ========================================================
    # PDF
    # ========================================================

    try:

        pdf_result = (
            r.copy()
        )


        pdf_result[
            "farmer_recommendations"
        ] = [

            (
                item[
                    "structure"
                ],

                (
                    item[
                        "reason"
                    ]

                    +

                    " Management: "

                    +

                    item[
                        "management"
                    ]
                )

            )

            for item
            in recommendations

        ]


        pdf = make_pdf(

            farmer_name,

            village,

            pdf_result,

        )


        st.download_button(

            "📄 Download Complete RWH Report",

            data=pdf,

            file_name=(
                "RiBhoi_RWH_Assessment.pdf"
            ),

            mime="application/pdf",

            use_container_width=True,

        )


    except Exception as error:

        st.warning(

            f"PDF report error: {error}"

        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
The DSS provides GIS-based planning and screening recommendations.

The farmer field and watershed boundaries used in the calculations
are the polygons manually drawn by the user.

LULC is clipped strictly inside the selected polygons and does not
determine the farmer boundary.

CHIRPS rainfall is a coarse-resolution rainfall product; alignment
with the GIS analysis grid does not create true 30-m rainfall
observations.

Final structure location, capacity, foundation, embankment,
spillway and hydraulic design require field survey and engineering
verification.
"""
    )


# ============================================================
# NEW ASSESSMENT
# ============================================================

st.markdown("---")


if st.button(

    "🔄 Start New Farmer Assessment",

    use_container_width=True,

):

    clear_field()

    st.rerun()
