# ============================================================
# RI BHOI RWH GIS DSS
# MOBILE-FIRST FARMER VERSION
# ============================================================

import numpy as np
import geopandas as gpd
import streamlit as st
import folium

from shapely.geometry import shape
from folium.plugins import (
    Draw,
    Fullscreen,
    MousePosition,
    MeasureControl,
)

from streamlit_folium import st_folium


# ============================================================
# PROJECT MODULES
# ============================================================

from config.settings import *

from database.db import (
    initialize,
    save_assessment,
)

from data.loader import load_vectors

from gis.spatial import (
    nearest_feature,
    zonal_stats,
    band_index_by_name,
)

from gis.lulc import (
    lulc_composition,
)

from decision.rules import (
    recommend_field,
    recommend_watershed,
)

from reports.pdf_report import make_pdf

from utils.ui import (
    load_css,
    render_html,
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title=APP_SHORT_NAME,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MOBILE CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   Main application
--------------------------------------------------------- */

.block-container {
    padding-top: 0.8rem;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}


/* ---------------------------------------------------------
   Buttons
--------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {

    min-height: 52px;
    font-size: 17px;
    font-weight: 600;
    border-radius: 12px;
    width: 100%;
}


/* ---------------------------------------------------------
   Mobile
--------------------------------------------------------- */

@media only screen and (max-width: 768px) {

    .block-container {

        padding-left: 0.35rem;
        padding-right: 0.35rem;
        padding-top: 0.35rem;

    }


    h1 {
        font-size: 1.55rem !important;
    }


    h2 {
        font-size: 1.30rem !important;
    }


    h3 {
        font-size: 1.10rem !important;
    }


    .stButton > button,
    .stDownloadButton > button {

        min-height: 58px;

        font-size: 18px;

        padding: 10px;

    }


    [data-testid="stMetric"] {

        padding: 8px !important;

    }


    [data-testid="stMetricValue"] {

        font-size: 1.25rem !important;

    }

}


/* ---------------------------------------------------------
   Status cards
--------------------------------------------------------- */

.mobile-info {

    background: #eef8fb;

    border-left: 5px solid #0b8793;

    padding: 12px;

    border-radius: 10px;

    margin-bottom: 10px;

}


.mobile-success {

    background: #effaf4;

    border-left: 5px solid #16a05d;

    padding: 12px;

    border-radius: 10px;

    margin-bottom: 10px;

}


.mobile-warning {

    background: #fff8e7;

    border-left: 5px solid #f0a000;

    padding: 12px;

    border-radius: 10px;

    margin-bottom: 10px;

}

</style>
""",
    unsafe_allow_html=True,
)


load_css()

initialize()


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required = [

    CORE_STACK,

    CN_RASTER,

    MEAN_RUNOFF_RASTER,

    DEPENDABLE_RUNOFF_RASTER,

    RUNOFF_COEFF_RASTER,

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
# LOAD GIS DATA
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
    .to_crs("EPSG:4326")
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "field_geometry": None,

    "watershed_geometry": None,

    "result": None,

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def last_drawn_polygon(
    map_data
):

    """
    Read ONLY polygons produced by Leaflet Draw.

    Ordinary map taps/clicks are ignored.

    This prevents the district/LULC layer from ever becoming
    the farmer field.
    """

    if not map_data:

        return None


    drawings = (

        map_data.get(
            "all_drawings"
        )

        or []

    )


    polygons = []


    for drawing in drawings:

        if not drawing:

            continue


        geometry = drawing.get(
            "geometry",
            {}
        )


        geometry_type = geometry.get(
            "type"
        )


        if geometry_type in [

            "Polygon",

            "MultiPolygon",

        ]:

            polygons.append(
                geometry
            )


    if not polygons:

        return None


    # Last polygon drawn becomes active
    return polygons[-1]


# ------------------------------------------------------------


def area_ha(
    geometry_geojson
):

    gdf = gpd.GeoDataFrame(

        geometry=[

            shape(
                geometry_geojson
            )

        ],

        crs="EPSG:4326",

    )


    gdf = gdf.to_crs(
        WORKING_CRS
    )


    return float(

        gdf.geometry.iloc[0].area

        / 10000.0

    )


# ------------------------------------------------------------


def geometry_center(
    geometry_geojson
):

    gdf = gpd.GeoDataFrame(

        geometry=[

            shape(
                geometry_geojson
            )

        ],

        crs="EPSG:4326",

    )


    projected = gdf.to_crs(
        WORKING_CRS
    )


    centroid = (
        projected
        .geometry
        .iloc[0]
        .centroid
    )


    centroid_wgs = (

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
            centroid_wgs.y
        ),

        float(
            centroid_wgs.x
        ),

    ]


# ------------------------------------------------------------


def nearest_stream_distance(
    geometry_geojson
):

    if (

        streams_utm is None

        or

        streams_utm.empty

    ):

        return None


    geom = (

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


    distances = (
        streams_utm
        .geometry
        .distance(
            geom
        )
    )


    if distances.empty:

        return None


    return float(
        distances.min()
    )


# ------------------------------------------------------------


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


# ------------------------------------------------------------


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
<div class="mobile-info">

<b>📱 Mobile Farmer Mode</b><br>

Use your finger to draw the actual farm boundary.
The DSS will analyse only the area enclosed by your polygon.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FARMER DETAILS
# ============================================================

with st.expander(
    "👨‍🌾 Farmer Details",
    expanded=False,
):

    farmer_name = st.text_input(
        "Farmer Name"
    )

    village = st.text_input(
        "Village"
    )


# ============================================================
# STEP 1 — FIELD
# ============================================================

st.header(
    "1️⃣ Draw Your Farm"
)


st.markdown(
    """
<div class="mobile-warning">

<b>How to draw using mobile:</b><br><br>

1. Tap the polygon tool <b>⬠</b> on the map.<br>
2. Tap the first corner of your farm.<br>
3. Tap the next corner.<br>
4. Continue tapping around the farm boundary.<br>
5. Tap the first point again to close the polygon.<br><br>

<b>Do not simply tap the map without selecting the polygon tool.</b>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FIELD MAP
# ============================================================

field_map = folium.Map(

    location=MAP_CENTER,

    zoom_start=MAP_ZOOM,

    tiles=None,

    control_scale=True,

    prefer_canvas=True,

)


# ------------------------------------------------------------
# Satellite
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Street Map
# ------------------------------------------------------------

folium.TileLayer(

    "OpenStreetMap",

    name="Street Map",

    show=False,

).add_to(
    field_map
)


# ------------------------------------------------------------
# District outline
#
# IMPORTANT:
# only displayed as reference
# ------------------------------------------------------------

folium.GeoJson(

    boundary_wgs.__geo_interface__,

    name="Ri Bhoi District",

    style_function=lambda feature: {

        "color": "#FFD600",

        "weight": 2,

        "fillOpacity": 0,

        "opacity": 0.8,

    },

).add_to(
    field_map
)


# ------------------------------------------------------------
# Existing field
# ------------------------------------------------------------

if (

    st.session_state[
        "field_geometry"
    ]

    is not None

):

    folium.GeoJson(

        st.session_state[
            "field_geometry"
        ],

        name="My Farm",

        style_function=lambda feature: {

            "color": "#00C853",

            "weight": 5,

            "fillColor": "#69F0AE",

            "fillOpacity": 0.30,

        },

        tooltip="Farmer selected field",

    ).add_to(
        field_map
    )


# ============================================================
# MOBILE DRAW CONTROL
# ============================================================

Draw(

    export=False,

    position="topleft",

    draw_options={

        # ----------------------------------------------------
        # Disable everything except polygon
        # ----------------------------------------------------

        "polyline": False,

        "rectangle": False,

        "circle": False,

        "circlemarker": False,

        "marker": False,


        # ----------------------------------------------------
        # Polygon
        # ----------------------------------------------------

        "polygon": {

            "allowIntersection": False,

            "showArea": True,

            "showLength": True,

            "repeatMode": False,

            "metric": True,

            "shapeOptions": {

                "color": "#00C853",

                "weight": 5,

                "fillColor": "#69F0AE",

                "fillOpacity": 0.30,

            },

        },

    },

    edit_options={

        "edit": True,

        "remove": True,

    },

).add_to(
    field_map
)


Fullscreen(
    position="topright"
).add_to(
    field_map
)


MeasureControl(

    position="bottomleft",

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


# ============================================================
# FIELD MAP RENDER
#
# IMPORTANT:
# last_clicked is NOT returned.
# Ordinary tap therefore cannot select a field.
# ============================================================

field_output = st_folium(

    field_map,

    height=520,

    use_container_width=True,

    key="mobile_field_map",

    returned_objects=[

        "all_drawings",

        "last_active_drawing",

    ],

)


# ============================================================
# READ FIELD POLYGON
# ============================================================

new_field = last_drawn_polygon(
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


        # Field changed:
        # reset watershed and old result

        st.session_state[
            "watershed_geometry"
        ] = None


        st.session_state[
            "result"
        ] = None


# ============================================================
# FIELD DETAILS
# ============================================================

field_geometry = (

    st.session_state[
        "field_geometry"
    ]

)


if field_geometry is not None:

    farm_area = area_ha(
        field_geometry
    )


    st.markdown(
        f"""
<div class="mobile-success">

✅ <b>Farm boundary captured</b><br>

Farm area:
<b>{farm_area:.3f} ha</b>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # FIELD LULC
    # ========================================================

    if lulc_utm is not None:

        field_lulc = (
            lulc_composition(

                field_geometry,

                lulc_utm,

            )
        )


        if field_lulc:

            dominant_lulc = (
                field_lulc[0]
            )


            st.markdown(
                f"""
<div class="mobile-info">

🌾 <b>Dominant Land Use:</b>
{dominant_lulc["name"]}

<br>

Coverage:
{dominant_lulc["percent"]:.1f}%

</div>
""",
                unsafe_allow_html=True,
            )


            with st.expander(
                "View complete farm LULC"
            ):

                st.dataframe(

                    field_lulc,

                    use_container_width=True,

                    hide_index=True,

                )


    if st.button(
        "🗑️ Redraw Farm Boundary",
        use_container_width=True,
    ):

        clear_field()

        st.rerun()


else:

    st.info(
        "Draw and close the green farm polygon to continue."
    )


# ============================================================
# STEP 2 — WATERSHED
# ============================================================

if field_geometry is not None:

    st.markdown("---")


    st.header(
        "2️⃣ Draw Watershed / Catchment"
    )


    st.markdown(
        """
<div class="mobile-info">

Use the same method:

<b>Tap polygon tool → tap watershed boundary points → close polygon.</b>

The orange polygon should represent the contributing
catchment surrounding the farm or proposed RWH location.

</div>
""",
        unsafe_allow_html=True,
    )


    map_center = geometry_center(
        field_geometry
    )


    watershed_map = folium.Map(

        location=map_center,

        zoom_start=14,

        tiles=None,

        control_scale=True,

        prefer_canvas=True,

    )


    # ========================================================
    # BASEMAP
    # ========================================================

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


    folium.TileLayer(

        "OpenStreetMap",

        name="Street Map",

        show=False,

    ).add_to(
        watershed_map
    )


    # ========================================================
    # FARM REFERENCE
    # ========================================================

    folium.GeoJson(

        field_geometry,

        name="My Farm",

        style_function=lambda feature: {

            "color": "#00C853",

            "weight": 5,

            "fillColor": "#69F0AE",

            "fillOpacity": 0.25,

        },

    ).add_to(
        watershed_map
    )


    # ========================================================
    # EXISTING WATERSHED
    # ========================================================

    if (

        st.session_state[
            "watershed_geometry"
        ]

        is not None

    ):

        folium.GeoJson(

            st.session_state[
                "watershed_geometry"
            ],

            name="My Watershed",

            style_function=lambda feature: {

                "color": "#FF6D00",

                "weight": 5,

                "fillColor": "#FFB74D",

                "fillOpacity": 0.20,

            },

        ).add_to(
            watershed_map
        )


    # ========================================================
    # WATERSHED DRAW TOOL
    # ========================================================

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

                "repeatMode": False,

                "shapeOptions": {

                    "color": "#FF6D00",

                    "weight": 5,

                    "fillColor": "#FFB74D",

                    "fillOpacity": 0.20,

                },

            },

        },

        edit_options={

            "edit": True,

            "remove": True,

        },

    ).add_to(
        watershed_map
    )


    Fullscreen(
        position="topright"
    ).add_to(
        watershed_map
    )


    MeasureControl(

        position="bottomleft",

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


    # ========================================================
    # RENDER WATERSHED MAP
    # ========================================================

    watershed_output = st_folium(

        watershed_map,

        height=520,

        use_container_width=True,

        key="mobile_watershed_map",

        returned_objects=[

            "all_drawings",

            "last_active_drawing",

        ],

    )


    # ========================================================
    # SAVE WATERSHED
    # ========================================================

    new_watershed = (
        last_drawn_polygon(
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


    watershed_geometry = (

        st.session_state[
            "watershed_geometry"
        ]

    )


    if watershed_geometry is not None:

        catchment_area = (
            area_ha(
                watershed_geometry
            )
        )


        st.markdown(
            f"""
<div class="mobile-success">

✅ <b>Watershed captured</b><br>

Watershed area:
<b>{catchment_area:.2f} ha</b>

</div>
""",
            unsafe_allow_html=True,
        )


        if lulc_utm is not None:

            watershed_lulc = (
                lulc_composition(

                    watershed_geometry,

                    lulc_utm,

                )
            )


            if watershed_lulc:

                with st.expander(
                    "🌿 View Watershed LULC"
                ):

                    st.dataframe(

                        watershed_lulc,

                        use_container_width=True,

                        hide_index=True,

                    )


        if st.button(
            "🗑️ Redraw Watershed",
            use_container_width=True,
        ):

            clear_watershed()

            st.rerun()


    else:

        st.info(
            "Draw and close the orange watershed polygon."
        )


# ============================================================
# STEP 3 — ANALYSIS
# ============================================================

if (

    st.session_state[
        "field_geometry"
    ]

    is not None

    and

    st.session_state[
        "watershed_geometry"
    ]

    is not None

):

    st.markdown("---")


    st.header(
        "3️⃣ Analyse Water & RWH Structure"
    )


    st.markdown(
        """
<div class="mobile-info">

The DSS will now analyse:

• Farm area  
• Watershed area  
• Land use  
• Slope  
• CN-II  
• Runoff depth  
• Dependable runoff  
• Available water volume  
• Distance to stream  
• Existing RWH structures  
• Suitable structure recommendation

</div>
""",
        unsafe_allow_html=True,
    )


    analyse = st.button(

        "💧 RUN RWH ASSESSMENT",

        type="primary",

        use_container_width=True,

    )


    if analyse:

        field = (
            st.session_state[
                "field_geometry"
            ]
        )


        watershed = (
            st.session_state[
                "watershed_geometry"
            ]
        )


        # ====================================================
        # AREA
        # ====================================================

        field_area = area_ha(
            field
        )


        watershed_area = area_ha(
            watershed
        )


        # ====================================================
        # FIELD LULC
        # ====================================================

        field_lulc = (

            lulc_composition(

                field,

                lulc_utm,

            )

            if lulc_utm is not None

            else []

        )


        dominant_name = (

            field_lulc[0]["name"]

            if field_lulc

            else None

        )


        # ====================================================
        # WATERSHED LULC
        # ====================================================

        watershed_lulc = (

            lulc_composition(

                watershed,

                lulc_utm,

            )

            if lulc_utm is not None

            else []

        )


        # ====================================================
        # CN
        # ====================================================

        cn = zonal_stats(

            CN_RASTER,

            field,

        )["mean"]


        # ====================================================
        # RUNOFF
        # ====================================================

        runoff_mm = zonal_stats(

            MEAN_RUNOFF_RASTER,

            field,

        )["mean"]


        # ====================================================
        # DEPENDABLE RUNOFF
        # ====================================================

        dependable_mm = zonal_stats(

            DEPENDABLE_RUNOFF_RASTER,

            field,

        )["mean"]


        # ====================================================
        # RUNOFF COEFFICIENT
        # ====================================================

        runoff_coefficient = zonal_stats(

            RUNOFF_COEFF_RASTER,

            field,

        )["mean"]


        # ====================================================
        # SLOPE
        # ====================================================

        slope = np.nan


        slope_band = (
            band_index_by_name(

                CORE_STACK,

                "slope",

            )
        )


        if slope_band:

            slope = zonal_stats(

                CORE_STACK,

                field,

                slope_band,

            )["mean"]


        # ====================================================
        # WATER VOLUME
        # ====================================================

        annual_water_m3 = np.nan


        dependable_water_m3 = np.nan


        if np.isfinite(
            runoff_mm
        ):

            annual_water_m3 = (

                runoff_mm

                * field_area

                * 10.0

            )


        if np.isfinite(
            dependable_mm
        ):

            dependable_water_m3 = (

                dependable_mm

                * field_area

                * 10.0

            )


        # ====================================================
        # STREAM DISTANCE
        # ====================================================

        stream_distance = (
            nearest_stream_distance(
                field
            )
        )


        # ====================================================
        # EXISTING STRUCTURES
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
        # STRUCTURE RECOMMENDATION
        # ====================================================

        field_recommendations = (
            recommend_field(

                field_area,

                slope,

                runoff_mm,

                cn,

                stream_distance,

                dominant_name,

            )
        )


        watershed_recommendations = (
            recommend_watershed(

                watershed_lulc

            )
        )


        # ====================================================
        # RESULT
        # ====================================================

        result = {

            "field_boundary_source":
                "Farmer mobile-drawn polygon",

            "field_area_ha":
                field_area,

            "watershed_area_ha":
                watershed_area,

            "dominant_lulc_name":
                dominant_name,

            "field_lulc_composition":
                field_lulc,

            "watershed_lulc_composition":
                watershed_lulc,

            "field_cn":
                cn,

            "field_slope":
                slope,

            "field_runoff_mm":
                runoff_mm,

            "field_dependable_mm":
                dependable_mm,

            "field_runoff_coeff":
                runoff_coefficient,

            "field_runoff_m3":
                annual_water_m3,

            "field_dependable_m3":
                dependable_water_m3,

            "field_stream_distance_m":
                stream_distance,

            "nearest_mgnrega":
                nearest_mgnrega,

            "nearest_final_site":
                nearest_final,

            "nearest_published":
                nearest_published,

            "farmer_recommendations":
                field_recommendations,

            "watershed_recommendations":
                watershed_recommendations,

        }


        st.session_state[
            "result"
        ] = result


        if farmer_name or village:

            try:

                save_assessment(

                    farmer_name,

                    village,

                    result,

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


    st.markdown("---")


    st.header(
        "✅ Your RWH Recommendation"
    )


    # ========================================================
    # MAIN STRUCTURE FIRST
    # ========================================================

    if r[
        "farmer_recommendations"
    ]:

        best_structure = (
            r[
                "farmer_recommendations"
            ][0]
        )


        st.success(

            f"""
### 🏗️ Recommended Structure

**{best_structure[0]}**

{best_structure[1]}
"""
        )


    # ========================================================
    # BASIC RESULTS
    # ========================================================

    st.subheader(
        "📍 Farm"
    )


    st.metric(

        "Farm Area",

        f"{r['field_area_ha']:.3f} ha",

    )


    st.metric(

        "Land Use",

        r[
            "dominant_lulc_name"
        ]

        or

        "Not available",

    )


    # ========================================================
    # WATER
    # ========================================================

    st.subheader(
        "💧 Water Availability"
    )


    st.metric(

        "Annual Runoff",

        (

            f"{r['field_runoff_mm']:.1f} mm"

            if np.isfinite(
                r["field_runoff_mm"]
            )

            else "No data"

        ),

    )


    st.metric(

        "Annual Available Water",

        (

            f"{r['field_runoff_m3']:,.0f} m³"

            if np.isfinite(
                r["field_runoff_m3"]
            )

            else "No data"

        ),

    )


    st.metric(

        "75% Dependable Water",

        (

            f"{r['field_dependable_m3']:,.0f} m³"

            if np.isfinite(
                r["field_dependable_m3"]
            )

            else "No data"

        ),

    )


    # ========================================================
    # SITE CHARACTERISTICS
    # ========================================================

    with st.expander(
        "📊 Detailed Site Information"
    ):

        st.metric(

            "Watershed Area",

            f"{r['watershed_area_ha']:.2f} ha",

        )


        st.metric(

            "CN-II",

            (

                f"{r['field_cn']:.1f}"

                if np.isfinite(
                    r["field_cn"]
                )

                else "No data"

            ),

        )


        st.metric(

            "Mean Slope",

            (

                f"{r['field_slope']:.2f}%"

                if np.isfinite(
                    r["field_slope"]
                )

                else "No data"

            ),

        )


        st.metric(

            "Runoff Coefficient",

            (

                f"{r['field_runoff_coeff']:.3f}"

                if np.isfinite(
                    r["field_runoff_coeff"]
                )

                else "No data"

            ),

        )


        st.metric(

            "Distance to Stream",

            (

                f"{r['field_stream_distance_m']:.0f} m"

                if r[
                    "field_stream_distance_m"
                ]
                is not None

                else "Not available"

            ),

        )


    # ========================================================
    # OTHER STRUCTURES
    # ========================================================

    if len(
        r[
            "farmer_recommendations"
        ]
    ) > 1:

        with st.expander(
            "🏗️ Alternative RWH Options"
        ):

            for (
                structure,
                reason
            ) in r[
                "farmer_recommendations"
            ][1:]:

                st.info(

                    f"**{structure}**\n\n"
                    f"{reason}"

                )


    # ========================================================
    # WATERSHED TREATMENTS
    # ========================================================

    with st.expander(
        "🌊 Watershed Treatment Recommendations"
    ):

        for (
            structure,
            reason
        ) in r[
            "watershed_recommendations"
        ]:

            st.info(

                f"**{structure}**\n\n"
                f"{reason}"

            )


    # ========================================================
    # LULC
    # ========================================================

    with st.expander(
        "🌾 Land Use Details"
    ):

        st.markdown(
            "**Farm LULC**"
        )


        if r[
            "field_lulc_composition"
        ]:

            st.dataframe(

                r[
                    "field_lulc_composition"
                ],

                use_container_width=True,

                hide_index=True,

            )


        st.markdown(
            "**Watershed LULC**"
        )


        if r[
            "watershed_lulc_composition"
        ]:

            st.dataframe(

                r[
                    "watershed_lulc_composition"
                ],

                use_container_width=True,

                hide_index=True,

            )


    # ========================================================
    # EXISTING STRUCTURES
    # ========================================================

    with st.expander(
        "📍 Existing / Proposed Structures Nearby"
    ):

        if r[
            "nearest_mgnrega"
        ]:

            st.write(

                "Nearest MGNREGA structure: "

                f"**{r['nearest_mgnrega']['distance_m']:.0f} m**"

            )


        if r[
            "nearest_final_site"
        ]:

            st.write(

                "Nearest proposed RWH site: "

                f"**{r['nearest_final_site']['distance_m']:.0f} m**"

            )


        if r[
            "nearest_published"
        ]:

            st.write(

                "Nearest published candidate: "

                f"**{r['nearest_published']['distance_m']:.0f} m**"

            )


    # ========================================================
    # REPORT
    # ========================================================

    try:

        pdf = make_pdf(

            farmer_name,

            village,

            r,

        )


        st.download_button(

            "📄 Download Farmer RWH Report",

            data=pdf,

            file_name=(
                "RiBhoi_RWH_Farmer_Report.pdf"
            ),

            mime="application/pdf",

            use_container_width=True,

        )


    except Exception as error:

        st.warning(

            f"Report generation error: {error}"

        )


    st.warning(
        """
The DSS provides planning-level recommendations.

Final RWH structure dimensions, embankment,
spillway, foundation and hydraulic design
must be verified through field investigation.
"""
    )


# ============================================================
# RESET EVERYTHING
# ============================================================

st.markdown("---")


if st.button(

    "🔄 Start New Farmer Assessment",

    use_container_width=True,

):

    clear_field()

    st.rerun()
