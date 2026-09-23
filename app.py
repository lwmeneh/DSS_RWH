import numpy as np
import geopandas as gpd
import streamlit as st
import folium

from shapely.geometry import shape
from folium.plugins import (
    Draw,
    Fullscreen,
    MousePosition,
    MeasureControl
)
from streamlit_folium import st_folium

from config.settings import *

from database.db import (
    initialize,
    save_assessment
)

from data.loader import load_vectors

from gis.spatial import (
    field_area_ha,
    nearest_stream_to_point,
    nearest_feature,
    field_watershed_relation,
    zonal_stats,
    band_index_by_name
)

from gis.lulc import (
    detect_agricultural_patch,
    dominant_lulc,
    lulc_composition
)

from hydrology.watershed import delineate

from decision.rules import (
    recommend_field,
    recommend_watershed
)

from reports.pdf_report import make_pdf

from utils.ui import (
    load_css,
    render_html
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_SHORT_NAME,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

initialize()


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

required = [

    CONDITIONED_DEM,

    FLOW_DIRECTION,

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


streams_utm = DATA["streams"]

boundary_utm = DATA["boundary"]

mgnrega_utm = DATA.get("mgnrega")

final94_utm = DATA.get("final94")

published_utm = DATA.get("published")

lulc_utm = DATA.get("lulc")


boundary_wgs = boundary_utm.to_crs(
    4326
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "farm_click": None,

    "selected_outlet": None,

    "auto_field": None,

    "auto_field_class": None,

    "auto_field_area": None,

    "manual_field": None,

    "field_mode": None,

    "result": None,

    "last_manual_geometry": None,

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_final_field():

    """
    Manual boundary ALWAYS overrides automatic LULC polygon.
    """

    if st.session_state.manual_field is not None:

        return (
            st.session_state.manual_field,
            "Farmer drawn / corrected boundary"
        )

    if st.session_state.auto_field is not None:

        return (
            st.session_state.auto_field,
            "Automatic LULC agricultural patch"
        )

    return None, None


# ------------------------------------------------------------


def parse_drawings(map_output):

    """
    Parse Leaflet drawings.

    Polygon:
        Farmer actual field.

    Point marker:
        Watershed outlet.

    Last drawn polygon is used as farmer boundary.
    """

    outlet = None

    polygon = None


    if map_output is None:

        return outlet, polygon


    drawings = (
        map_output.get("all_drawings")
        or []
    )


    if (
        not drawings
        and map_output.get(
            "last_active_drawing"
        )
    ):

        drawings = [
            map_output[
                "last_active_drawing"
            ]
        ]


    for drawing in drawings:

        if drawing is None:

            continue


        geometry = drawing.get(
            "geometry",
            {}
        )


        geometry_type = geometry.get(
            "type"
        )


        # ------------------------------------
        # Watershed outlet marker
        # ------------------------------------

        if geometry_type == "Point":

            coordinates = geometry.get(
                "coordinates"
            )

            if coordinates:

                outlet = {

                    "lon": float(
                        coordinates[0]
                    ),

                    "lat": float(
                        coordinates[1]
                    )

                }


        # ------------------------------------
        # Farmer field
        # ------------------------------------

        elif geometry_type in (
            "Polygon",
            "MultiPolygon"
        ):

            polygon = geometry


    return outlet, polygon


# ------------------------------------------------------------


def clear_all():

    for key, value in defaults.items():

        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

render_html(
    "hero.html"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    f"## 💧 {APP_SHORT_NAME}"
)


st.sidebar.success(
    "Smart Assessment Mode"
)


farmer_name = st.sidebar.text_input(
    "Farmer name"
)


village = st.sidebar.text_input(
    "Village"
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    "### Map Legend"
)


st.sidebar.markdown(
    """
🟢 **Green** — Farmer field

🟡 **Yellow/Green** — Detected LULC agricultural patch

🟠 **Orange** — Watershed

🔵 **Blue** — Stream

🔴 **Red** — Selected outlet
"""
)


# ============================================================
# STEP 1
# ============================================================

st.header(
    "🌾 Step 1 — Select Agricultural Field"
)


st.info(
    """
Click anywhere inside the agricultural area.

Then click **Detect Agricultural LULC Patch**.

The detected polygon is only a mapped land-use patch.
It may contain several individual farms.

If the polygon is larger than your actual field,
use the **polygon drawing tool on the map**
to draw the actual farmer boundary.
"""
)


# ============================================================
# MAP CENTER
# ============================================================

center = MAP_CENTER


if st.session_state.farm_click:

    center = [

        st.session_state.farm_click[
            "lat"
        ],

        st.session_state.farm_click[
            "lon"
        ]

    ]


# ============================================================
# CREATE MAP
# ============================================================

m = folium.Map(

    location=center,

    zoom_start=MAP_ZOOM,

    tiles=None,

    control_scale=True

)


# ============================================================
# BASEMAPS
# ============================================================

folium.TileLayer(

    tiles=(
        "https://server.arcgisonline.com/"
        "ArcGIS/rest/services/"
        "World_Imagery/MapServer/"
        "tile/{z}/{y}/{x}"
    ),

    attr="Esri",

    name="Satellite",

    show=True

).add_to(m)


folium.TileLayer(

    "OpenStreetMap",

    name="Street Map",

    show=False

).add_to(m)


# ============================================================
# DISTRICT BOUNDARY
# ============================================================

folium.GeoJson(

    boundary_wgs.__geo_interface__,

    name="Ri Bhoi Boundary",

    style_function=lambda feature: {

        "color": "#FFD54F",

        "weight": 2,

        "fillOpacity": 0

    }

).add_to(m)


# ============================================================
# AUTOMATIC LULC PATCH
# ============================================================

if st.session_state.auto_field is not None:

    folium.GeoJson(

        st.session_state.auto_field,

        name="Detected Agricultural LULC Patch",

        style_function=lambda feature: {

            "color": "#9ACD32",

            "weight": 3,

            "dashArray": "7,5",

            "fillColor": "#CDDC39",

            "fillOpacity": 0.15

        },

        tooltip=(
            "Detected agricultural LULC patch "
            "- not cadastral boundary"
        )

    ).add_to(m)


# ============================================================
# FARMER MANUAL FIELD
# ============================================================

if st.session_state.manual_field is not None:

    folium.GeoJson(

        st.session_state.manual_field,

        name="Farmer Confirmed Field",

        style_function=lambda feature: {

            "color": "#00C853",

            "weight": 4,

            "fillColor": "#69F0AE",

            "fillOpacity": 0.30

        },

        tooltip="Farmer-confirmed field boundary"

    ).add_to(m)


# ============================================================
# WATERSHED
# ============================================================

if (

    st.session_state.result

    and

    st.session_state.result.get(
        "watershed_geometry"
    )

):

    folium.GeoJson(

        st.session_state.result[
            "watershed_geometry"
        ],

        name="DEM Watershed",

        style_function=lambda feature: {

            "color": "#FF6D00",

            "weight": 3,

            "fillColor": "#FFC107",

            "fillOpacity": 0.16

        },

        tooltip="DEM-derived watershed"

    ).add_to(m)


# ============================================================
# SELECTED OUTLET
# ============================================================

if st.session_state.selected_outlet:

    p = st.session_state.selected_outlet


    folium.CircleMarker(

        location=[
            p["lat"],
            p["lon"]
        ],

        radius=7,

        color="#D50000",

        weight=3,

        fill=True,

        fill_color="#FF1744",

        fill_opacity=1,

        tooltip="Selected watershed outlet"

    ).add_to(m)


# ============================================================
# DRAW TOOL
# ============================================================

Draw(

    export=False,

    position="topleft",

    draw_options={

        # Disable unwanted tools

        "polyline": False,

        "rectangle": False,

        "circle": False,

        "circlemarker": False,


        # Marker used for watershed outlet

        "marker": {

            "repeatMode": False

        },


        # Polygon used for actual farmer field

        "polygon": {

            "allowIntersection": False,

            "showArea": True,

            "showLength": True,

            "repeatMode": False,

            "shapeOptions": {

                "color": "#00C853",

                "weight": 4,

                "fillColor": "#69F0AE",

                "fillOpacity": 0.30

            }

        }

    },


    edit_options={

        "edit": True,

        "remove": True

    }

).add_to(m)


# ============================================================
# MAP UTILITIES
# ============================================================

Fullscreen().add_to(m)


MousePosition(

    position="bottomright",

    separator=" | ",

    prefix="Coordinates"

).add_to(m)


MeasureControl(

    primary_length_unit="meters",

    primary_area_unit="hectares"

).add_to(m)


folium.LayerControl(

    collapsed=False

).add_to(m)


# ============================================================
# SHOW MAP
# ============================================================

map_data = st_folium(

    m,

    height=680,

    use_container_width=True,

    key="rwh_main_map",

    returned_objects=[

        "all_drawings",

        "last_active_drawing",

        "last_clicked"

    ]

)


# ============================================================
# MAP CLICK
# ============================================================

if (

    map_data

    and

    map_data.get(
        "last_clicked"
    )

):

    click = map_data[
        "last_clicked"
    ]


    clicked_point = {

        "lat": float(
            click["lat"]
        ),

        "lon": float(
            click["lng"]
        )

    }


    # Update only if location has changed

    if (

        st.session_state.farm_click
        != clicked_point

    ):

        st.session_state.farm_click = (
            clicked_point
        )


# ============================================================
# DRAWN FEATURES
# ============================================================

outlet, manual_polygon = (
    parse_drawings(
        map_data
    )
)


# ------------------------------------------------------------
# OUTLET
# ------------------------------------------------------------

if outlet is not None:

    if (

        st.session_state.selected_outlet
        != outlet

    ):

        st.session_state.selected_outlet = (
            outlet
        )


# ------------------------------------------------------------
# MANUAL FIELD
# ------------------------------------------------------------

if manual_polygon is not None:

    if (

        st.session_state.last_manual_geometry
        != manual_polygon

    ):

        st.session_state.manual_field = (
            manual_polygon
        )

        st.session_state.last_manual_geometry = (
            manual_polygon
        )

        st.session_state.field_mode = (
            "manual"
        )

        # Old result invalid after boundary change

        st.session_state.result = None


# ============================================================
# FARM CLICK INFO
# ============================================================

if st.session_state.farm_click:

    pt = st.session_state.farm_click


    st.caption(

        f"Selected location: "

        f"{pt['lat']:.6f}, "

        f"{pt['lon']:.6f}"

    )


# ============================================================
# FIELD CONTROLS
# ============================================================

c1, c2, c3, c4 = st.columns(
    [1.7, 1.4, 1.4, 1]
)


# ------------------------------------------------------------
# DETECT LULC
# ------------------------------------------------------------

with c1:

    detect_disabled = (

        lulc_utm is None

        or

        st.session_state.farm_click is None

    )


    if st.button(

        "🌾 Detect Agricultural LULC Patch",

        type="primary",

        use_container_width=True,

        disabled=detect_disabled

    ):

        point = (
            st.session_state.farm_click
        )


        try:

            detected = (
                detect_agricultural_patch(

                    point["lon"],

                    point["lat"],

                    lulc_utm

                )
            )


            if not detected[
                "is_agriculture"
            ]:

                class_name = (

                    detected[
                        "class_name"
                    ]

                    or

                    "No mapped LULC feature"

                )


                st.warning(

                    f"Selected location is classified as "
                    f"**{class_name}**. "

                    "Please click inside Crop land, "
                    "Plantation or Shifting Cultivation, "
                    "or draw the actual field manually."
                )


            else:

                st.session_state.auto_field = (
                    detected[
                        "geometry"
                    ]
                )


                st.session_state.auto_field_class = (
                    detected[
                        "class_name"
                    ]
                )


                st.session_state.auto_field_area = (
                    detected[
                        "area_ha"
                    ]
                )


                # Do not automatically use the
                # LULC polygon as farmer field

                st.session_state.field_mode = (
                    "suggested"
                )


                st.session_state.result = None


                st.rerun()


        except Exception as error:

            st.error(
                f"LULC detection error: {error}"
            )


# ------------------------------------------------------------
# ACCEPT AUTO
# ------------------------------------------------------------

with c2:

    if st.button(

        "✓ Accept Detected Patch",

        use_container_width=True,

        disabled=(
            st.session_state.auto_field
            is None
        )

    ):

        st.session_state.manual_field = None

        st.session_state.field_mode = (
            "automatic"
        )

        st.session_state.result = None


        st.success(
            "Detected LULC patch accepted as field boundary."
        )


# ------------------------------------------------------------
# USE MANUAL
# ------------------------------------------------------------

with c3:

    if st.button(

        "✏️ Use Drawn Field",

        use_container_width=True,

        disabled=(
            st.session_state.manual_field
            is None
        )

    ):

        st.session_state.field_mode = (
            "manual"
        )

        st.session_state.result = None


        st.success(
            "Farmer-drawn boundary selected."
        )


# ------------------------------------------------------------
# CLEAR
# ------------------------------------------------------------

with c4:

    if st.button(

        "🗑 Clear",

        use_container_width=True

    ):

        clear_all()

        st.rerun()


# ============================================================
# AUTOMATIC PATCH INFORMATION
# ============================================================

if st.session_state.auto_field is not None:

    st.warning(

        f"""
**Detected LULC:** {st.session_state.auto_field_class}

**Mapped patch area:** {st.session_state.auto_field_area:.3f} ha

This automatic polygon represents a mapped agricultural LULC patch.
It is **not necessarily the farmer's cadastral or actual field boundary**.

If it is larger than the actual farm, use the polygon tool on the
left side of the map to draw the real field boundary.
"""
    )


# ============================================================
# FINAL FIELD
# ============================================================

field, boundary_source = (
    get_final_field()
)


if field is not None:

    area_ha = field_area_ha(
        field
    )


    if (

        st.session_state.manual_field
        is not None

    ):

        st.success(

            f"✅ Farmer-drawn field active | "
            f"Area = {area_ha:.3f} ha"
        )


    else:

        st.info(

            f"Automatic LULC patch active | "
            f"Area = {area_ha:.3f} ha"
        )


    # ----------------------------------
    # LULC
    # ----------------------------------

    if lulc_utm is not None:

        dominant = dominant_lulc(

            field,

            lulc_utm

        )


        if dominant:

            st.write(

                f"Dominant LULC: "
                f"**{dominant['name']}** "
                f"({dominant['percent']:.1f}%)"

            )


else:

    st.warning(

        "No field boundary confirmed yet."
    )


# ============================================================
# DRAWING INSTRUCTIONS
# ============================================================

with st.expander(
    "✏️ How to draw the actual farm boundary"
):

    st.markdown(
        """
1. Zoom to the farmer's field using the satellite map.

2. Click the **polygon icon** on the left side of the map.

3. Click each corner of the actual farm.

4. Continue around the field boundary.

5. Click the first point again or double-click the final point.

6. The polygon will close automatically.

7. The manually drawn green polygon overrides the automatic LULC patch.

8. All runoff and RWH calculations will then use the manually drawn field.
"""
    )


# ============================================================
# STEP 2 — WATERSHED OUTLET
# ============================================================

st.markdown("---")


st.header(
    "🌊 Step 2 — Select Watershed Outlet"
)


st.info(
    """
Use the **marker tool** on the map and place the marker near the drainage
outlet associated with the field.

The DSS searches for the nearest mapped stream within 200 m.

If no mapped stream is found, the field-level RWH assessment will still run.
"""
)


# ============================================================
# STREAM SNAP INFORMATION
# ============================================================

snapped_stream = None


if st.session_state.selected_outlet:

    outlet_point = (
        st.session_state.selected_outlet
    )


    snapped_stream = (
        nearest_stream_to_point(

            outlet_point["lon"],

            outlet_point["lat"],

            streams_utm,

            MAX_STREAM_SNAP_M

        )
    )


    if snapped_stream:

        st.success(

            f"✅ Mapped stream found "
            f"{snapped_stream['distance_m']:.1f} m "
            f"from selected outlet."

        )


    else:

        st.warning(

            "No mapped stream found within 200 m. "
            "Field-level analysis can still proceed."

        )


# ============================================================
# ACTION BUTTONS
# ============================================================

reset_col, run_col = st.columns(
    [1, 4]
)


with reset_col:

    if st.button(

        "🔄 Reset",

        use_container_width=True

    ):

        clear_all()

        st.rerun()


with run_col:

    run_assessment = st.button(

        "💧 RUN AUTOMATIC RWH ASSESSMENT",

        type="primary",

        use_container_width=True

    )


# ============================================================
# ANALYSIS
# ============================================================

if run_assessment:

    field, boundary_source = (
        get_final_field()
    )


    # -----------------------------------
    # FIELD CHECK
    # -----------------------------------

    if field is None:

        st.error(

            "Please detect or draw the farmer field first."

        )

        st.stop()


    # -----------------------------------
    # FIELD AREA
    # -----------------------------------

    field_area = field_area_ha(
        field
    )


    # -----------------------------------
    # CN
    # -----------------------------------

    cn_stats = zonal_stats(

        CN_RASTER,

        field

    )


    field_cn = cn_stats[
        "mean"
    ]


    # -----------------------------------
    # ANNUAL RUNOFF
    # -----------------------------------

    runoff_stats = zonal_stats(

        MEAN_RUNOFF_RASTER,

        field

    )


    field_runoff_mm = (
        runoff_stats[
            "mean"
        ]
    )


    # -----------------------------------
    # DEPENDABLE RUNOFF
    # -----------------------------------

    dependable_stats = zonal_stats(

        DEPENDABLE_RUNOFF_RASTER,

        field

    )


    dependable_mm = (
        dependable_stats[
            "mean"
        ]
    )


    # -----------------------------------
    # RUNOFF COEFFICIENT
    # -----------------------------------

    coeff_stats = zonal_stats(

        RUNOFF_COEFF_RASTER,

        field

    )


    runoff_coeff = (
        coeff_stats[
            "mean"
        ]
    )


    # -----------------------------------
    # SLOPE
    # -----------------------------------

    field_slope = np.nan


    slope_band = band_index_by_name(

        CORE_STACK,

        "slope"

    )


    if slope_band:

        slope_stats = zonal_stats(

            CORE_STACK,

            field,

            slope_band

        )


        field_slope = slope_stats[
            "mean"
        ]


    # ========================================================
    # FIELD GEOMETRY PROJECTED
    # ========================================================

    field_geometry_utm = (

        gpd.GeoDataFrame(

            geometry=[
                shape(field)
            ],

            crs=4326

        )

        .to_crs(
            WORKING_CRS
        )

        .geometry.iloc[0]

    )


    # ========================================================
    # STREAM DISTANCE
    # ========================================================

    stream_distance = None


    if (

        streams_utm is not None

        and

        not streams_utm.empty

    ):

        distances = (
            streams_utm.geometry.distance(
                field_geometry_utm
            )
        )


        if len(distances):

            stream_distance = float(
                distances.min()
            )


    # ========================================================
    # DOMINANT LULC
    # ========================================================

    dominant = None

    field_lulc_comp = []


    if lulc_utm is not None:

        dominant = dominant_lulc(

            field,

            lulc_utm

        )


        field_lulc_comp = (
            lulc_composition(

                field,

                lulc_utm

            )
        )


    dominant_name = (

        dominant["name"]

        if dominant

        else None

    )


    # ========================================================
    # WATER VOLUME
    # ========================================================

    annual_runoff_m3 = np.nan


    dependable_runoff_m3 = np.nan


    if np.isfinite(
        field_runoff_mm
    ):

        annual_runoff_m3 = (

            field_runoff_mm

            *

            field_area

            *

            10.0

        )


    if np.isfinite(
        dependable_mm
    ):

        dependable_runoff_m3 = (

            dependable_mm

            *

            field_area

            *

            10.0

        )


    # ========================================================
    # NEAREST STRUCTURES
    # ========================================================

    nearest_mgnrega = (
        nearest_feature(

            field,

            mgnrega_utm

        )

        if mgnrega_utm is not None

        else None
    )


    nearest_final_site = (
        nearest_feature(

            field,

            final94_utm

        )

        if final94_utm is not None

        else None
    )


    nearest_published = (
        nearest_feature(

            field,

            published_utm

        )

        if published_utm is not None

        else None
    )


    # ========================================================
    # FIELD RECOMMENDATIONS
    # ========================================================

    farmer_recommendations = (
        recommend_field(

            field_area,

            field_slope,

            field_runoff_mm,

            field_cn,

            stream_distance,

            dominant_name

        )
    )


    # ========================================================
    # INITIAL RESULT
    # ========================================================

    result = {

        "field_boundary_source":
            boundary_source,

        "field_area_ha":
            field_area,

        "dominant_lulc_name":
            dominant_name,

        "field_lulc_composition":
            field_lulc_comp,

        "field_cn":
            field_cn,

        "field_slope":
            field_slope,

        "field_runoff_mm":
            field_runoff_mm,

        "field_dependable_mm":
            dependable_mm,

        "field_runoff_coeff":
            runoff_coeff,

        "field_runoff_m3":
            annual_runoff_m3,

        "field_dependable_m3":
            dependable_runoff_m3,

        "field_stream_distance_m":
            stream_distance,

        "nearest_mgnrega":
            nearest_mgnrega,

        "nearest_final_site":
            nearest_final_site,

        "nearest_published":
            nearest_published,

        "watershed_geometry":
            None,

        "watershed_area_ha":
            None,

        "field_watershed_relation":
            None,

        "watershed_lulc_composition":
            [],

        "farmer_recommendations":
            farmer_recommendations,

        "watershed_recommendations":
            []

    }


    # ========================================================
    # WATERSHED ANALYSIS
    # ========================================================

    if (

        st.session_state.selected_outlet

        is not None

    ):

        outlet_point = (
            st.session_state.selected_outlet
        )


        snapped = (
            nearest_stream_to_point(

                outlet_point["lon"],

                outlet_point["lat"],

                streams_utm,

                MAX_STREAM_SNAP_M

            )
        )


        if snapped:

            try:

                watershed = delineate(

                    snapped["lon"],

                    snapped["lat"]

                )


                result[
                    "watershed_geometry"
                ] = watershed[
                    "geometry"
                ]


                result[
                    "watershed_area_ha"
                ] = watershed[
                    "area_ha"
                ]


                # --------------------------------
                # FIELD VS WATERSHED
                # --------------------------------

                result[
                    "field_watershed_relation"
                ] = (

                    field_watershed_relation(

                        field,

                        watershed[
                            "geometry"
                        ]

                    )

                )


                # --------------------------------
                # WATERSHED LULC
                # --------------------------------

                if lulc_utm is not None:

                    result[
                        "watershed_lulc_composition"
                    ] = (

                        lulc_composition(

                            watershed[
                                "geometry"
                            ],

                            lulc_utm

                        )

                    )


                # --------------------------------
                # WATERSHED RECOMMENDATIONS
                # --------------------------------

                result[
                    "watershed_recommendations"
                ] = (

                    recommend_watershed(

                        result[
                            "watershed_lulc_composition"
                        ]

                    )

                )


            except Exception as error:

                st.warning(

                    f"Watershed delineation "
                    f"could not be completed: {error}"

                )


    # ========================================================
    # SAVE
    # ========================================================

    st.session_state.result = (
        result
    )


    if farmer_name or village:

        try:

            save_assessment(

                farmer_name,

                village,

                result

            )

        except Exception as error:

            st.warning(

                f"Assessment completed but "
                f"database save failed: {error}"

            )


    st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result:

    result = (
        st.session_state.result
    )


    st.markdown("---")


    st.header(
        "📊 Automatic RWH Assessment"
    )


    # ========================================================
    # FIELD SOURCE
    # ========================================================

    st.caption(

        "Boundary used for analysis: "
        f"**{result['field_boundary_source']}**"

    )


    # ========================================================
    # FIRST METRIC ROW
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(

        "Field Area",

        f"{result['field_area_ha']:.3f} ha"

    )


    c2.metric(

        "Dominant LULC",

        result[
            "dominant_lulc_name"
        ]

        or

        "No data"

    )


    c3.metric(

        "Mean CN-II",

        (
            f"{result['field_cn']:.1f}"

            if np.isfinite(
                result["field_cn"]
            )

            else

            "No data"
        )

    )


    c4.metric(

        "Mean Slope",

        (
            f"{result['field_slope']:.2f}%"

            if np.isfinite(
                result["field_slope"]
            )

            else

            "No data"
        )

    )


    # ========================================================
    # SECOND METRIC ROW
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(

        "Mean Annual Runoff",

        (
            f"{result['field_runoff_mm']:.1f} mm"

            if np.isfinite(
                result[
                    "field_runoff_mm"
                ]
            )

            else

            "No data"
        )

    )


    c2.metric(

        "75% Dependable Runoff",

        (
            f"{result['field_dependable_mm']:.1f} mm"

            if np.isfinite(
                result[
                    "field_dependable_mm"
                ]
            )

            else

            "No data"
        )

    )


    c3.metric(

        "Annual Runoff Volume",

        (
            f"{result['field_runoff_m3']:,.0f} m³"

            if np.isfinite(
                result[
                    "field_runoff_m3"
                ]
            )

            else

            "No data"
        )

    )


    c4.metric(

        "Dependable Water",

        (
            f"{result['field_dependable_m3']:,.0f} m³"

            if np.isfinite(
                result[
                    "field_dependable_m3"
                ]
            )

            else

            "No data"
        )

    )


    # ========================================================
    # THIRD METRIC ROW
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(

        "Runoff Coefficient",

        (
            f"{result['field_runoff_coeff']:.3f}"

            if np.isfinite(
                result[
                    "field_runoff_coeff"
                ]
            )

            else

            "No data"
        )

    )


    c2.metric(

        "Nearest Stream",

        (
            f"{result['field_stream_distance_m']:.0f} m"

            if result[
                "field_stream_distance_m"
            ]

            is not None

            else

            "No data"
        )

    )


    c3.metric(

        "Nearest MGNREGA Structure",

        (
            f"{result['nearest_mgnrega']['distance_m']:.0f} m"

            if result[
                "nearest_mgnrega"
            ]

            else

            "No data"
        )

    )


    c4.metric(

        "Nearest Proposed Site",

        (
            f"{result['nearest_final_site']['distance_m']:.0f} m"

            if result[
                "nearest_final_site"
            ]

            else

            "No data"
        )

    )


    # ========================================================
    # FIELD LULC
    # ========================================================

    if result[
        "field_lulc_composition"
    ]:

        st.subheader(
            "🌾 Field LULC Composition"
        )


        st.dataframe(

            result[
                "field_lulc_composition"
            ],

            use_container_width=True,

            hide_index=True

        )


    # ========================================================
    # FIELD RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "🏗️ Farmer-Level RWH Recommendations"
    )


    for (
        structure,
        explanation
    ) in result[
        "farmer_recommendations"
    ]:

        st.markdown(

            f"""
<div class="good-card">

<b>✓ {structure}</b>

<br>

{explanation}

</div>
""",

            unsafe_allow_html=True

        )


    # ========================================================
    # WATERSHED RESULTS
    # ========================================================

    if (

        result[
            "watershed_area_ha"
        ]

        is not None

    ):

        st.subheader(
            "🌊 DEM Watershed Assessment"
        )


        w1, w2 = st.columns(
            2
        )


        w1.metric(

            "Watershed Area",

            f"{result['watershed_area_ha']:.2f} ha"

        )


        relation = result[
            "field_watershed_relation"
        ]


        if relation:

            w2.metric(

                "Field Inside Watershed",

                f"{relation['percent_inside']:.1f}%"

            )


            if relation[
                "fully_inside"
            ]:

                st.success(

                    "The selected field is fully inside "
                    "the delineated watershed."

                )


            elif relation[
                "intersects"
            ]:

                st.warning(

                    "The selected field only partially "
                    "intersects the delineated watershed."

                )


            else:

                st.warning(

                    "The selected field lies outside "
                    "the delineated watershed."

                )


        # ----------------------------------------
        # WATERSHED LULC
        # ----------------------------------------

        if result[
            "watershed_lulc_composition"
        ]:

            st.subheader(
                "🌿 Watershed LULC Composition"
            )


            st.dataframe(

                result[
                    "watershed_lulc_composition"
                ],

                use_container_width=True,

                hide_index=True

            )


        # ----------------------------------------
        # WATERSHED RECOMMENDATIONS
        # ----------------------------------------

        st.subheader(
            "🧱 Watershed-Level Recommendations"
        )


        for (
            structure,
            explanation
        ) in result[
            "watershed_recommendations"
        ]:

            st.markdown(

                f"""
<div class="good-card">

<b>✓ {structure}</b>

<br>

{explanation}

</div>
""",

                unsafe_allow_html=True

            )


    else:

        st.info(

            "A DEM watershed was not delineated. "
            "The field-level RWH assessment remains available."

        )


    # ========================================================
    # SCIENTIFIC DISCLAIMER
    # ========================================================

    st.markdown(

        """
<div class="warn-card">

<b>Planning-level GIS Decision Support Output</b>

<br><br>

The automatically detected LULC polygon represents a mapped
land-use unit and not a cadastral/legal landholding boundary.

Where available, the farmer-drawn boundary should be preferred
for field-level runoff and rainwater harvesting calculations.

The recommended RWH structures represent GIS-based screening.
Final structure location, dimensions, storage capacity, spillway,
foundation and structural design require field verification,
detailed survey and engineering design.

</div>
""",

        unsafe_allow_html=True

    )


    # ========================================================
    # PDF REPORT
    # ========================================================

    try:

        pdf = make_pdf(

            farmer_name,

            village,

            result

        )


        st.download_button(

            "📄 Download RWH Assessment Report",

            data=pdf,

            file_name=(
                "RiBhoi_RWH_GIS_DSS_Report.pdf"
            ),

            mime="application/pdf",

            use_container_width=True

        )


    except Exception as error:

        st.warning(

            f"PDF report could not be generated: {error}"

        )
