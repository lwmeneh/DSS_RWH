import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from shapely.geometry import shape
from folium.plugins import Draw, Fullscreen, MousePosition, MeasureControl
from streamlit_folium import st_folium

from config.settings import *
from database.db import initialize, save_assessment
from data.loader import load_vectors
from gis.spatial import (
    field_area_ha, nearest_stream_to_point, nearest_feature,
    field_watershed_relation, zonal_stats, band_index_by_name
)
from gis.lulc import (
    detect_agricultural_patch, dominant_lulc, lulc_composition
)
from hydrology.watershed import delineate
from decision.rules import recommend_field, recommend_watershed
from reports.pdf_report import make_pdf
from utils.ui import load_css, render_html

st.set_page_config(
    page_title=APP_SHORT_NAME,
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()
initialize()

required = [
    DEM_PATH, CONDITIONED_DEM, FLOW_DIRECTION,
    CORE_STACK, TERRAIN_STACK, CN_RASTER,
    MEAN_RUNOFF_RASTER, DEPENDABLE_RUNOFF_RASTER,
    RUNOFF_COEFF_RASTER, STREAMS_GPKG, BOUNDARY_GPKG
]

missing = [str(p) for p in required if not p.exists()]

if missing:
    st.error("Required GIS cache is incomplete.")
    st.code("\n".join(missing))
    st.stop()

DATA = load_vectors()
streams_utm = DATA["streams"]
boundary_utm = DATA["boundary"]
boundary_wgs = boundary_utm.to_crs(4326)
lulc_utm = DATA["lulc"]

defaults = {
    "farm_click": None,
    "selected_outlet": None,
    "auto_field": None,
    "manual_field": None,
    "result": None
}

for k,v in defaults.items():
    st.session_state.setdefault(k,v)

def final_field():
    if st.session_state.manual_field is not None:
        return st.session_state.manual_field, "Farmer corrected/manual boundary"
    if st.session_state.auto_field is not None:
        return st.session_state.auto_field, "Automatic LULC agricultural patch"
    return None, None

def parse_drawings(data):
    outlet = None
    polygon = None

    drawings = (data or {}).get("all_drawings") or []

    if not drawings and (data or {}).get("last_active_drawing"):
        drawings = [data["last_active_drawing"]]

    for d in drawings:
        geom = (d or {}).get("geometry", {})
        gtype = geom.get("type")

        if gtype == "Point":
            c = geom["coordinates"]
            outlet = {
                "lon": float(c[0]),
                "lat": float(c[1])
            }

        elif gtype in ("Polygon", "MultiPolygon"):
            polygon = geom

    return outlet, polygon

render_html("hero.html")

st.sidebar.markdown(f"## 💧 {APP_SHORT_NAME}")
st.sidebar.success("Default workflow: Smart Assessment")
farmer_name = st.sidebar.text_input("Farmer name")
village = st.sidebar.text_input("Village")

st.markdown("### 1. Locate the agricultural field")
st.caption(
    "Click anywhere inside the farm. Then press **Detect Agricultural Boundary**. "
    "If the detected LULC patch is too large or does not match the actual field, "
    "draw a corrected polygon using the map drawing tool."
)

center = MAP_CENTER

if st.session_state.farm_click:
    center = [
        st.session_state.farm_click["lat"],
        st.session_state.farm_click["lon"]
    ]

m = folium.Map(
    location=center,
    zoom_start=MAP_ZOOM,
    tiles=None,
    control_scale=True
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
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

folium.GeoJson(
    boundary_wgs.__geo_interface__,
    name="Ri Bhoi Boundary",
    style_function=lambda x: {
        "color":"#FFD54F",
        "weight":2,
        "fillOpacity":0
    }
).add_to(m)

field, source = final_field()

if field:
    folium.GeoJson(
        field,
        name="Confirmed Farmer Field",
        style_function=lambda x: {
            "color":"#00C853",
            "weight":3,
            "fillColor":"#69F0AE",
            "fillOpacity":0.25
        }
    ).add_to(m)

if st.session_state.selected_outlet:
    p = st.session_state.selected_outlet
    folium.CircleMarker(
        [p["lat"], p["lon"]],
        radius=7,
        color="#D50000",
        fill=True,
        fill_opacity=1,
        tooltip="Selected outlet"
    ).add_to(m)

if st.session_state.result and st.session_state.result.get("watershed_geometry"):
    folium.GeoJson(
        st.session_state.result["watershed_geometry"],
        name="Delineated Watershed",
        style_function=lambda x: {
            "color":"#FF6D00",
            "weight":3,
            "fillColor":"#FFC107",
            "fillOpacity":0.18
        }
    ).add_to(m)

Draw(
    export=False,
    draw_options={
        "polyline":False,
        "rectangle":False,
        "circle":False,
        "circlemarker":False,
        "marker":True,
        "polygon":{
            "allowIntersection":False,
            "showArea":True,
            "shapeOptions":{
                "color":"#00C853",
                "weight":3,
                "fillOpacity":0.25
            }
        }
    },
    edit_options={
        "edit":True,
        "remove":True
    }
).add_to(m)

Fullscreen().add_to(m)
MousePosition().add_to(m)
MeasureControl(
    primary_length_unit="meters",
    primary_area_unit="hectares"
).add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

map_data = st_folium(
    m,
    height=680,
    use_container_width=True,
    key="rwh_map",
    returned_objects=[
        "all_drawings",
        "last_active_drawing",
        "last_clicked"
    ]
)

if map_data and map_data.get("last_clicked"):
    c = map_data["last_clicked"]
    st.session_state.farm_click = {
        "lat": float(c["lat"]),
        "lon": float(c["lng"])
    }

outlet, manual_poly = parse_drawings(map_data)

if outlet:
    st.session_state.selected_outlet = outlet

if manual_poly:
    st.session_state.manual_field = manual_poly

c1,c2,c3 = st.columns([1.5,1,1])

with c1:
    if st.button(
        "🌾 Detect Agricultural Boundary",
        type="primary",
        use_container_width=True,
        disabled=(
            lulc_utm is None or
            st.session_state.farm_click is None
        )
    ):
        pt = st.session_state.farm_click

        try:
            det = detect_agricultural_patch(
                pt["lon"],
                pt["lat"],
                lulc_utm
            )

            if not det["is_agriculture"]:
                label = det["class_name"] or "No LULC feature"
                st.warning(
                    f"Selected location is **{label}**. "
                    "Click inside Crop land, Plantation or Shifting Cultivation, "
                    "or draw the field manually."
                )
            else:
                st.session_state.auto_field = det["geometry"]
                st.session_state.manual_field = None
                st.success(
                    f'Automatic agricultural patch: {det["class_name"]}, '
                    f'{det["area_ha"]:.3f} ha'
                )
                st.rerun()

        except Exception as e:
            st.error(str(e))

with c2:
    if st.button(
        "✓ Use Automatic Boundary",
        use_container_width=True,
        disabled=(st.session_state.auto_field is None)
    ):
        st.session_state.manual_field = None
        st.success("Automatic boundary accepted.")

with c3:
    if st.button("Clear Field", use_container_width=True):
        st.session_state.auto_field = None
        st.session_state.manual_field = None
        st.session_state.result = None
        st.rerun()

field, source = final_field()

if field:
    st.success(
        f"Field boundary ready: {field_area_ha(field):.3f} ha | {source}"
    )
    if lulc_utm is not None:
        dom = dominant_lulc(field, lulc_utm)
        if dom:
            st.write(
                f'Dominant LULC: **{dom["name"]}** '
                f'({dom["percent"]:.1f}%)'
            )
else:
    st.info(
        "No field boundary yet. Use automatic detection or draw the actual field manually."
    )

st.markdown("### 2. Select watershed outlet")
st.caption(
    "Use the map marker tool to place the outlet near the drainage. "
    "The DSS searches the mapped stream network within 200 m."
)

if st.session_state.selected_outlet:
    p = st.session_state.selected_outlet
    sm = nearest_stream_to_point(
        p["lon"], p["lat"],
        streams_utm,
        MAX_STREAM_SNAP_M
    )

    if sm:
        st.success(
            f'Mapped stream found {sm["distance_m"]:.1f} m from the selected outlet.'
        )
    else:
        st.info(
            "No mapped stream within 200 m. Field-level assessment will still run."
        )

x,y = st.columns([1,3])

if x.button("Reset", use_container_width=True):
    for k,v in defaults.items():
        st.session_state[k] = v
    st.rerun()

if y.button(
    "💧 RUN AUTOMATIC RWH ASSESSMENT",
    type="primary",
    use_container_width=True
):
    field, source = final_field()

    if field is None:
        st.error("Detect or draw the agricultural field first.")
        st.stop()

    area = field_area_ha(field)

    cn = zonal_stats(
        CN_RASTER,
        field
    )["mean"]

    runoff = zonal_stats(
        MEAN_RUNOFF_RASTER,
        field
    )["mean"]

    dependable = zonal_stats(
        DEPENDABLE_RUNOFF_RASTER,
        field
    )["mean"]

    coeff = zonal_stats(
        RUNOFF_COEFF_RASTER,
        field
    )["mean"]

    slope = np.nan
    slope_band = band_index_by_name(
        CORE_STACK,
        "slope"
    )

    if slope_band:
        slope = zonal_stats(
            CORE_STACK,
            field,
            slope_band
        )["mean"]

    field_utm = gpd.GeoDataFrame(
        geometry=[shape(field)],
        crs=4326
    ).to_crs(WORKING_CRS).geometry.iloc[0]

    stream_distances = streams_utm.geometry.distance(
        field_utm
    )

    field_stream_distance = (
        float(stream_distances.min())
        if len(stream_distances)
        else None
    )

    dom = (
        dominant_lulc(field, lulc_utm)
        if lulc_utm is not None
        else None
    )

    field_comp = (
        lulc_composition(field, lulc_utm)
        if lulc_utm is not None
        else []
    )

    result = {
        "field_boundary_source": source,
        "field_area_ha": area,
        "dominant_lulc_name": (
            dom["name"]
            if dom
            else None
        ),
        "field_lulc_composition": field_comp,
        "field_cn": cn,
        "field_slope": slope,
        "field_runoff_mm": runoff,
        "field_dependable_mm": dependable,
        "field_runoff_coeff": coeff,
        "field_runoff_m3": (
            runoff * area * 10
            if np.isfinite(runoff)
            else np.nan
        ),
        "field_dependable_m3": (
            dependable * area * 10
            if np.isfinite(dependable)
            else np.nan
        ),
        "field_stream_distance_m": field_stream_distance,
        "nearest_mgnrega": nearest_feature(
            field,
            DATA["mgnrega"]
        ),
        "nearest_final_site": nearest_feature(
            field,
            DATA["final94"]
        ),
        "watershed_geometry": None,
        "watershed_area_ha": None,
        "field_watershed_relation": None,
        "watershed_lulc_composition": [],
        "farmer_recommendations": [],
        "watershed_recommendations": []
    }

    result["farmer_recommendations"] = recommend_field(
        area,
        slope,
        runoff,
        cn,
        field_stream_distance,
        result["dominant_lulc_name"]
    )

    if st.session_state.selected_outlet:
        p = st.session_state.selected_outlet

        sm = nearest_stream_to_point(
            p["lon"],
            p["lat"],
            streams_utm,
            MAX_STREAM_SNAP_M
        )

        if sm:
            ws = delineate(
                sm["lon"],
                sm["lat"]
            )

            result["watershed_geometry"] = ws["geometry"]
            result["watershed_area_ha"] = ws["area_ha"]

            result["field_watershed_relation"] = (
                field_watershed_relation(
                    field,
                    ws["geometry"]
                )
            )

            if lulc_utm is not None:
                result["watershed_lulc_composition"] = (
                    lulc_composition(
                        ws["geometry"],
                        lulc_utm
                    )
                )

            result["watershed_recommendations"] = (
                recommend_watershed(
                    result["watershed_lulc_composition"]
                )
            )

    st.session_state.result = result

    if farmer_name or village:
        save_assessment(
            farmer_name,
            village,
            result
        )

    st.rerun()

if st.session_state.result:
    r = st.session_state.result

    st.markdown("---")
    st.header("📊 Automatic RWH Assessment")

    a,b,c,d = st.columns(4)

    a.metric(
        "Field Area",
        f'{r["field_area_ha"]:.3f} ha'
    )

    b.metric(
        "Dominant LULC",
        r["dominant_lulc_name"] or "Not available"
    )

    c.metric(
        "Mean CN-II",
        f'{r["field_cn"]:.1f}'
        if np.isfinite(r["field_cn"])
        else "No data"
    )

    d.metric(
        "Mean Slope",
        f'{r["field_slope"]:.2f}%'
        if np.isfinite(r["field_slope"])
        else "No data"
    )

    a,b,c,d = st.columns(4)

    a.metric(
        "Mean Annual Runoff",
        f'{r["field_runoff_mm"]:.1f} mm'
        if np.isfinite(r["field_runoff_mm"])
        else "No data"
    )

    b.metric(
        "75% Dependable Runoff",
        f'{r["field_dependable_mm"]:.1f} mm'
        if np.isfinite(r["field_dependable_mm"])
        else "No data"
    )

    c.metric(
        "Annual Water",
        f'{r["field_runoff_m3"]:,.0f} m³'
        if np.isfinite(r["field_runoff_m3"])
        else "No data"
    )

    d.metric(
        "Dependable Water",
        f'{r["field_dependable_m3"]:,.0f} m³'
        if np.isfinite(r["field_dependable_m3"])
        else "No data"
    )

    a,b,c = st.columns(3)

    a.metric(
        "Nearest Stream",
        f'{r["field_stream_distance_m"]:.0f} m'
        if r["field_stream_distance_m"] is not None
        else "No data"
    )

    b.metric(
        "Nearest MGNREGA Structure",
        f'{r["nearest_mgnrega"]["distance_m"]:.0f} m'
        if r["nearest_mgnrega"]
        else "Not available"
    )

    c.metric(
        "Nearest Proposed RWH Site",
        f'{r["nearest_final_site"]["distance_m"]:.0f} m'
        if r["nearest_final_site"]
        else "Not available"
    )

    if r["field_lulc_composition"]:
        st.subheader("🌾 Field LULC Composition")
        st.dataframe(
            r["field_lulc_composition"],
            use_container_width=True,
            hide_index=True
        )

    st.subheader("🏗️ Farmer-level RWH Recommendations")

    for name, reason in r["farmer_recommendations"]:
        st.markdown(
            f'<div class="good-card">'
            f'<b>✓ {name}</b><br>{reason}'
            f'</div>',
            unsafe_allow_html=True
        )

    if r["watershed_area_ha"] is not None:
        st.subheader("🌊 DEM Watershed Assessment")

        a,b = st.columns(2)

        a.metric(
            "Watershed Area",
            f'{r["watershed_area_ha"]:.2f} ha'
        )

        if r["field_watershed_relation"]:
            b.metric(
                "Field Inside Watershed",
                f'{r["field_watershed_relation"]["percent_inside"]:.1f}%'
            )

        if r["watershed_lulc_composition"]:
            st.subheader("🌿 Watershed LULC Composition")
            st.dataframe(
                r["watershed_lulc_composition"],
                use_container_width=True,
                hide_index=True
            )

        st.subheader("🧱 Watershed-level Recommendations")

        for name, reason in r["watershed_recommendations"]:
            st.markdown(
                f'<div class="good-card">'
                f'<b>✓ {name}</b><br>{reason}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info(
            "No DEM watershed was delineated. "
            "Field-level assessment remains valid."
        )

    st.markdown(
        '<div class="warn-card">'
        '<b>Planning-level DSS output.</b><br>'
        'Automatic LULC detection identifies a mapped agricultural patch, '
        'not a cadastral/legal landholding boundary. Final structure location, '
        'dimensions, storage, spillway and structural design require field survey '
        'and engineering verification.'
        '</div>',
        unsafe_allow_html=True
    )

    pdf = make_pdf(
        farmer_name,
        village,
        r
    )

    st.download_button(
        "📄 Download RWH Assessment Report",
        data=pdf,
        file_name="RiBhoi_RWH_GIS_DSS_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )
