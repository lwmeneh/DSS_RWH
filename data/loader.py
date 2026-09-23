import geopandas as gpd
import streamlit as st
from config.settings import (
    STREAMS_GPKG, BOUNDARY_GPKG, MGNREGA_GPKG,
    FINAL94_GPKG, PUBLISHED_GPKG, LULC_GPKG, LULC_GEOJSON,
    WORKING_CRS
)

@st.cache_resource
def load_vectors():
    lulc = None
    if LULC_GPKG.exists():
        lulc = gpd.read_file(LULC_GPKG)
    elif LULC_GEOJSON.exists():
        lulc = gpd.read_file(LULC_GEOJSON)
        if lulc.crs is not None:
            lulc = lulc.to_crs(WORKING_CRS)

    return {
        "streams": gpd.read_file(STREAMS_GPKG),
        "boundary": gpd.read_file(BOUNDARY_GPKG),
        "mgnrega": gpd.read_file(MGNREGA_GPKG) if MGNREGA_GPKG.exists() else None,
        "final94": gpd.read_file(FINAL94_GPKG) if FINAL94_GPKG.exists() else None,
        "published": gpd.read_file(PUBLISHED_GPKG) if PUBLISHED_GPKG.exists() else None,
        "lulc": lulc
    }
