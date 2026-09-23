import numpy as np
import geopandas as gpd
import streamlit as st
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pysheds.grid import Grid
from pyproj import Transformer
from config.settings import (
    CONDITIONED_DEM, FLOW_DIRECTION,
    WORKING_CRS, WGS84
)

@st.cache_resource
def load_hydrology():
    grid = Grid.from_raster(str(CONDITIONED_DEM))
    fdir = grid.read_raster(str(FLOW_DIRECTION))
    return grid, fdir

@st.cache_data(show_spinner=False)
def delineate(lon, lat):
    grid, fdir = load_hydrology()

    tx = Transformer.from_crs(
        WGS84,
        WORKING_CRS,
        always_xy=True
    )

    x, y = tx.transform(lon, lat)

    catch = grid.catchment(
        x=x,
        y=y,
        fdir=fdir,
        xytype="coordinate"
    )

    arr = np.asarray(catch).astype("uint8")
    valid = arr > 0

    if valid.sum() == 0:
        raise RuntimeError("No upstream watershed could be delineated.")

    polygons = []

    for geom, value in shapes(
        arr,
        mask=valid,
        transform=grid.affine
    ):
        if value == 1:
            polygons.append(shape(geom))

    poly = unary_union(polygons)

    gdf = gpd.GeoDataFrame(
        geometry=[poly],
        crs=WORKING_CRS
    )

    area_ha = float(
        gdf.geometry.area.iloc[0] / 10000.0
    )

    out = gdf.to_crs(WGS84).geometry.iloc[0]

    return {
        "geometry": mapping(out),
        "area_ha": area_ha
    }
