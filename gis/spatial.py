import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, shape, mapping
from shapely.ops import nearest_points
from config.settings import WORKING_CRS, WGS84

def projected_geometry(geometry_wgs):
    return gpd.GeoDataFrame(
        geometry=[shape(geometry_wgs)],
        crs=WGS84
    ).to_crs(WORKING_CRS).geometry.iloc[0]

def field_area_ha(geometry_wgs):
    return float(projected_geometry(geometry_wgs).area / 10000.0)

def nearest_stream_to_point(lon, lat, streams_utm, max_distance=200):
    p = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(WORKING_CRS).iloc[0]
    search = p.buffer(max_distance)
    ids = list(streams_utm.sindex.intersection(search.bounds))
    if not ids:
        return None
    cand = streams_utm.iloc[ids]
    cand = cand[cand.intersects(search)]
    if cand.empty:
        return None
    distances = cand.geometry.distance(p)
    idx = distances.idxmin()
    dist = float(distances.loc[idx])
    if dist > max_distance:
        return None
    snapped = nearest_points(p, cand.loc[idx].geometry)[1]
    wgs = gpd.GeoSeries([snapped], crs=WORKING_CRS).to_crs(WGS84).iloc[0]
    return {
        "distance_m": dist,
        "lon": float(wgs.x),
        "lat": float(wgs.y)
    }

def nearest_feature(field_geometry, layer_utm):
    if layer_utm is None or layer_utm.empty:
        return None
    field = projected_geometry(field_geometry)
    distances = layer_utm.geometry.distance(field)
    idx = distances.idxmin()
    target = layer_utm.loc[idx].geometry
    _, pt = nearest_points(field, target)
    wgs = gpd.GeoSeries([pt], crs=WORKING_CRS).to_crs(WGS84).iloc[0]
    attrs = {}
    for c in layer_utm.columns:
        if c != "geometry":
            try:
                attrs[c] = str(layer_utm.loc[idx, c])
            except Exception:
                pass
    return {
        "distance_m": float(distances.loc[idx]),
        "lon": float(wgs.x),
        "lat": float(wgs.y),
        "attributes": attrs
    }

def field_watershed_relation(field_geometry, watershed_geometry):
    field = projected_geometry(field_geometry)
    watershed = projected_geometry(watershed_geometry)
    inter = field.intersection(watershed)
    pct = 0.0 if field.area == 0 else float(inter.area / field.area * 100)
    return {
        "fully_inside": bool(watershed.covers(field)),
        "intersects": bool(watershed.intersects(field)),
        "percent_inside": pct
    }

def zonal_stats(raster_path, geometry_wgs, band=1):
    gdf = gpd.GeoDataFrame(
        geometry=[shape(geometry_wgs)],
        crs=WGS84
    )
    with rasterio.open(raster_path) as src:
        gdf = gdf.to_crs(src.crs)
        arr, _ = rio_mask(
            src,
            [mapping(gdf.geometry.iloc[0])],
            crop=True,
            indexes=[band],
            filled=False
        )
        values = arr[0].compressed()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {
                "mean": np.nan,
                "median": np.nan,
                "min": np.nan,
                "max": np.nan
            }
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }

def band_index_by_name(raster_path, keyword):
    with rasterio.open(raster_path) as src:
        for i, desc in enumerate(src.descriptions, start=1):
            if desc and keyword.lower() in desc.lower():
                return i
    return None
