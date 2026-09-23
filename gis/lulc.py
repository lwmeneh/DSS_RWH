import geopandas as gpd
from shapely.geometry import Point, mapping, shape
from config.settings import (
    AGRICULTURE_CLASSES, LULC_FIELD, WORKING_CRS, WGS84
)

def detect_agricultural_patch(lon, lat, lulc_utm):
    if lulc_utm is None or lulc_utm.empty:
        raise RuntimeError("LULC layer is not loaded.")

    p = gpd.GeoSeries([Point(lon, lat)], crs=WGS84).to_crs(WORKING_CRS).iloc[0]

    ids = list(lulc_utm.sindex.intersection(p.bounds))
    if not ids:
        return {
            "is_agriculture": False,
            "class_name": None,
            "geometry": None,
            "area_ha": None
        }

    cand = lulc_utm.iloc[ids]
    cand = cand[cand.geometry.covers(p)]

    if cand.empty:
        return {
            "is_agriculture": False,
            "class_name": None,
            "geometry": None,
            "area_ha": None
        }

    row = cand.iloc[0]
    class_name = str(row.get(LULC_FIELD, ""))

    if class_name not in AGRICULTURE_CLASSES:
        return {
            "is_agriculture": False,
            "class_name": class_name,
            "geometry": None,
            "area_ha": None
        }

    geom = row.geometry
    area_ha = float(geom.area / 10000.0)
    out = gpd.GeoSeries([geom], crs=WORKING_CRS).to_crs(WGS84).iloc[0]

    return {
        "is_agriculture": True,
        "class_name": class_name,
        "geometry": mapping(out),
        "area_ha": area_ha
    }

def lulc_composition(geometry_wgs, lulc_utm):
    if lulc_utm is None or lulc_utm.empty:
        return []

    geom = gpd.GeoDataFrame(
        geometry=[shape(geometry_wgs)],
        crs=WGS84
    ).to_crs(WORKING_CRS).geometry.iloc[0]

    ids = list(lulc_utm.sindex.intersection(geom.bounds))
    if not ids:
        return []

    cand = lulc_utm.iloc[ids]
    cand = cand[cand.intersects(geom)]

    if cand.empty:
        return []

    total = geom.area
    out = []

    for cls, grp in cand.groupby(LULC_FIELD):
        area = 0.0
        for g in grp.geometry:
            area += g.intersection(geom).area
        if area > 0:
            out.append({
                "name": str(cls),
                "area_ha": float(area / 10000.0),
                "percent": float(area / total * 100.0)
            })

    return sorted(out, key=lambda x: x["percent"], reverse=True)

def dominant_lulc(geometry_wgs, lulc_utm):
    comp = lulc_composition(geometry_wgs, lulc_utm)
    return comp[0] if comp else None
