from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from config.settings import (
    CACHE_DIR, LULC_GEOJSON, LULC_GPKG, LULC_FIELD,
    WORKING_CRS
)

def polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(polygon_parts(g))
        return out
    return []

def prepare_lulc():
    if LULC_GPKG.exists():
        print("Already exists:", LULC_GPKG)
        return

    if not LULC_GEOJSON.exists():
        raise FileNotFoundError(
            f"Missing LULC file: {LULC_GEOJSON}"
        )

    gdf = gpd.read_file(LULC_GEOJSON)

    if LULC_FIELD not in gdf.columns:
        raise RuntimeError(
            f"Expected LULC field '{LULC_FIELD}' not found. "
            f"Available fields: {list(gdf.columns)}"
        )

    if gdf.crs is None:
        raise RuntimeError("LULC layer has no CRS.")

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    rows = []
    for _, row in gdf.iterrows():
        for part in polygon_parts(row.geometry):
            new = row.copy()
            new.geometry = part
            rows.append(new)

    clean = gpd.GeoDataFrame(rows, crs=gdf.crs)
    clean = clean[clean.geometry.notna() & ~clean.geometry.is_empty].copy()
    clean = clean.to_crs(WORKING_CRS)

    keep = [c for c in clean.columns if c != "geometry"]
    clean = clean[keep + ["geometry"]]

    clean.to_file(LULC_GPKG, driver="GPKG")
    print("Created:", LULC_GPKG)
    print("Features:", len(clean))
    print("Classes:")
    print(clean[LULC_FIELD].value_counts(dropna=False))

if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prepare_lulc()
