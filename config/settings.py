from pathlib import Path

APP_NAME = "Ri Bhoi Rainwater Harvesting GIS Decision Support System"
APP_SHORT_NAME = "RiBhoi-RWH GIS DSS"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DB_PATH = PROJECT_ROOT / "data" / "users.db"

WORKING_CRS = "EPSG:32646"
WGS84 = "EPSG:4326"

DEM_PATH = CACHE_DIR / "DEM_30m.tif"
CONDITIONED_DEM = CACHE_DIR / "DEM_conditioned.tif"
FLOW_DIRECTION = CACHE_DIR / "flow_direction.tif"
FLOW_ACCUMULATION = CACHE_DIR / "flow_accumulation.tif"

CORE_STACK = CACHE_DIR / "core_stack.tif"
TERRAIN_STACK = CACHE_DIR / "terrain_stack.tif"

CN_RASTER = CACHE_DIR / "CN_II.tif"
MEAN_RUNOFF_RASTER = CACHE_DIR / "mean_runoff_mm.tif"
DEPENDABLE_RUNOFF_RASTER = CACHE_DIR / "dependable_runoff_mm.tif"
RUNOFF_COEFF_RASTER = CACHE_DIR / "runoff_coefficient.tif"

STREAMS_GPKG = CACHE_DIR / "streams_utm46.gpkg"
BOUNDARY_GPKG = CACHE_DIR / "boundary_utm46.gpkg"
MGNREGA_GPKG = CACHE_DIR / "mgnrega_utm46.gpkg"
FINAL94_GPKG = CACHE_DIR / "final94_utm46.gpkg"
PUBLISHED_GPKG = CACHE_DIR / "published_sites_utm46.gpkg"

# Preferred detailed LULC source, based on the user's 2021 1:10,000 layer.
LULC_GEOJSON = CACHE_DIR / "17_RiBhoi_LULC_2021_10K_MixedGeometry.geojson"
LULC_GPKG = CACHE_DIR / "lulc_2021_10k_clean_utm46.gpkg"
LULC_FIELD = "LULC_2022"

AGRICULTURE_CLASSES = {
    "Crop land",
    "Plantation",
    "Shifting Cultivation",
}

MAX_STREAM_SNAP_M = 200
LOCAL_STREAM_DISPLAY_RADIUS_M = 3000

MAP_CENTER = [25.75, 91.90]
MAP_ZOOM = 13
