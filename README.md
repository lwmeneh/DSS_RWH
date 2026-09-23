# Ri Bhoi Automatic Rainwater Harvesting GIS DSS

## Default Smart Assessment
1. Click inside the agricultural field.
2. Detect agricultural LULC polygon automatically.
3. Accept or manually draw/correct the field.
4. Place watershed outlet.
5. Search actual mapped stream within 200 m.
6. Delineate watershed from conditioned DEM + D8 flow direction.
7. Calculate CN, slope, runoff, dependable runoff and water volumes.
8. Calculate nearby existing/proposed RWH structures.
9. Generate LULC-aware field and watershed recommendations.
10. Download PDF report.

## Required cache files

Place these in `data/cache/`:

- DEM_30m.tif
- DEM_conditioned.tif
- flow_direction.tif
- flow_accumulation.tif
- core_stack.tif
- terrain_stack.tif
- CN_II.tif
- mean_runoff_mm.tif
- dependable_runoff_mm.tif
- runoff_coefficient.tif
- streams_utm46.gpkg
- boundary_utm46.gpkg
- 17_RiBhoi_LULC_2021_10K_MixedGeometry.geojson

Optional:
- mgnrega_utm46.gpkg
- final94_utm46.gpkg
- published_sites_utm46.gpkg

## Important LULC classes
The dashboard uses the actual Ri Bhoi detailed LULC field:
`LULC_2022`

Agricultural classes:
- Crop land
- Plantation
- Shifting Cultivation

## One-time LULC optimization

Run:

python -m data.prepare_cache

This repairs/extracts polygon geometries and creates:

data/cache/lulc_2021_10k_clean_utm46.gpkg

## Launch

pip install -r requirements.txt
streamlit run app.py
