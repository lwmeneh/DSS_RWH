# GeoPond-DSS v2.0

Geospatial Pond Siting and Preliminary Design Decision Support System
ICAR Research Complex for NEH Region, Umiam, Meghalaya

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Folder layout is unchanged from v1:

```
app.py
.streamlit/config.toml
assets/icar_logo.png
data/cache/   (all .tif / .gpkg / .shp layers, same file names as before)
```

## Workflow

1. **Map field** – GPS, place search, draw polygon, or upload GeoJSON / KML / GPKG / zipped SHP.
2. **Verify land use** – 2022 GIS LULC evidence vs. the observed land use (donut + table).
3. **Set assumptions** – sidebar: storage share, land limit, depth, freeboard, side slope, cost rates.
4. **Run analysis** – hydrology zonal stats, weighted-overlay site search, exclusion buffers,
   geology / seepage screening, pond sizing, composite suitability index, cost.
5. **Report** – HTML report (print to PDF), CSV table, GeoJSON layers for QGIS.

## Composite suitability index (0–100)

| Criterion | Weight | Scoring |
|---|---|---|
| Slope | 0.20 | 0–3° = 100 → 24° = 0 |
| Wetness (TWI) | 0.15 | 3 = 15 → 12 = 100 |
| Water availability | 0.20 | dependable runoff 0 mm = 0 → 1200 mm = 100 |
| Seepage | 0.20 | LOW 100 · MODERATE 60 · HIGH 25 |
| Terrain / landform | 0.15 | LOW 100 · MOD 70 · HIGH 40 · SEVERE 10 |
| Shear-zone distance | 0.10 | 0 m = 0 → 500 m = 100 |

Capped at 35 for restricted land use (dense forest, built-up, water) and at 40 for severe terrain.
Classes: S1 ≥ 75, S2 60–75, S3 45–60, N < 45. Weights and thresholds are in
`SUITABILITY_WEIGHTS` and `suitability_index()` — calibrate them against field-validated ponds.

## LULC 2015 + 2022 (v2.1)

The app compares both LULC years inside the field. A 2015 layer is found automatically if:

- the 2022 layer has a column such as `LULC_2015`, or
- `data/cache/` has a file with `2015` in its name (`.gpkg`, `.shp` or `.geojson`); its class
  column is detected from common names (`LULC_2015`, `LULC`, `Class`, `Landuse`, …).

Configured for the NRSC 2015-16 layer `Meghalaya_LULC_2015_16_UTM46N.gpkg` (class column `DESCR_2`),
or its district subset `RiBhoi_LULC_2015_16_UTM46N.gpkg` (18 MB, preferred). The state-wide file is read
only within the district extent. Other names: add the path to `LULC_2015_FILES` near the top of `app.py`.
Sidebar → GIS database → Layer status shows which file and column were used.

**Reconciliation:** if the field was agricultural in 2015 but scrub / open forest / fallow in 2022
(typical of fallow and jhum cycles), *Crop land* (or the 2015 agricultural class) is pre-selected.
The user confirms or overrides it.

**Runoff correction:** the CN and runoff rasters reflect mapped land use. When the confirmed class
differs, CN is shifted by the TR-55 table difference (`CN_BY_LULC`) and runoff is rescaled. The
event depth is calibrated so SCS-CN reproduces the mapped runoff coefficient. It can be switched
off in the sidebar. Mapped and corrected values are both shown and exported.

## Structure type & farmer-declared land use (v2.2)

- **Land use** is declared by the farmer (required before the run). LULC 2015 / 2022 are shown
  only as a collapsed GIS reference; the runoff correction uses the farmer's class.
- **Structure type**:
  - *Permanent farm pond*: earthen pond sized from runoff; liner per the sidebar rule.
  - *Jalkund (plastic-lined)*: ICAR NEH micro-tank, default 5 × 4 × 1.5 m (30,000 L), always lined.
    Liner = bottom + walls + 0.5 m rim anchor, +5% wastage. "Auto" sizes the number of units to the
    storage target, capped at 10 (`JALKUND_MAX_AUTO_UNITS`).
- **Liner**: 500 micron plastic at ₹350/m² by default (sidebar → Cost rates; thickness and rate editable).
