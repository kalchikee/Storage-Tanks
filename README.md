# Ghost Infrastructure
## Mapping Abandoned Underground Storage Tanks Leaching Into Groundwater

**Study area:** Harris County, Texas (Houston metro)
**Stack:** Python · GeoPandas · pysheds · pykrige · Leaflet.js · PostGIS-ready outputs

---

### Project Overview

Hundreds of thousands of decommissioned gas stations, dry cleaners, and industrial
facilities sit atop buried underground storage tanks (USTs) that were never properly
remediated. This project constructs an integrated, spatially explicit contamination
risk model connecting known leaking UST sites to downstream drinking-water wells,
using soil permeability modelling and DEM-derived groundwater flow vectors to predict
which wells are most likely to be affected before contamination is detected.

---

### Repository Structure

```
Storage Tanks/
├── config.py                     # Central configuration (paths, CRS, thresholds)
├── requirements.txt              # Python dependencies
├── run_pipeline.py               # Master runner — executes all 6 steps
│
├── scripts/
│   ├── 01_data_acquisition.py   # Download / generate all raw datasets
│   ├── 02_preprocessing.py      # Project, clean, and harmonise data
│   ├── 03_hydro_modeling.py     # Flow routing & contamination risk rasters
│   ├── 04_risk_scoring.py       # Well vulnerability, hot-spots, EJ analysis
│   ├── 05_static_maps.py        # Publication-quality cartographic maps
│   └── 06_web_map.py            # Export GeoJSON for the interactive map
│
├── data/
│   ├── raw/                     # Downloaded source data (gitignored)
│   └── processed/               # Analysis-ready GeoPackages and rasters
│
├── output/
│   ├── maps/                    # PNG maps (01–05)
│   └── reports/                 # CSV / text analysis reports
│
└── web/
    ├── index.html               # Standalone Leaflet.js interactive map
    └── data/                    # GeoJSON feeds for the web map
```

---

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (≈10–20 min first run; data auto-generated if downloads fail)
python run_pipeline.py

# 3. Open the interactive map
#    Navigate to web/index.html in your browser
#    (use a local server if fetch() is blocked by CORS: python -m http.server 8000 --directory web)
```

Resume from a specific step:
```bash
python run_pipeline.py --from 3   # resume from hydrological modeling
python run_pipeline.py --only 5   # regenerate static maps only
```

---

### Data Sources

| Dataset | Source | Access |
|---|---|---|
| Leaking UST sites | TCEQ PST Database | Free download |
| EPA Brownfields | EPA Brownfields / ECHO API | Free API |
| Drinking-water wells | EPA SDWIS | Free download |
| Groundwater levels | USGS NWIS | Free API |
| Soil permeability (Ksat) | SSURGO / NRCS Soil Data Access | Free API |
| Digital Elevation Model | USGS 3DEP (1/3 arc-second) | Free download |
| Land use / land cover | NLCD 2021 | Free download |

Real data sources are attempted automatically; realistic synthetic data is generated
as a fallback so the pipeline runs end-to-end without manual intervention.

---

### Methodology

**Phase 1 — Data Acquisition & Preprocessing**
Download raw datasets; project everything to NAD83 / UTM Zone 14N (EPSG:32614);
build Ksat raster from SSURGO; krige groundwater-depth surface from USGS monitoring wells.

**Phase 2 — Hydrological Flow Modelling**
Subtract kriged groundwater depth from DEM to derive the water-table surface; condition
surface with pysheds (pit-fill, depression-fill, flat resolution); compute D8 flow
direction and accumulation; build weighted cost surface (Ksat × NLCD imperviousness × slope);
propagate contamination risk downstream from each LUST source using Gaussian plume convolution
along the flow network; classify plume probability zones (High / Moderate / Low).

**Phase 3 — Risk Scoring & Well Vulnerability**
Score each drinking-water well on five factors:
- Number of upstream LUST sites within 3 km (25 %)
- Distance-weighted LUST hazard score (25 %)
- Modelled contamination risk at well location (25 %)
- Soil Ksat at well location (15 %)
- Well depth / aquifer protection (10 %)

Validate against simulated USGS groundwater sampling detections.
Apply Getis-Ord Gi* hot-spot analysis to identify significant clusters.
Cross-reference with Census ACS data for environmental justice implications.

**Phase 4 — Visualisation**
Five publication-quality static maps (PNG) + interactive Leaflet.js web map with:
- Clustered LUST markers, risk zone polygons, well vulnerability choropleth
- Prioritised remediation sidebar, EJ community highlighting, info panel

---

### Key Deliverables

| Deliverable | Path |
|---|---|
| Interactive web map | `web/index.html` |
| LUST overview map | `output/maps/01_lust_overview.png` |
| Contamination risk map | `output/maps/02_contamination_risk.png` |
| Well vulnerability map | `output/maps/03_well_vulnerability.png` |
| Environmental justice map | `output/maps/04_environmental_justice.png` |
| Remediation priority map | `output/maps/05_remediation_priority.png` |
| Prioritised LUST ranking | `output/reports/remediation_priority.csv` |
| Model validation report | `output/reports/model_validation.csv` |
| Analysis summary | `output/reports/analysis_summary.txt` |

---

### Technical Stack

| Component | Tool |
|---|---|
| GIS / vector analysis | GeoPandas, Shapely, Fiona |
| Raster analysis | rasterio, numpy, scipy |
| Hydrological modelling | pysheds |
| Kriging interpolation | pykrige |
| Spatial statistics (Gi*) | esda, libpysal |
| Static cartography | matplotlib, contextily |
| Interactive mapping | Leaflet.js, Leaflet.markercluster |
| Data downloads | requests, USGS NWIS API, NRCS Soil Data Access API |
| Version control | Git / GitHub |

---

### Portfolio Value

This project demonstrates expertise in:
- Hydrogeological risk modelling and raster analysis
- Spatial interpolation (ordinary kriging of groundwater surfaces)
- Cost-distance contaminant transport modelling
- Spatial statistics (Getis-Ord Gi* hot-spot analysis)
- Environmental justice / socioeconomic overlay analysis
- End-to-end reproducible geospatial pipeline design
- Interactive cartographic web application development

The core question — which drinking-water wells are at elevated risk from nearby leaking
USTs, *before* contamination is detected — has genuine regulatory value and is not
routinely modelled by state environmental agencies.
