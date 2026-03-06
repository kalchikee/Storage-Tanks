"""
config.py  –  Ghost Infrastructure Project Configuration
Central settings for all scripts.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR    = ROOT_DIR / "output"
MAPS_DIR      = OUTPUT_DIR / "maps"
REPORTS_DIR   = OUTPUT_DIR / "reports"
GEOJSON_DIR   = OUTPUT_DIR / "geojson"
WEB_DIR       = ROOT_DIR / "web"

# ── Study Area: Harris County, Texas ─────────────────────────────────────────
STUDY_AREA = {
    "name":       "Harris County, Texas",
    "fips":       "48201",
    "state_fips": "48",
    "county_fips":"201",
    "state":      "TX",
    "county":     "Harris",
    # (minlon, minlat, maxlon, maxlat) in WGS-84
    "bbox_wgs84": (-95.90, 29.40, -94.90, 30.20),
}

# ── Coordinate Reference Systems ──────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32614"   # NAD83 / UTM Zone 14N  (meters)

# ── Raster Settings ───────────────────────────────────────────────────────────
RASTER_RESOLUTION = 30   # metres

# ── Risk Zone Distances (metres along flow path) ──────────────────────────────
RISK_ZONES = {
    "high":     500,
    "moderate": 1500,
    "low":      3000,
}

# ── Ensure all directories exist ──────────────────────────────────────────────
for _d in [
    RAW_DIR / "lust", RAW_DIR / "wells",  RAW_DIR / "soil",
    RAW_DIR / "dem",  RAW_DIR / "groundwater", RAW_DIR / "nlcd",
    RAW_DIR / "brownfields",
    PROCESSED_DIR, MAPS_DIR, REPORTS_DIR, GEOJSON_DIR,
    WEB_DIR / "css", WEB_DIR / "js", WEB_DIR / "data",
]:
    _d.mkdir(parents=True, exist_ok=True)
