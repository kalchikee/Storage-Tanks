#!/usr/bin/env python3
"""
06_web_map.py
=============
Export processed datasets to GeoJSON and inject them into the standalone
Leaflet.js web map (web/index.html).

Outputs
-------
web/data/lust_sites.geojson          – all LUST sites with hazard scores
web/data/wells.geojson               – wells with vulnerability + EJ flags
web/data/risk_zones.geojson          – High / Moderate / Low risk polygons
web/data/remediation_priority.json   – top 50 ranked LUST sites
web/index.html                       – regenerated with embedded metadata
"""

import os as _os, importlib.util as _iu
def _fix_proj():
    _spec = _iu.find_spec('rasterio')
    if _spec:
        import pathlib as _pl
        _proj = _pl.Path(_spec.origin).parent / 'proj_data'
        if _proj.exists():
            _os.environ.setdefault('GDAL_DATA', str(_pl.Path(_spec.origin).parent / 'gdal_data'))
            _os.environ.setdefault('PROJ_DATA', str(_proj))
            _os.environ.setdefault('PROJ_LIB',  str(_proj))
            _os.environ.setdefault('PROJ_NETWORK', 'OFF')
_fix_proj(); del _fix_proj

import sys
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import mapping

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

WEB_DATA = WEB_DIR / "data"


# ── GeoJSON Export Helpers ────────────────────────────────────────────────────

def _clean_props(props: dict) -> dict:
    """Convert numpy types and NaN to JSON-serialisable values."""
    out = {}
    for k, v in props.items():
        if isinstance(v, (np.integer,)):  out[k] = int(v)
        elif isinstance(v, (np.floating,)): out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, float):        out[k] = None if np.isnan(v) else round(v, 5)
        elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
            out[k] = None
        else:
            out[k] = v
    return out


def export_geojson(gdf: gpd.GeoDataFrame,
                   dest: Path,
                   prop_cols: list[str],
                   target_crs: str = "EPSG:4326"):
    """Reproject to WGS-84 and write a minimal GeoJSON."""
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_UTM)   # assume UTM 14N if CRS missing
    gdf2 = gdf.to_crs(target_crs).copy()
    features = []
    for _, row in gdf2.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = _clean_props({c: row[c] for c in prop_cols if c in row.index})
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": props,
        })
    fc = {"type": "FeatureCollection", "features": features}
    dest.write_text(json.dumps(fc, separators=(",", ":")))
    print(f"    Exported {len(features)} features -> {dest.name}")


# ── 1  LUST Sites ─────────────────────────────────────────────────────────────

def export_lust(lust: gpd.GeoDataFrame):
    cols = [c for c in ["Site_Id","Facility_Name","Facility_Type","Status",
                        "Remediation_Status","Contaminants","Tank_Age_Years",
                        "Gallons_Released","HAZARD_SCORE","STATUS_RISK",
                        "AGE_RISK","REMED_RISK"]
            if c in lust.columns]
    # Add lat/lon for popup convenience and ensure SITE_ID is string
    lust2 = lust.to_crs("EPSG:4326").copy()
    lust2["lat"] = lust2.geometry.y.round(6)
    lust2["lon"] = lust2.geometry.x.round(6)
    if "Site_Id" in lust2.columns:
        lust2["Site_Id"] = lust2["Site_Id"].astype(str)
    cols += ["lat","lon"]
    export_geojson(lust2, WEB_DATA / "lust_sites.geojson", cols)


# ── 2  Wells ──────────────────────────────────────────────────────────────────

def export_wells(wells: gpd.GeoDataFrame):
    cols = [c for c in ["PWSID","PWS_Name","PWS_Type","Population_Served",
                        "Well_Depth_Ft","N_LUST_3KM","CONTAMINATION_RISK",
                        "VULNERABILITY_SCORE","VULN_CLASS","EJ_FLAG",
                        "MED_INCOME","PCT_POVERTY","GI_STAR","HOTSPOT"]
            if c in wells.columns]
    wells2 = wells.to_crs("EPSG:4326").copy()
    wells2["lat"] = wells2.geometry.y.round(6)
    wells2["lon"] = wells2.geometry.x.round(6)
    cols += ["lat","lon"]
    export_geojson(wells, WEB_DATA / "wells.geojson", cols)


# ── 3  Risk Zones ─────────────────────────────────────────────────────────────

def export_risk_zones(zones: gpd.GeoDataFrame):
    if zones is None or len(zones) == 0:
        WEB_DATA.joinpath("risk_zones.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}')
        print("    No risk zones to export.")
        return
    export_geojson(zones, WEB_DATA / "risk_zones.geojson",
                   [c for c in ["RISK_CLASS","RISK_CODE"] if c in zones.columns])


# ── 4  Remediation Priority JSON ─────────────────────────────────────────────

def export_priority(priority_df: pd.DataFrame):
    from pyproj import Transformer
    to_wgs84 = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)

    records = priority_df.head(50).reset_index().to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
        # Convert UTM LONGITUDE/LATITUDE to WGS84 for map fly-to
        lon_utm = r.get("LONGITUDE")
        lat_utm = r.get("LATITUDE")
        if lon_utm is not None and lat_utm is not None:
            try:
                wgs_lon, wgs_lat = to_wgs84.transform(float(lon_utm), float(lat_utm))
                r["LNG"] = round(wgs_lon, 6)
                r["LAT"] = round(wgs_lat, 6)
            except Exception:
                pass
        # Ensure SITE_ID is a string for consistent JS lookup
        if "SITE_ID" in r:
            r["SITE_ID"] = str(r["SITE_ID"])
    (WEB_DATA / "remediation_priority.json").write_text(
        json.dumps(records, separators=(",", ":")))
    print(f"    Exported top {len(records)} priority sites -> remediation_priority.json")


# ── 5  Metadata sidebar JSON ──────────────────────────────────────────────────

def export_metadata(lust: gpd.GeoDataFrame, wells: gpd.GeoDataFrame):
    vuln_counts = {}
    if "VULN_CLASS" in wells.columns:
        vc = wells.VULN_CLASS.value_counts().to_dict()
        vuln_counts = {str(k): int(v) for k, v in vc.items()}

    meta = {
        "study_area":    STUDY_AREA["name"],
        "lust_count":    int(len(lust)),
        "well_count":    int(len(wells)),
        "vuln_counts":   vuln_counts,
        "ej_flagged":    int(wells.get("EJ_FLAG", pd.Series(dtype=bool)).sum()),
        "generated":     pd.Timestamp.now().strftime("%Y-%m-%d"),
    }
    (WEB_DATA / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"    Metadata -> metadata.json")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Ghost Infrastructure – Web Map Data Export")
    print("=" * 62)

    lust_path  = PROCESSED_DIR / "lust_sites.gpkg"
    wells_path = PROCESSED_DIR / "wells_ej.gpkg"
    if not wells_path.exists():
        wells_path = PROCESSED_DIR / "wells_scored.gpkg"
    if not wells_path.exists():
        wells_path = PROCESSED_DIR / "wells.gpkg"

    for p in [lust_path, wells_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run earlier pipeline steps first.")

    lust  = gpd.read_file(lust_path)
    wells = gpd.read_file(wells_path)

    zones = None
    zones_path = PROCESSED_DIR / "risk_zones.gpkg"
    if zones_path.exists():
        zones = gpd.read_file(zones_path)

    priority = None
    priority_path = REPORTS_DIR / "remediation_priority.csv"
    if priority_path.exists():
        priority = pd.read_csv(priority_path)

    print("\n[1] Exporting LUST sites...")
    export_lust(lust)

    print("[2] Exporting wells...")
    export_wells(wells)

    print("[3] Exporting risk zones...")
    export_risk_zones(zones)

    if priority is not None:
        print("[4] Exporting remediation priority list...")
        export_priority(priority)

    print("[5] Writing metadata...")
    export_metadata(lust, wells)

    print("\n" + "=" * 62)
    print(f"  GeoJSON data written to {WEB_DATA}")
    print("  Open web/index.html in a browser to view the map.")
    print("=" * 62)
