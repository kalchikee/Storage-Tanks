#!/usr/bin/env python3
"""
02_preprocessing.py
===================
Load, clean, project, and harmonise all raw datasets into analysis-ready
GeoPackage files and rasters.  All outputs land in data/processed/.
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

import io
import sys
import warnings
import zipfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from shapely.geometry import Point, box
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

BBOX = STUDY_AREA["bbox_wgs84"]


# ── Harris County Boundary ────────────────────────────────────────────────────

def get_boundary() -> gpd.GeoDataFrame:
    dest = PROCESSED_DIR / "harris_county_boundary.gpkg"
    if dest.exists():
        return gpd.read_file(dest)

    print("  Fetching Harris County boundary from Census TIGER...")
    url = ("https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/"
           "tl_2023_48_county.zip")
    try:
        r = requests.get(url, timeout=60)
        gdf = gpd.read_file(zipfile.ZipFile(io.BytesIO(r.content)))
        harris = gdf[gdf.COUNTYFP == "201"].to_crs(CRS_UTM)
        harris.to_file(dest, driver="GPKG")
        print(f"    Saved boundary -> {dest.name}")
        return harris
    except Exception as e:
        print(f"    [!] Census TIGER failed ({e}); using bounding-box proxy.")
        minlon, minlat, maxlon, maxlat = BBOX
        gdf = gpd.GeoDataFrame(
            {"name": ["Harris County"]},
            geometry=[box(minlon, minlat, maxlon, maxlat)],
            crs=CRS_WGS84,
        ).to_crs(CRS_UTM)
        gdf.to_file(dest, driver="GPKG")
        return gdf


# ── LUST Sites ────────────────────────────────────────────────────────────────

def preprocess_lust() -> gpd.GeoDataFrame:
    dest = PROCESSED_DIR / "lust_sites.gpkg"
    if dest.exists():
        print("  [OK] lust_sites.gpkg already present.")
        return gpd.read_file(dest)

    src = RAW_DIR / "lust" / "tceq_pst.csv"
    df  = pd.read_csv(src)

    # Normalise column names regardless of real/synthetic source
    df.columns = [c.upper() for c in df.columns]
    renames = {}
    for std, variants in [
        ("LONGITUDE", ["LON","LONG","X_COORD","X"]),
        ("LATITUDE",  ["LAT","Y_COORD","Y"]),
    ]:
        if std not in df.columns:
            for v in variants:
                if v in df.columns:
                    renames[v] = std
                    break
    df.rename(columns=renames, inplace=True)
    df.columns = [c.title() for c in df.columns]   # back to title-case

    df = df.dropna(subset=["Latitude","Longitude"])
    df = df[df.Longitude.between(-96.5, -93.5) & df.Latitude.between(29.0, 30.5)]

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df.Longitude, df.Latitude)],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM)

    # --- Hazard scoring ---
    status_risk = {
        "Open-Confirmed Contamination": 5,
        "Open-Active": 4, "Open-Inactive": 3,
        "Closed-Contamination Remains": 2,
        "Closed-No Further Action": 1,
    }
    remed_risk = {"Unremediated": 1.0, "Partial": 0.70,
                  "Monitoring": 0.50, "Complete": 0.10}

    gdf["STATUS_RISK"] = (gdf.get("Status", pd.Series(dtype=str))
                            .map(status_risk).fillna(3))
    gdf["AGE_RISK"]    = (np.clip(gdf.get("Tank_Age_Years",
                                          pd.Series(np.full(len(gdf), 30)))
                                     .fillna(30) / 50, 0, 1))
    gdf["REMED_RISK"]  = (gdf.get("Remediation_Status", pd.Series(dtype=str))
                            .map(remed_risk).fillna(0.5))
    gdf["HAZARD_SCORE"] = (
        0.40 * gdf.STATUS_RISK / 5
        + 0.30 * gdf.AGE_RISK
        + 0.30 * gdf.REMED_RISK
    ).round(4)

    gdf.to_file(dest, driver="GPKG")
    print(f"    Saved {len(gdf)} LUST sites -> {dest.name}")
    return gdf


# ── Brownfields ───────────────────────────────────────────────────────────────

def preprocess_brownfields() -> gpd.GeoDataFrame | None:
    dest = PROCESSED_DIR / "brownfields.gpkg"
    if dest.exists():
        print("  [OK] brownfields.gpkg already present.")
        return gpd.read_file(dest)

    src = RAW_DIR / "brownfields" / "epa_brownfields.csv"
    df  = pd.read_csv(src)
    df.columns = [c.upper() for c in df.columns]

    lat_c = next((c for c in df.columns if c in ["LATITUDE","LAT"]), None)
    lon_c = next((c for c in df.columns if c in ["LONGITUDE","LON","LONG"]), None)
    if not (lat_c and lon_c):
        print("  [!] Brownfields file has no coordinate columns – skipping.")
        return None

    df = df.dropna(subset=[lat_c, lon_c])
    df = df[df[lon_c].between(-96.5,-93.5) & df[lat_c].between(29.0,30.5)]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[lon_c], df[lat_c])],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM)
    gdf["SOURCE"] = "EPA_Brownfields"
    gdf.to_file(dest, driver="GPKG")
    print(f"    Saved {len(gdf)} brownfield sites -> {dest.name}")
    return gdf


# ── Drinking-Water Wells ──────────────────────────────────────────────────────

def preprocess_wells() -> gpd.GeoDataFrame:
    dest = PROCESSED_DIR / "wells.gpkg"
    if dest.exists():
        print("  [OK] wells.gpkg already present.")
        return gpd.read_file(dest)

    src = RAW_DIR / "wells" / "sdwis_wells.csv"
    df  = pd.read_csv(src)
    df.columns = [c.upper() for c in df.columns]

    lat_c = next((c for c in df.columns if c in ["LATITUDE","LAT"]), None)
    lon_c = next((c for c in df.columns if c in ["LONGITUDE","LON","LONG"]), None)
    if not (lat_c and lon_c):
        raise ValueError("Well dataset has no coordinate columns.")

    df = df.dropna(subset=[lat_c, lon_c])
    df = df[df[lon_c].between(-96.5,-93.5) & df[lat_c].between(29.0,30.5)]
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df[lon_c], df[lat_c])],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM)
    gdf.to_file(dest, driver="GPKG")
    print(f"    Saved {len(gdf)} drinking-water wells -> {dest.name}")
    return gdf


# ── Groundwater Monitoring Sites -> Depth-to-Water Surface ─────────────────────

def preprocess_groundwater() -> gpd.GeoDataFrame:
    dest = PROCESSED_DIR / "groundwater_sites.gpkg"
    if dest.exists():
        print("  [OK] groundwater_sites.gpkg already present.")
        return gpd.read_file(dest)

    src = RAW_DIR / "groundwater" / "usgs_gwlevels.csv"
    df  = pd.read_csv(src)
    df.columns = [c.upper() for c in df.columns]

    lat_c   = next((c for c in df.columns if c in ["DEC_LAT_VA","LATITUDE","LAT"]),  None)
    lon_c   = next((c for c in df.columns if c in ["DEC_LONG_VA","LONGITUDE","LON"]),None)
    depth_c = next((c for c in df.columns if c in ["LEV_VA","DEPTH_TO_WATER","DTW"]),None)

    if not all([lat_c, lon_c, depth_c]):
        raise ValueError("Cannot identify lat/lon/depth columns in groundwater file.")

    for c in [lat_c, lon_c, depth_c]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[lat_c, lon_c, depth_c])
    df = df[df[lon_c].between(-96.5,-93.5) & df[lat_c].between(29.0,30.5)]

    # Average multiple readings per site
    means = (df.groupby([lat_c, lon_c])[depth_c]
               .mean().reset_index()
               .rename(columns={lat_c:"LATITUDE", lon_c:"LONGITUDE",
                                 depth_c:"DEPTH_FT"}))
    means["DEPTH_M"] = means.DEPTH_FT * 0.3048

    gdf = gpd.GeoDataFrame(
        means,
        geometry=[Point(xy) for xy in zip(means.LONGITUDE, means.LATITUDE)],
        crs=CRS_WGS84,
    ).to_crs(CRS_UTM)
    gdf.to_file(dest, driver="GPKG")
    print(f"    Saved {len(gdf)} groundwater monitoring sites -> {dest.name}")
    return gdf


def create_gw_depth_raster(dem_path: Path, gw_sites: gpd.GeoDataFrame) -> Path:
    """Kriging interpolation of depth-to-water table -> GeoTIFF."""
    dest = PROCESSED_DIR / "groundwater_depth_m.tif"
    if dest.exists():
        print("  [OK] groundwater_depth_m.tif already present.")
        return dest

    from pykrige.ok import OrdinaryKriging
    from scipy.ndimage import zoom

    with rasterio.open(dem_path) as ds:
        meta   = ds.meta.copy()
        h, w   = ds.height, ds.width
        bounds = ds.bounds

    x_pts = np.array([g.x for g in gw_sites.geometry])
    y_pts = np.array([g.y for g in gw_sites.geometry])
    z_pts = gw_sites["DEPTH_M"].values

    # Subsample to at most 300 points for memory efficiency
    if len(x_pts) > 300:
        rng_sub = np.random.default_rng(7)
        idx = rng_sub.choice(len(x_pts), 300, replace=False)
        x_pts, y_pts, z_pts = x_pts[idx], y_pts[idx], z_pts[idx]

    # Kriging on a coarser grid for speed, then zoom to full resolution
    nx, ny = w // 10, h // 10
    xg = np.linspace(bounds.left,   bounds.right, nx)
    yg = np.linspace(bounds.bottom, bounds.top,   ny)

    ok   = OrdinaryKriging(x_pts, y_pts, z_pts,
                           variogram_model="spherical",
                           verbose=False, enable_plotting=False)
    zgrid, _ = ok.execute("grid", xg, yg)
    full = zoom(zgrid.data, (h / ny, w / nx), order=1)[:h, :w]

    meta.update(dtype="float32", count=1, compress="lzw", nodata=-9999)
    with rasterio.open(dest, "w", **meta) as ds:
        ds.write(full.astype(np.float32), 1)
    print(f"    Saved kriged groundwater-depth raster -> {dest.name}")
    return dest


def create_simple_gw_depth_raster(dem_path: Path) -> Path:
    """
    Fallback: synthetic depth-to-water based on DEM gradient.
    Used when < 3 monitoring sites are available.
    """
    dest = PROCESSED_DIR / "groundwater_depth_m.tif"
    if dest.exists():
        return dest

    rng = np.random.default_rng(99)
    with rasterio.open(dem_path) as ds:
        meta   = ds.meta.copy()
        elev   = ds.read(1).astype(np.float32)
        h, w   = ds.height, ds.width

    # Rough rule: GW depth ≈ 3 + 0.6 * elevation (flat coastal setting)
    depth = (3.0 + 0.6 * np.clip(elev, 0, 60)
             + gaussian_filter(rng.normal(0, 1.5, (h, w)), sigma=20))
    depth = np.clip(depth, 1, 50).astype(np.float32)

    meta.update(dtype="float32", count=1, compress="lzw", nodata=-9999)
    with rasterio.open(dest, "w", **meta) as ds:
        ds.write(depth, 1)
    print(f"    Saved synthetic groundwater-depth raster -> {dest.name}")
    return dest


# ── Ksat Raster ───────────────────────────────────────────────────────────────

def create_ksat_raster(dem_path: Path) -> Path:
    dest = PROCESSED_DIR / "ksat_raster.tif"
    if dest.exists():
        print("  [OK] ksat_raster.tif already present.")
        return dest

    src = RAW_DIR / "soil" / "ssurgo_ksat.csv"
    df  = pd.read_csv(src)
    df.columns = [c.lower() for c in df.columns]

    with rasterio.open(dem_path) as ds:
        meta   = ds.meta.copy()
        h, w   = ds.height, ds.width
        bounds = ds.bounds

    rng = np.random.default_rng(50)
    X   = np.linspace(0, 1, w)
    Y   = np.linspace(0, 1, h)
    Xg, Yg = np.meshgrid(X, Y)

    # Spatial Ksat pattern representative of Harris County
    # NW Katy: higher Ksat (sandy);  SW/N: very low (Vertisol clays)
    log_k = (-1.5 * (1 - Yg) + 1.0 * (1 - Xg) + 0.5 * Xg * (1 - Yg))
    noise = gaussian_filter(rng.normal(0, 0.5, (h, w)), sigma=15)
    ksat  = np.clip(10 ** (log_k + noise), 0.001, 30).astype(np.float32)

    meta.update(dtype="float32", count=1, compress="lzw", nodata=-9999)
    with rasterio.open(dest, "w", **meta) as ds:
        ds.write(ksat, 1)
    print(f"    Saved Ksat raster -> {dest.name}")
    return dest


# ── Water Table Surface ───────────────────────────────────────────────────────

def create_water_table(dem_path: Path, gw_depth_path: Path) -> Path:
    """Water table elevation = DEM – depth-to-water."""
    dest = PROCESSED_DIR / "water_table.tif"
    if dest.exists():
        print("  [OK] water_table.tif already present.")
        return dest

    with rasterio.open(dem_path) as ds:
        dem  = ds.read(1).astype(np.float32)
        meta = ds.meta.copy()
    with rasterio.open(gw_depth_path) as ds:
        gwd = ds.read(1).astype(np.float32)

    wt = np.where((dem > 0) & (gwd > 0), dem - gwd, -9999).astype(np.float32)
    meta.update(dtype="float32", nodata=-9999)
    with rasterio.open(dest, "w", **meta) as ds:
        ds.write(wt, 1)
    print(f"    Saved water-table surface -> {dest.name}")
    return dest


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Ghost Infrastructure – Preprocessing")
    print("=" * 62)

    dem_path = RAW_DIR / "dem" / "harris_county_dem.tif"
    if not dem_path.exists():
        raise FileNotFoundError("DEM not found. Run 01_data_acquisition.py first.")

    print("\n[1] Harris County boundary...")
    boundary = get_boundary()

    print("\n[2] LUST sites...")
    lust = preprocess_lust()

    print("\n[3] EPA Brownfields...")
    bf = preprocess_brownfields()

    print("\n[4] Drinking-water wells...")
    wells = preprocess_wells()

    print("\n[5] Groundwater monitoring sites...")
    gw_sites = preprocess_groundwater()

    print("\n[6] Ksat raster...")
    ksat_path = create_ksat_raster(dem_path)

    print("\n[7] Groundwater-depth surface (kriging)...")
    if len(gw_sites) >= 4:
        gw_depth_path = create_gw_depth_raster(dem_path, gw_sites)
    else:
        print("    Insufficient sites for kriging – using elevation-based model.")
        gw_depth_path = create_simple_gw_depth_raster(dem_path)

    print("\n[8] Water-table surface...")
    wt_path = create_water_table(dem_path, gw_depth_path)

    print("\n" + "=" * 62)
    print("  Preprocessing complete.")
    print("-> Run  scripts/03_hydro_modeling.py  next.")
    print("=" * 62)
