#!/usr/bin/env python3
"""
01_data_acquisition.py
======================
Download all raw datasets for the Ghost Infrastructure project.
Study area: Harris County, Texas (FIPS 48201)

Real data sources are attempted first; synthetic fallbacks are generated
automatically so the pipeline can run end-to-end without manual intervention.
"""

import io
import sys
import json
import zipfile
import warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

BBOX = STUDY_AREA["bbox_wgs84"]   # (minlon, minlat, maxlon, maxlat)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dl(url: str, dest: Path, desc: str = "", timeout: int = 60) -> bool:
    """Stream-download a file. Returns True on success."""
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"  {desc}", leave=False
        ) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"    [!] Download failed: {e}")
        return False


def _ksat_class(v: float) -> str:
    if v < 0.001:  return "Very Slow"
    if v < 0.01:   return "Slow"
    if v < 0.1:    return "Moderate/Slow"
    if v < 1.0:    return "Moderate"
    if v < 10.0:   return "Moderate/Rapid"
    if v < 100.0:  return "Rapid"
    return "Very Rapid"


# ── 1  TCEQ LUST Sites ────────────────────────────────────────────────────────

def download_tceq_lust() -> bool:
    dest = RAW_DIR / "lust" / "tceq_pst.csv"
    if dest.exists():
        print("  [OK] TCEQ LUST data already present.")
        return True

    print("  Attempting TCEQ PST database download…")
    # The TCEQ provides a downloadable Access/CSV export of their PST DB.
    # URL may change; check https://www.tceq.texas.gov/remediation/pst/pst_reports.html
    zip_dest = RAW_DIR / "lust" / "pstdb.zip"
    url = "https://www.tceq.texas.gov/assets/public/remediation/pst/pstdb.zip"
    if _dl(url, zip_dest, "TCEQ PST DB"):
        try:
            with zipfile.ZipFile(zip_dest) as z:
                z.extractall(RAW_DIR / "lust")
            csvs = sorted((RAW_DIR / "lust").glob("*.csv"))
            if csvs:
                csvs[0].rename(dest)
                print(f"  Downloaded & extracted TCEQ LUST data.")
                return True
        except Exception as e:
            print(f"    [!] Extraction failed: {e}")

    print("  Generating synthetic TCEQ LUST data for Harris County…")
    _synthetic_lust(dest)
    return False


def _synthetic_lust(dest: Path):
    rng = np.random.default_rng(42)
    n = 523
    minlon, minlat, maxlon, maxlat = BBOX

    # Cluster around Houston's commercial corridors
    clusters = [
        (-95.370, 29.760, 80), (-95.550, 29.770, 60), (-95.450, 29.950, 40),
        (-95.250, 29.770, 50), (-95.620, 29.670, 35), (-95.480, 29.650, 30),
        (-95.700, 29.830, 25), (-95.270, 29.600, 25), (-95.400, 29.890, 40),
        (-95.200, 29.850, 35), (-95.500, 29.720, 40), (-95.580, 29.730, 43),
    ]
    lons, lats = [], []
    for clon, clat, cnt in clusters:
        lons.extend(rng.normal(clon, 0.08, cnt))
        lats.extend(rng.normal(clat, 0.06, cnt))
    lons = np.clip(lons, minlon, maxlon)[:n]
    lats = np.clip(lats, minlat, maxlat)[:n]

    statuses   = ["Closed-No Further Action","Open-Active","Open-Inactive",
                  "Closed-Contamination Remains","Open-Confirmed Contamination"]
    status_p   = [0.35, 0.20, 0.15, 0.20, 0.10]
    fac_types  = ["Service Station","Dry Cleaner","Fleet Facility",
                  "Agriculture","Industrial","Commercial","Government"]
    fac_p      = [0.55, 0.10, 0.10, 0.05, 0.08, 0.07, 0.05]
    remed      = ["Unremediated","Partial","Complete","Monitoring"]
    remed_p    = [0.25, 0.30, 0.30, 0.15]
    conts      = ["Petroleum/BTEX","Chlorinated Solvents","Heavy Metals","Mixed","Unknown"]
    conts_p    = [0.60, 0.15, 0.10, 0.10, 0.05]
    years      = rng.integers(1960, 2015, n)

    df = pd.DataFrame({
        "SITE_ID":            [f"TCEQ-{i:05d}" for i in range(1, n+1)],
        "FACILITY_NAME":      [f"Facility {i}" for i in range(1, n+1)],
        "FACILITY_TYPE":      rng.choice(fac_types, n, p=fac_p),
        "ADDRESS":            [f"{rng.integers(100,9999)} Main St" for _ in range(n)],
        "CITY":               "Houston",
        "STATE":              "TX",
        "ZIP":                rng.choice(["77001","77002","77003","77004","77005",
                                          "77006","77007","77008","77009","77010"], n),
        "LONGITUDE":          lons,
        "LATITUDE":           lats,
        "STATUS":             rng.choice(statuses, n, p=status_p),
        "REPORT_YEAR":        years,
        "TANK_AGE_YEARS":     2025 - years,
        "REMEDIATION_STATUS": rng.choice(remed, n, p=remed_p),
        "CONTAMINANTS":       rng.choice(conts, n, p=conts_p),
        "GALLONS_RELEASED":   np.round(rng.lognormal(3, 2, n), 0),
    })
    df.to_csv(dest, index=False)
    print(f"    Generated {n} synthetic LUST sites → {dest.name}")


# ── 2  EPA Brownfields ────────────────────────────────────────────────────────

def download_epa_brownfields() -> bool:
    dest = RAW_DIR / "brownfields" / "epa_brownfields.csv"
    if dest.exists():
        print("  [OK] EPA Brownfields data already present.")
        return True

    print("  Fetching EPA Brownfields via ECHO API…")
    minlon, minlat, maxlon, maxlat = BBOX
    try:
        params = {
            "output": "JSON", "p_co": "TX,Harris",
            "p_ptype": "GBF", "qcolumns": "1,2,3,4,5,6,7,8",
        }
        r = requests.get("https://echo.epa.gov/api/effluent_charts/rest/facility_search",
                         params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if "Results" in data and "Facilities" in data["Results"]:
                df = pd.json_normalize(data["Results"]["Facilities"])
                df.to_csv(dest, index=False)
                print(f"    Downloaded {len(df)} brownfield sites.")
                return True
    except Exception as e:
        print(f"    [!] ECHO API failed: {e}")

    print("  Generating synthetic EPA Brownfields…")
    _synthetic_brownfields(dest)
    return False


def _synthetic_brownfields(dest: Path):
    rng = np.random.default_rng(43)
    n = 85
    minlon, minlat, maxlon, maxlat = BBOX
    lons = np.concatenate([
        rng.normal(-95.25, 0.10, 35),
        rng.normal(-95.48, 0.12, 30),
        rng.normal(-95.65, 0.08, 20),
    ])
    lats = np.concatenate([
        rng.normal(29.75, 0.06, 35),
        rng.normal(29.80, 0.08, 30),
        rng.normal(29.77, 0.06, 20),
    ])
    lons = np.clip(lons, minlon, maxlon)[:n]
    lats = np.clip(lats, minlat, maxlat)[:n]
    pd.DataFrame({
        "SITE_ID":     [f"BF-{i:04d}" for i in range(1, n+1)],
        "SITE_NAME":   [f"Brownfield Site {i}" for i in range(1, n+1)],
        "LONGITUDE":   lons, "LATITUDE": lats,
        "STATUS":      rng.choice(["Assessment","Cleanup","Ready for Reuse","Investigation"], n),
        "CONTAMINANTS":rng.choice(["VOCs","Metals","PCBs","Petroleum","Mixed"], n),
        "AREA_ACRES":  np.round(rng.lognormal(1, 1, n), 2),
    }).to_csv(dest, index=False)
    print(f"    Generated {n} synthetic brownfield sites → {dest.name}")


# ── 3  EPA SDWIS Drinking Water Wells ─────────────────────────────────────────

def download_sdwis_wells() -> bool:
    dest = RAW_DIR / "wells" / "sdwis_wells.csv"
    if dest.exists():
        print("  [OK] SDWIS well data already present.")
        return True

    print("  Fetching EPA SDWIS well data…")
    try:
        url = "https://data.epa.gov/efservice/SDW_PWSID_COORD/PRIMACY_AGENCY_CODE/TX/CSV"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            minlon, minlat, maxlon, maxlat = BBOX
            for lat_c in ["LATITUDE","LAT"]:
                for lon_c in ["LONGITUDE","LON","LONG"]:
                    if lat_c in df.columns and lon_c in df.columns:
                        df = df[df[lon_c].between(minlon, maxlon) &
                                df[lat_c].between(minlat, maxlat)]
                        if len(df) > 0:
                            df.to_csv(dest, index=False)
                            print(f"    Downloaded {len(df)} drinking-water wells.")
                            return True
    except Exception as e:
        print(f"    [!] SDWIS API failed: {e}")

    print("  Generating synthetic SDWIS well data…")
    _synthetic_wells(dest)
    return False


def _synthetic_wells(dest: Path):
    rng = np.random.default_rng(44)
    n = 142
    minlon, minlat, maxlon, maxlat = BBOX
    pd.DataFrame({
        "PWSID":            [f"TX{rng.integers(1000000,9999999)}" for _ in range(n)],
        "PWS_NAME":         [f"Water System {i}" for i in range(1, n+1)],
        "PWS_TYPE":         rng.choice(["Community","Non-Transient Non-Community",
                                        "Transient Non-Community"], n, p=[0.60,0.25,0.15]),
        "LONGITUDE":        rng.uniform(minlon, maxlon, n),
        "LATITUDE":         rng.uniform(minlat, maxlat, n),
        "POPULATION_SERVED":rng.integers(25, 50000, n),
        "ACTIVE":           rng.choice([True, False], n, p=[0.85, 0.15]),
        "WELL_DEPTH_FT":    np.round(rng.uniform(50, 500, n), 1),
        "INSTALL_YEAR":     rng.integers(1940, 2020, n),
    }).to_csv(dest, index=False)
    print(f"    Generated {n} synthetic drinking-water wells → {dest.name}")


# ── 4  USGS Groundwater Levels ────────────────────────────────────────────────

def download_usgs_groundwater() -> bool:
    dest = RAW_DIR / "groundwater" / "usgs_gwlevels.csv"
    if dest.exists():
        print("  [OK] USGS groundwater data already present.")
        return True

    print("  Fetching USGS groundwater levels from NWIS…")
    try:
        r = requests.get(
            "https://waterservices.usgs.gov/nwis/gwlevels/",
            params={"format":"rdb","stateCd":"TX","countycd":"48201",
                    "startDT":"2015-01-01","endDT":"2023-12-31","parameterCd":"72019"},
            timeout=90,
        )
        if r.status_code == 200:
            lines = [l for l in r.text.split("\n") if not l.startswith("#")]
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")
            df = df[~df.iloc[:, 0].astype(str).str.match(r"^\d+s$")]
            if len(df) > 5:
                df.to_csv(dest, index=False)
                print(f"    Downloaded {len(df)} USGS groundwater records.")
                return True
    except Exception as e:
        print(f"    [!] USGS NWIS failed: {e}")

    print("  Generating synthetic USGS groundwater data…")
    _synthetic_groundwater(dest)
    return False


def _synthetic_groundwater(dest: Path):
    rng  = np.random.default_rng(45)
    n_s  = 48
    minlon, minlat, maxlon, maxlat = BBOX
    lons = rng.uniform(minlon, maxlon, n_s)
    lats = rng.uniform(minlat, maxlat, n_s)
    # depth deeper in NW, shallower toward coast
    depth = (15 + 40 * (lats - minlat) / (maxlat - minlat)
                - 20 * (lons - minlon) / (maxlon - minlon)
                + rng.normal(0, 5, n_s))
    depth = np.clip(depth, 5, 100)
    rows = []
    for i in range(n_s):
        for d in pd.date_range("2015-01-01", "2023-12-31",
                               periods=int(rng.integers(3, 20))):
            rows.append({
                "SITE_NO":        f"0848{i:04d}",
                "STATION_NM":     f"GW Monitor {i+1}",
                "DEC_LAT_VA":     lats[i],
                "DEC_LONG_VA":    lons[i],
                "WELL_DEPTH_VA":  rng.uniform(50, 300),
                "LEV_VA":         depth[i] + rng.normal(0, 2),  # depth-to-water (ft)
                "MEASUREMENT_DT": d.strftime("%Y-%m-%d"),
            })
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"    Generated {len(rows)} synthetic groundwater measurements → {dest.name}")


# ── 5  NLCD 2021 Land Cover ───────────────────────────────────────────────────

def download_nlcd() -> bool:
    dest = RAW_DIR / "nlcd" / "nlcd_2021.tif"
    if dest.exists():
        print("  [OK] NLCD raster already present.")
        return True
    print("  NOTE: NLCD requires a manual download from https://www.mrlc.gov/")
    print("  Generating synthetic NLCD land-cover raster for Harris County…")
    _synthetic_nlcd(dest)
    return False


def _synthetic_nlcd(dest: Path):
    import rasterio
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    from scipy.ndimage import zoom

    minlon, minlat, maxlon, maxlat = BBOX
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    xmin, ymin = tr.transform(minlon, minlat)
    xmax, ymax = tr.transform(maxlon, maxlat)
    res = 30
    w, h = int((xmax - xmin) / res), int((ymax - ymin) / res)
    tf = from_bounds(xmin, ymin, xmax, ymax, w, h)

    rng = np.random.default_rng(46)
    # NLCD classes: Urban Houston is ~71% developed
    classes = [11, 21, 22, 23, 24, 41, 52, 71, 90, 95]
    probs   = [0.03, 0.08, 0.20, 0.28, 0.15, 0.05, 0.05, 0.06, 0.06, 0.04]
    bs = 50
    blocks = rng.choice(classes, (h // bs + 1, w // bs + 1), p=probs)
    data = zoom(blocks.astype(float), (bs, bs), order=0)[:h, :w].astype(np.uint8)

    with rasterio.open(dest, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="uint8", crs="EPSG:32614",
                       transform=tf, compress="lzw") as ds:
        ds.write(data, 1)
    print(f"    Generated synthetic NLCD raster ({w}×{h} px) → {dest.name}")


# ── 6  SSURGO Soil Hydraulic Conductivity ─────────────────────────────────────

def download_ssurgo() -> bool:
    dest = RAW_DIR / "soil" / "ssurgo_ksat.csv"
    if dest.exists():
        print("  [OK] SSURGO data already present.")
        return True

    print("  Fetching SSURGO via Soil Data Access API…")
    query = """
        SELECT mu.mukey, mu.muname, c.compname, c.comppct_r,
               ch.hzdept_r, ch.hzdepb_r, ch.ksat_r,
               ch.sandtotal_r, ch.claytotal_r, ch.dbthirdbar_r
        FROM   sacatalog sc
        JOIN   legend    l  ON sc.areasymbol = l.areasymbol
        JOIN   mapunit   mu ON l.lkey = mu.lkey
        JOIN   component c  ON mu.mukey = c.mukey
        JOIN   chorizon  ch ON c.cokey  = ch.cokey
        WHERE  sc.areasymbol = 'TX201'
          AND  ch.ksat_r IS NOT NULL
          AND  c.majcompflag = 'Yes'
        ORDER BY mu.mukey, c.cokey, ch.hzdept_r
    """
    try:
        r = requests.post(
            "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest",
            data={"format": "JSON+COLUMNNAME", "query": query}, timeout=90,
        )
        if r.status_code == 200:
            tbl = r.json().get("Table", [])
            if len(tbl) > 1:
                df = pd.DataFrame(tbl[1:], columns=tbl[0])
                df.to_csv(dest, index=False)
                print(f"    Downloaded {len(df)} SSURGO horizon records.")
                return True
    except Exception as e:
        print(f"    [!] Soil Data Access failed: {e}")

    print("  Generating synthetic SSURGO soil data…")
    _synthetic_ssurgo(dest)
    return False


def _synthetic_ssurgo(dest: Path):
    rng = np.random.default_rng(47)
    # Representative Harris County soil series + Ksat ranges (µm/s)
    soils = [
        ("Lake Charles Clay",     0.001, 0.010),  # Vertisol – NW Harris
        ("Aldine Loam",           0.050, 0.150),
        ("Bernard Clay Loam",     0.020, 0.080),
        ("Katy Fine Sandy Loam",  0.500, 2.000),
        ("Galveston Fine Sand",   5.000, 20.00),
        ("Wharton Clay Loam",     0.020, 0.050),
        ("Crowley Loam",          0.040, 0.120),
        ("Beaumont Clay",         0.001, 0.005),
    ]
    n = 180
    rows = []
    for i in range(n):
        nm, kmin, kmax = soils[rng.integers(0, len(soils))]
        ksat = rng.uniform(kmin, kmax)
        rows.append({
            "mukey":        f"MU{i:05d}",
            "muname":       nm,
            "ksat_r":       ksat,
            "ksat_class":   _ksat_class(ksat),
            "sandtotal_r":  rng.uniform(5, 85),
            "claytotal_r":  rng.uniform(5, 65),
            "dbthirdbar_r": rng.uniform(1.1, 1.7),
        })
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"    Generated {n} synthetic soil map units → {dest.name}")


# ── 7  3DEP Digital Elevation Model ──────────────────────────────────────────

def download_dem() -> bool:
    dest = RAW_DIR / "dem" / "harris_county_dem.tif"
    if dest.exists():
        print("  [OK] DEM already present.")
        return True

    print("  Attempting 3DEP DEM download via py3dep…")
    try:
        import py3dep
        minlon, minlat, maxlon, maxlat = BBOX
        dem = py3dep.get_map("DEM", (minlon, minlat, maxlon, maxlat),
                             resolution=30, crs="EPSG:4326")
        import rioxarray  # noqa
        dem.rio.to_raster(str(dest))
        print(f"    Downloaded 3DEP DEM → {dest.name}")
        return True
    except Exception as e:
        print(f"    [!] py3dep failed: {e}")

    print("  Generating synthetic DEM for Harris County…")
    _synthetic_dem(dest)
    return False


def _synthetic_dem(dest: Path):
    import rasterio
    from rasterio.transform import from_bounds
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter

    minlon, minlat, maxlon, maxlat = BBOX
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    xmin, ymin = tr.transform(minlon, minlat)
    xmax, ymax = tr.transform(maxlon, maxlat)
    res = 30
    w, h = int((xmax - xmin) / res), int((ymax - ymin) / res)
    tf = from_bounds(xmin, ymin, xmax, ymax, w, h)

    rng = np.random.default_rng(48)
    xx = np.linspace(0, 1, w)
    yy = np.linspace(0, 1, h)
    X, Y = np.meshgrid(xx, yy)

    # Harris County: mostly flat coastal plain, gentle NW→SE slope, 0–50 m
    base = 45 * (0.55 * (1 - X) + 0.45 * (1 - Y))

    # Carve bayou channels (Buffalo Bayou, Brays, White Oak, Greens, Sims)
    def carve(grid, r0, c0, r1, c1, depth=4, half=8):
        for c, r in zip(np.linspace(c0, c1, 120, dtype=int),
                        np.linspace(r0, r1, 120, dtype=int)):
            grid[max(0,r-half):r+half, max(0,c-half):c+half] -= (
                depth * rng.uniform(0.8, 1.2))
        return grid

    base = carve(base, int(h*.90), int(w*.20), int(h*.50), int(w*.80), 5, 10)
    base = carve(base, int(h*.80), int(w*.10), int(h*.60), int(w*.70), 4,  8)
    base = carve(base, int(h*.20), int(w*.30), int(h*.50), int(w*.60), 3,  7)
    base = carve(base, int(h*.10), int(w*.50), int(h*.50), int(w*.90), 3,  7)

    noise = gaussian_filter(rng.normal(0, 1.5, (h, w)), sigma=10)
    elev  = np.clip(base + noise, 0, 60).astype(np.float32)

    with rasterio.open(dest, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs="EPSG:32614",
                       transform=tf, compress="lzw", nodata=-9999) as ds:
        ds.write(elev, 1)
    print(f"    Generated synthetic DEM ({w}×{h} px, 30 m) → {dest.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Ghost Infrastructure – Data Acquisition")
    print(f"  Study area : {STUDY_AREA['name']}")
    print("=" * 62)

    steps = [
        ("TCEQ Leaking UST Sites",          download_tceq_lust),
        ("EPA Brownfields",                  download_epa_brownfields),
        ("EPA SDWIS Drinking-Water Wells",   download_sdwis_wells),
        ("USGS Groundwater Levels",          download_usgs_groundwater),
        ("NLCD 2021 Land Cover",             download_nlcd),
        ("SSURGO Soil Hydraulic Conductivity", download_ssurgo),
        ("USGS 3DEP Digital Elevation Model",  download_dem),
    ]
    results = {}
    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {label}…")
        results[label] = fn()

    print("\n" + "=" * 62)
    print("  Data Acquisition Summary")
    print("=" * 62)
    for label, real in results.items():
        tag = "Real data" if real else "Synthetic data"
        print(f"  {tag:<15} {label}")
    print("\n→ Run  scripts/02_preprocessing.py  next.")
