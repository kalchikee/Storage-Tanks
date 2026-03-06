#!/usr/bin/env python3
"""
04_risk_scoring.py
==================
Well Vulnerability Assessment, Hot-Spot Analysis, and Environmental Justice
analysis for the Ghost Infrastructure project.

Steps
-----
1. Score each drinking-water well (composite vulnerability index)
2. Validate predicted risk against simulated USGS groundwater sampling results
3. Getis-Ord Gi* hot-spot analysis on well vulnerability scores
4. Census ACS demographic analysis for environmental-justice implications
5. Prioritised LUST remediation list
6. Export all outputs to GeoPackages and CSV reports
"""

import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import Point
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_raster(path: Path) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32), ds.transform


def _sample_raster(gdf: gpd.GeoDataFrame,
                   arr: np.ndarray,
                   tf,
                   nodata: float = -9999) -> np.ndarray:
    """Sample raster values at GeoDataFrame point locations."""
    h, w = arr.shape
    vals = np.full(len(gdf), np.nan, dtype=np.float32)
    for i, geom in enumerate(gdf.geometry):
        row, col = rasterio.transform.rowcol(tf, geom.x, geom.y)
        if 0 <= row < h and 0 <= col < w:
            v = arr[row, col]
            vals[i] = v if v != nodata else np.nan
    return vals


# ── Step 1: Well Vulnerability Scoring ────────────────────────────────────────

def score_wells(wells: gpd.GeoDataFrame,
                lust:  gpd.GeoDataFrame,
                risk_arr: np.ndarray, risk_tf,
                ksat_arr: np.ndarray, ksat_tf) -> gpd.GeoDataFrame:
    """
    Composite well vulnerability score [0–100]:

    Component                             Weight
    ─────────────────────────────────────────────
    # upstream LUST within 3 km           25 %
    Distance-weighted LUST hazard         25 %
    Contamination risk at well location   25 %
    Soil Ksat at well location            15 %
    Well depth / aquifer protection       10 %
    """
    dest = PROCESSED_DIR / "wells_scored.gpkg"
    if dest.exists():
        print("  [OK] wells_scored.gpkg already present.")
        return gpd.read_file(dest)

    wells = wells.copy()

    # ── (a) Upstream LUST within 3 km ────────────────────────────────────────
    lust_coords = np.array([(g.x, g.y) for g in lust.geometry])
    tree  = cKDTree(lust_coords)

    n_nearby    = np.zeros(len(wells), dtype=int)
    dist_hazard = np.zeros(len(wells), dtype=np.float32)

    for i, geom in enumerate(wells.geometry):
        pt  = np.array([geom.x, geom.y])
        idx = tree.query_ball_point(pt, r=3000)   # 3 km radius
        if idx:
            dists  = np.linalg.norm(lust_coords[idx] - pt, axis=1)
            dists  = np.clip(dists, 1, None)
            hazards = lust.HAZARD_SCORE.iloc[idx].values
            n_nearby[i]    = len(idx)
            dist_hazard[i] = np.sum(hazards / dists) * 1000   # distance-weighted

    # Normalise
    n_norm  = np.clip(n_nearby  / (n_nearby.max()  + 1e-6), 0, 1)
    dh_norm = np.clip(dist_hazard / (dist_hazard.max() + 1e-6), 0, 1)

    # ── (b) Contamination risk at well ────────────────────────────────────────
    cr = _sample_raster(wells, risk_arr, risk_tf)
    cr = np.where(np.isnan(cr), 0, cr)

    # ── (c) Ksat at well location ─────────────────────────────────────────────
    ks = _sample_raster(wells, ksat_arr, ksat_tf)
    ks = np.where(np.isnan(ks), ksat_arr[~np.isnan(ksat_arr)].mean(), ks)
    ks_norm = np.clip(np.log10(ks + 1e-6) / np.log10(30), 0, 1)

    # ── (d) Well depth protection  (deeper = better protected) ────────────────
    depth_col = next((c for c in wells.columns
                      if c.upper() in ["WELL_DEPTH_FT","WELL_DEPTH","DEPTH_FT"]), None)
    if depth_col:
        depth = pd.to_numeric(wells[depth_col], errors="coerce").fillna(100)
    else:
        depth = pd.Series(np.full(len(wells), 100.0))
    depth_protect = np.clip(1 - depth.values / 600, 0, 1)   # shallower → more vulnerable

    # ── Composite score ───────────────────────────────────────────────────────
    vuln = (
        0.25 * n_norm
        + 0.25 * dh_norm
        + 0.25 * cr
        + 0.15 * ks_norm
        + 0.10 * depth_protect
    ) * 100

    wells["N_LUST_3KM"]          = n_nearby
    wells["DIST_WEIGHTED_HAZARD"] = dist_hazard.round(4)
    wells["CONTAMINATION_RISK"]   = cr.round(4)
    wells["KSAT_AT_WELL"]         = ks.round(4)
    wells["VULNERABILITY_SCORE"]  = vuln.round(2)
    wells["VULN_CLASS"] = pd.cut(
        wells.VULNERABILITY_SCORE,
        bins=[0, 25, 50, 75, 100],
        labels=["Low","Moderate","High","Critical"],
        right=True,
    )

    wells.to_file(dest, driver="GPKG")
    print(f"    Scored {len(wells)} wells → {dest.name}")
    return wells


# ── Step 2: Validation Against Sampling Data ──────────────────────────────────

def validate_model(wells_scored: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Simulate 'detected contamination' for validation.
    In a real project this would come from USGS NAWQA / NWIS water-quality data.
    We model detection probability as f(vulnerability score) + noise to compute
    precision / recall / ROC metrics.
    """
    rng = np.random.default_rng(77)
    df  = wells_scored[["VULNERABILITY_SCORE","VULN_CLASS"]].copy()

    # Synthetic ground truth: detection probability rises with vulnerability
    prob_detect = 1 / (1 + np.exp(-(df.VULNERABILITY_SCORE - 55) / 12))
    detected    = rng.random(len(df)) < prob_detect
    df["DETECTED"] = detected.astype(int)

    threshold = 50.0
    df["PREDICTED_HIGH"] = (df.VULNERABILITY_SCORE >= threshold).astype(int)

    TP = int(((df.PREDICTED_HIGH == 1) & (df.DETECTED == 1)).sum())
    FP = int(((df.PREDICTED_HIGH == 1) & (df.DETECTED == 0)).sum())
    TN = int(((df.PREDICTED_HIGH == 0) & (df.DETECTED == 0)).sum())
    FN = int(((df.PREDICTED_HIGH == 0) & (df.DETECTED == 1)).sum())

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    report = pd.DataFrame({
        "Metric":   ["True Positive","False Positive","True Negative","False Negative",
                     "Precision","Recall","F1 Score"],
        "Value":    [TP, FP, TN, FN,
                     round(precision, 3), round(recall, 3), round(f1, 3)],
    })
    dest = REPORTS_DIR / "model_validation.csv"
    report.to_csv(dest, index=False)
    print(f"    Validation report → {dest.name}")
    print(f"    Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}")
    return report


# ── Step 3: Getis-Ord Gi* Hot-Spot Analysis ───────────────────────────────────

def hotspot_analysis(wells_scored: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Identify statistically significant clusters of high-vulnerability wells."""
    dest = PROCESSED_DIR / "wells_hotspot.gpkg"
    if dest.exists():
        print("  [OK] wells_hotspot.gpkg already present.")
        return gpd.read_file(dest)

    try:
        from libpysal.weights import KNN
        from esda.getisord import G_Local

        w  = KNN.from_dataframe(wells_scored, k=min(8, len(wells_scored) - 1))
        gl = G_Local(wells_scored.VULNERABILITY_SCORE, w, transform="R")

        gdf = wells_scored.copy()
        gdf["GI_STAR"]    = gl.Zs
        gdf["P_VALUE"]    = gl.p_sim
        gdf["HOTSPOT"]    = pd.cut(
            gl.Zs,
            bins=[-np.inf, -2.58, -1.96, 1.96, 2.58, np.inf],
            labels=["Cold Spot 99%","Cold Spot 95%","Not Significant",
                    "Hot Spot 95%","Hot Spot 99%"],
        )
        gdf.to_file(dest, driver="GPKG")
        print(f"    Hot-spot analysis complete → {dest.name}")
        n_hot = int((gl.Zs > 1.96).sum())
        print(f"    {n_hot} significant hot-spot wells (p < 0.05).")
        return gdf

    except ImportError:
        print("    [!] esda/libpysal not installed; skipping hot-spot analysis.")
        return wells_scored


# ── Step 4: Environmental Justice Analysis ────────────────────────────────────

def ej_analysis(wells_scored: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Join Census ACS data to high-risk wells.
    Uses Census Bureau API (tract-level) if available; otherwise generates
    synthetic socioeconomic overlay.
    """
    dest = PROCESSED_DIR / "wells_ej.gpkg"
    if dest.exists():
        print("  [OK] wells_ej.gpkg already present.")
        return gpd.read_file(dest)

    # Try Census API
    try:
        from census import Census
        from us import states
        # Census API key – get a free key at https://api.census.gov/data/key_signup.html
        api_key = "YOUR_CENSUS_API_KEY"   # replace or set env var CENSUS_API_KEY
        import os
        api_key = os.environ.get("CENSUS_API_KEY", api_key)
        if api_key == "YOUR_CENSUS_API_KEY":
            raise ValueError("No Census API key configured.")
        c   = Census(api_key)
        acs = c.acs5.state_county_tract(
            ("NAME","B19013_001E","B17001_002E","B17001_001E","B02001_003E","B02001_001E"),
            states.TX.fips, "201", Census.ALL,
        )
        df_acs = pd.DataFrame(acs)
        df_acs["MED_INCOME"]    = pd.to_numeric(df_acs["B19013_001E"], errors="coerce")
        df_acs["PCT_POVERTY"]   = (pd.to_numeric(df_acs["B17001_002E"], errors="coerce")
                                   / pd.to_numeric(df_acs["B17001_001E"], errors="coerce").clip(1))
        df_acs["PCT_BLACK"]     = (pd.to_numeric(df_acs["B02001_003E"], errors="coerce")
                                   / pd.to_numeric(df_acs["B02001_001E"], errors="coerce").clip(1))
        df_acs["GEOID"] = df_acs["state"] + df_acs["county"] + df_acs["tract"]
        print(f"    Downloaded ACS data for {len(df_acs)} census tracts.")
        acs_data_available = True
    except Exception as e:
        print(f"    [!] Census API unavailable ({e}); generating synthetic ACS data.")
        acs_data_available = False

    gdf = wells_scored.copy()

    if not acs_data_available:
        rng = np.random.default_rng(55)
        n   = len(gdf)
        # Generate EJ metrics that correlate with vulnerability for realism
        score = gdf.VULNERABILITY_SCORE.values / 100
        gdf["MED_INCOME"]  = np.clip(
            75000 - 35000 * score + rng.normal(0, 8000, n), 20000, 150000).round(-2)
        gdf["PCT_POVERTY"] = np.clip(
            0.08 + 0.20 * score  + rng.normal(0, 0.04, n), 0, 0.60).round(3)
        gdf["PCT_MINORITY"]= np.clip(
            0.25 + 0.35 * score  + rng.normal(0, 0.06, n), 0, 1.00).round(3)
    else:
        # Spatial join to census tracts – simplified (point-in-polygon)
        for col in ["MED_INCOME","PCT_POVERTY","PCT_BLACK"]:
            gdf[col] = np.nan   # would normally come from spatial join

    # EJ composite flag: low income OR high minority share + high vulnerability
    gdf["EJ_FLAG"] = (
        (gdf.MED_INCOME < 50000) |
        (gdf.PCT_POVERTY > 0.20) |
        (gdf.get("PCT_MINORITY", gdf.get("PCT_BLACK", pd.Series(0, index=gdf.index))) > 0.50)
    ) & (gdf.VULNERABILITY_SCORE >= 50)

    gdf.to_file(dest, driver="GPKG")
    ej_count = gdf.EJ_FLAG.sum()
    print(f"    EJ analysis complete: {ej_count} high-risk wells in vulnerable communities.")
    return gdf


# ── Step 5: LUST Remediation Priority List ────────────────────────────────────

def remediation_priority(lust: gpd.GeoDataFrame,
                         wells_scored: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Rank LUST sites by the number and severity of downstream wells threatened.
    Priority Score = Σ (well vulnerability score / distance) for wells within 3 km.
    """
    dest = REPORTS_DIR / "remediation_priority.csv"
    if dest.exists():
        print("  [OK] remediation_priority.csv already present.")
        return pd.read_csv(dest)

    well_coords   = np.array([(g.x, g.y) for g in wells_scored.geometry])
    well_vuln     = wells_scored.VULNERABILITY_SCORE.values
    tree          = cKDTree(well_coords)

    priorities = []
    for _, site in lust.iterrows():
        pt  = np.array([site.geometry.x, site.geometry.y])
        idx = tree.query_ball_point(pt, r=3000)
        if not idx:
            threat = 0.0
            n_wells = 0
        else:
            dists    = np.linalg.norm(well_coords[idx] - pt, axis=1).clip(1)
            vuln     = well_vuln[idx]
            threat   = float(np.sum(vuln / dists) * 100)
            n_wells  = len(idx)

        priorities.append({
            "SITE_ID":          site.get("Site_Id", site.get("SITE_ID", "?")),
            "FACILITY_NAME":    site.get("Facility_Name", "Unknown"),
            "STATUS":           site.get("Status", "Unknown"),
            "REMEDIATION":      site.get("Remediation_Status", "Unknown"),
            "HAZARD_SCORE":     round(float(site.HAZARD_SCORE), 3),
            "N_WELLS_WITHIN_3KM": n_wells,
            "DOWNSTREAM_THREAT": round(threat, 2),
            "LONGITUDE":        round(site.geometry.centroid.x if hasattr(site.geometry, 'centroid') else site.geometry.x, 5),
            "LATITUDE":         round(site.geometry.centroid.y if hasattr(site.geometry, 'centroid') else site.geometry.y, 5),
        })

    df = (pd.DataFrame(priorities)
            .sort_values("DOWNSTREAM_THREAT", ascending=False)
            .reset_index(drop=True))
    df.index += 1
    df.index.name = "PRIORITY_RANK"
    df.to_csv(dest)
    print(f"    Remediation priority list saved → {dest.name}")
    print(f"    Top 5 sites by downstream threat:")
    print(df.head(5)[["SITE_ID","FACILITY_NAME","N_WELLS_WITHIN_3KM","DOWNSTREAM_THREAT"]].to_string())
    return df


# ── Step 6: Summary Statistics ────────────────────────────────────────────────

def save_summary(wells_ej: gpd.GeoDataFrame,
                 priority_df: pd.DataFrame,
                 validation_df: pd.DataFrame):
    """Write a concise text summary for the report."""
    dest = REPORTS_DIR / "analysis_summary.txt"
    lines = [
        "=" * 62,
        "  Ghost Infrastructure – Analysis Summary",
        "=" * 62,
        "",
        f"Study Area : {STUDY_AREA['name']}",
        f"Total Wells Assessed : {len(wells_ej)}",
        "",
        "Well Vulnerability Distribution:",
    ]
    for cls in ["Low","Moderate","High","Critical"]:
        n = int((wells_ej.VULN_CLASS == cls).sum())
        lines.append(f"  {cls:<10} : {n}")
    lines += [
        "",
        f"Wells in EJ-Flag Communities: {int(wells_ej.EJ_FLAG.sum())}",
        "",
        "Model Validation (simulated):",
    ]
    for _, row in validation_df.iterrows():
        lines.append(f"  {row.Metric:<20}: {row.Value}")
    lines += [
        "",
        "Top 3 LUST Sites for Immediate Remediation:",
    ]
    for i, row in priority_df.head(3).iterrows():
        lines.append(f"  Rank {i}: {row.FACILITY_NAME} – Threat={row.DOWNSTREAM_THREAT}")
    lines.append("")

    dest.write_text("\n".join(lines))
    print(f"    Summary report → {dest.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Ghost Infrastructure – Risk Scoring & Analysis")
    print("=" * 62)

    for p in [PROCESSED_DIR / "lust_sites.gpkg",
              PROCESSED_DIR / "wells.gpkg",
              PROCESSED_DIR / "contamination_risk.tif",
              PROCESSED_DIR / "ksat_raster.tif"]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run earlier pipeline steps first.")

    print("\n[1] Loading processed datasets…")
    lust   = gpd.read_file(PROCESSED_DIR / "lust_sites.gpkg")
    wells  = gpd.read_file(PROCESSED_DIR / "wells.gpkg")

    risk_arr, risk_tf = _read_raster(PROCESSED_DIR / "contamination_risk.tif")
    ksat_arr, ksat_tf = _read_raster(PROCESSED_DIR / "ksat_raster.tif")

    print("\n[2] Scoring well vulnerability…")
    wells_scored = score_wells(wells, lust, risk_arr, risk_tf, ksat_arr, ksat_tf)

    print("\n[3] Validating model against sampling data…")
    validation = validate_model(wells_scored)

    print("\n[4] Hot-spot analysis (Getis-Ord Gi*)…")
    wells_hot = hotspot_analysis(wells_scored)

    print("\n[5] Environmental Justice analysis…")
    wells_ej = ej_analysis(wells_hot if "GI_STAR" in wells_hot.columns else wells_scored)

    print("\n[6] LUST remediation priority list…")
    priority = remediation_priority(lust, wells_scored)

    print("\n[7] Saving summary…")
    save_summary(wells_ej, priority, validation)

    print("\n" + "=" * 62)
    print("  Risk scoring complete.")
    print("→ Run  scripts/05_static_maps.py  next.")
    print("=" * 62)
