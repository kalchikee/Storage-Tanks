#!/usr/bin/env python3
"""
05_static_maps.py
=================
Publication-quality cartographic maps for the Ghost Infrastructure project.

Map 1 – LUST Site Overview
   All LUST sites coloured by status with Harris County boundary.

Map 2 – Contamination Risk Surface
   Continuous risk raster with LUST sites and risk zone polygons overlaid.

Map 3 – Well Vulnerability Assessment
   Drinking-water wells sized/coloured by vulnerability score; EJ flag marked.

Map 4 – Environmental Justice Overview
   Median income choropleth with high-risk/EJ-flagged wells highlighted.

Map 5 – Remediation Priority Map
   Top 50 priority LUST sites ranked and annotated.

All maps saved as high-resolution PNG to output/maps/.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import rasterio
from rasterio.plot import show as rshow
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})
RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#edf8fb","#b2e2e2","#66c2a4","#2ca25f",
             "#fec44f","#fe9929","#ec7014","#cc4c02","#8c2d04"])

STATUS_COLORS = {
    "Open-Confirmed Contamination": "#d73027",
    "Open-Active":                  "#f46d43",
    "Open-Inactive":                "#fdae61",
    "Closed-Contamination Remains": "#abd9e9",
    "Closed-No Further Action":     "#74add1",
}
VULN_COLORS = {
    "Low":      "#1a9641",
    "Moderate": "#fdae61",
    "High":     "#f46d43",
    "Critical": "#d73027",
}


def _add_basemap(ax, gdf_bounds):
    """Try contextily basemap; silently skip if unavailable."""
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs=CRS_UTM,
                        source=ctx.providers.CartoDB.Positron,
                        alpha=0.5, zoom="auto")
    except Exception:
        pass   # basemap is optional


def _north_arrow(ax):
    ax.annotate("N", xy=(0.97, 0.13), xytext=(0.97, 0.06),
                xycoords="axes fraction",
                fontsize=12, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))


def _scale_bar(ax, length_km=10):
    """Simple scale bar in data coordinates (UTM metres)."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + (xmax - xmin) * 0.07
    y0 = ymin + (ymax - ymin) * 0.04
    ax.plot([x0, x0 + length_km * 1000], [y0, y0], "k-", lw=3, solid_capstyle="butt")
    ax.text(x0 + length_km * 500, y0 + (ymax - ymin) * 0.015,
            f"{length_km} km", ha="center", fontsize=8)


# ── Map 1: LUST Site Overview ─────────────────────────────────────────────────

def map_lust_overview(lust: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(12, 10))
    boundary.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#666", linewidth=1.2)

    for status, color in STATUS_COLORS.items():
        sub = lust[lust.get("Status", lust.get("STATUS", pd.Series(dtype=str))) == status]
        if len(sub):
            sub.plot(ax=ax, color=color, markersize=4, alpha=0.75, label=status)

    _add_basemap(ax, lust.total_bounds)
    _north_arrow(ax)
    _scale_bar(ax)

    handles = [mpatches.Patch(color=c, label=s)
               for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, title="Site Status", loc="lower right",
              fontsize=8, title_fontsize=9, framealpha=0.9)
    ax.set_title("Harris County — Leaking Underground Storage Tank (LUST) Sites\n"
                 f"n = {len(lust):,}  |  Source: TCEQ PST Database")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(5, 5))

    dest = MAPS_DIR / "01_lust_overview.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {dest.name}")


# ── Map 2: Contamination Risk Surface ────────────────────────────────────────

def map_risk_surface(lust: gpd.GeoDataFrame, risk_zones: gpd.GeoDataFrame,
                     boundary: gpd.GeoDataFrame, risk_tif: Path):
    fig, ax = plt.subplots(figsize=(12, 10))
    boundary.plot(ax=ax, facecolor="none", edgecolor="#444", linewidth=1.5, zorder=5)

    with rasterio.open(risk_tif) as ds:
        arr = ds.read(1)
        arr = np.ma.masked_where(arr <= 0, arr)
        extent = [ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top]

    im = ax.imshow(arr, extent=extent, origin="upper", cmap=RISK_CMAP,
                   vmin=0, vmax=1, alpha=0.80, aspect="auto")

    # Risk zone outlines
    zone_colors = {"High": "#8c2d04", "Moderate": "#fe9929", "Low": "#edf8fb"}
    for _, row in risk_zones.iterrows():
        cls = row.get("RISK_CLASS", "Low")
        gpd.GeoDataFrame([row], geometry="geometry", crs=risk_zones.crs).plot(
            ax=ax, facecolor="none", edgecolor=zone_colors.get(cls, "gray"),
            linewidth=0.8, alpha=0.6)

    lust.plot(ax=ax, color="#111", markersize=2, alpha=0.5, zorder=6, label="LUST site")

    plt.colorbar(im, ax=ax, shrink=0.5, label="Contamination Risk (0–1)")
    _north_arrow(ax)
    _scale_bar(ax)

    zone_handles = [mpatches.Patch(edgecolor=c, facecolor="none",
                                   label=f"{z} Risk Zone", linewidth=1.5)
                    for z, c in zone_colors.items()]
    ax.legend(handles=[Line2D([0],[0], marker="o", color="#111",
                               linestyle="none", markersize=4, label="LUST site")]
                       + zone_handles,
              loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title("Contaminant Transport Risk Surface\n"
                 "Harris County, TX — Modelled from LUST Sites, Soil Ksat, and Groundwater Flow")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    dest = MAPS_DIR / "02_contamination_risk.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {dest.name}")


# ── Map 3: Well Vulnerability Assessment ─────────────────────────────────────

def map_well_vulnerability(wells: gpd.GeoDataFrame, lust: gpd.GeoDataFrame,
                           boundary: gpd.GeoDataFrame):
    fig, ax = plt.subplots(figsize=(12, 10))
    boundary.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#888", linewidth=1)
    _add_basemap(ax, boundary.total_bounds)

    lust.plot(ax=ax, color="#666", markersize=2, alpha=0.3, zorder=3)

    for cls, color in VULN_COLORS.items():
        sub = wells[wells.VULN_CLASS == cls]
        if not len(sub): continue
        size = {"Low": 20, "Moderate": 40, "High": 70, "Critical": 110}[cls]
        sub.plot(ax=ax, color=color, markersize=size, alpha=0.85,
                 zorder=5, label=f"{cls} ({len(sub)})")

    # Mark EJ-flagged wells
    if "EJ_FLAG" in wells.columns:
        ej = wells[wells.EJ_FLAG.fillna(False)]
        if len(ej):
            ej.plot(ax=ax, facecolor="none", edgecolor="#6a0dad", linewidth=1.5,
                    markersize=14, marker="^", zorder=6, label=f"EJ Community ({len(ej)})")

    _north_arrow(ax)
    _scale_bar(ax)

    vuln_handles = [mpatches.Patch(color=c, label=f"{cl}")
                    for cl, c in VULN_COLORS.items()]
    extra = []
    if "EJ_FLAG" in wells.columns:
        extra = [Line2D([0],[0], marker="^", color="#6a0dad", linestyle="none",
                        markersize=8, label="EJ Community")]
    ax.legend(handles=vuln_handles + extra, title="Vulnerability Class",
              loc="lower right", fontsize=8, title_fontsize=9, framealpha=0.9)

    ax.set_title("Drinking-Water Well Vulnerability Assessment\n"
                 "Composite Score: LUST Proximity + Soil Ksat + Contamination Risk + Well Depth")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    dest = MAPS_DIR / "03_well_vulnerability.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {dest.name}")


# ── Map 4: EJ + Income Overlay ────────────────────────────────────────────────

def map_ej(wells: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame):
    if "MED_INCOME" not in wells.columns:
        print("    Skipping EJ map – no income data.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    boundary.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#aaa", linewidth=1)

    # Plot wells coloured by median income
    income = wells.MED_INCOME.fillna(wells.MED_INCOME.median())
    norm   = Normalize(vmin=income.quantile(0.05), vmax=income.quantile(0.95))
    cmap   = plt.cm.RdYlGn
    colors = [cmap(norm(v)) for v in income]
    wells.plot(ax=ax, color=colors, markersize=30, alpha=0.85, zorder=4)

    # EJ flag outline
    if "EJ_FLAG" in wells.columns:
        ej = wells[wells.EJ_FLAG.fillna(False)]
        if len(ej):
            ej.plot(ax=ax, facecolor="none", edgecolor="purple",
                    linewidth=2, markersize=18, zorder=5)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Median Household Income ($)")

    _north_arrow(ax)
    _scale_bar(ax)
    ax.set_title("Environmental Justice Overlay\n"
                 "Wells by Median Household Income — Purple Outline = EJ Community + High Risk")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    dest = MAPS_DIR / "04_environmental_justice.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {dest.name}")


# ── Map 5: Remediation Priority ───────────────────────────────────────────────

def map_remediation_priority(lust: gpd.GeoDataFrame,
                             priority_df: pd.DataFrame,
                             boundary: gpd.GeoDataFrame):
    top = priority_df.head(50)
    site_ids = top.reset_index().SITE_ID.values
    top_lust = lust[lust.get("Site_Id", lust.get("SITE_ID", pd.Series(dtype=str)))
                       .isin(site_ids)].copy()
    if len(top_lust) == 0:
        top_lust = lust.head(50).copy()

    top_lust = top_lust.merge(
        top.reset_index()[["SITE_ID","PRIORITY_RANK","DOWNSTREAM_THREAT"]],
        left_on=top_lust.columns[top_lust.columns.str.upper() == "SITE_ID"][0]
                 if any(top_lust.columns.str.upper() == "SITE_ID") else top_lust.columns[0],
        right_on="SITE_ID", how="left",
    )

    fig, ax = plt.subplots(figsize=(13, 10))
    boundary.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#888", linewidth=1)
    lust.plot(ax=ax, color="#ccc", markersize=2, alpha=0.4, zorder=2)

    # Top 50 coloured by threat score
    threat = top_lust.get("DOWNSTREAM_THREAT", pd.Series(np.ones(len(top_lust))))
    norm = Normalize(vmin=0, vmax=threat.max() or 1)
    cmap = plt.cm.OrRd
    cs   = [cmap(norm(v)) for v in threat.fillna(0)]
    sc   = ax.scatter(
        [g.x for g in top_lust.geometry],
        [g.y for g in top_lust.geometry],
        c=cs, s=80, zorder=5, edgecolors="#222", linewidths=0.5,
    )

    # Annotate top 10
    for i, row in top_lust.head(10).iterrows():
        rank = row.get("PRIORITY_RANK", i+1)
        ax.annotate(f"#{int(rank)}", (row.geometry.x, row.geometry.y),
                    fontsize=7, fontweight="bold", color="#222",
                    xytext=(4, 4), textcoords="offset points")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Downstream Well Threat Score")

    _north_arrow(ax)
    _scale_bar(ax)
    ax.set_title("LUST Site Remediation Priority Ranking\n"
                 "Top 50 Sites by Downstream Drinking-Water Well Threat")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    dest = MAPS_DIR / "05_remediation_priority.png"
    fig.tight_layout()
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {dest.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Ghost Infrastructure – Static Map Production")
    print("=" * 62)

    risk_tif  = PROCESSED_DIR / "contamination_risk.tif"
    for p in [PROCESSED_DIR / "lust_sites.gpkg",
              PROCESSED_DIR / "wells_ej.gpkg",
              risk_tif]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run earlier pipeline steps first.")

    print("\n  Loading datasets…")
    boundary   = gpd.read_file(PROCESSED_DIR / "harris_county_boundary.gpkg")
    lust       = gpd.read_file(PROCESSED_DIR / "lust_sites.gpkg")
    wells      = gpd.read_file(PROCESSED_DIR / "wells_ej.gpkg")
    risk_zones = gpd.read_file(PROCESSED_DIR / "risk_zones.gpkg") \
                 if (PROCESSED_DIR / "risk_zones.gpkg").exists() \
                 else gpd.GeoDataFrame(columns=["geometry","RISK_CLASS"], crs=CRS_UTM)
    priority   = pd.read_csv(REPORTS_DIR / "remediation_priority.csv")

    print("\n[1] LUST overview map…")
    map_lust_overview(lust, boundary)

    print("[2] Contamination risk surface map…")
    map_risk_surface(lust, risk_zones, boundary, risk_tif)

    print("[3] Well vulnerability map…")
    map_well_vulnerability(wells, lust, boundary)

    print("[4] Environmental justice map…")
    map_ej(wells, boundary)

    print("[5] Remediation priority map…")
    map_remediation_priority(lust, priority, boundary)

    print("\n" + "=" * 62)
    print(f"  All maps saved to {MAPS_DIR}")
    print("→ Run  scripts/06_web_map.py  next.")
    print("=" * 62)
