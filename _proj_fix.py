"""
_proj_fix.py
============
Sets GDAL/PROJ environment variables to rasterio's bundled data directories
BEFORE any CRS operations, preventing conflicts with PostgreSQL/PostGIS PROJ.

Import this module at the very top of every script, before rasterio.
"""
import os, pathlib, importlib.util

def _apply():
    # Find rasterio's package directory without fully importing it
    spec = importlib.util.find_spec("rasterio")
    if spec is None:
        return
    base = pathlib.Path(spec.origin).parent  # .../site-packages/rasterio/
    gdal = base / "gdal_data"
    proj = base / "proj_data"
    if proj.exists():
        os.environ["GDAL_DATA"]     = str(gdal)
        os.environ["PROJ_DATA"]     = str(proj)
        os.environ["PROJ_LIB"]      = str(proj)
        os.environ["PROJ_NETWORK"]  = "OFF"

_apply()
