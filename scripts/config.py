"""Shared settings for the Yosemite Valley flyover."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
OUT = ROOT / "out"

CRS = "EPSG:26911"             # NAD83 / UTM zone 11N, metres (the lidar's own grid)

# The terrain: Yosemite Valley from west of Tunnel View to past Clouds Rest.
LEFT, TOP = 261_000.0, 4_187_100.0
DEM_RES = 3.0                  # metres; the viewer's mesh is coarser than this anyway
DEM_W, DEM_H = 7680, 4200      # 23.0 km x 12.6 km
RIGHT, BOTTOM = LEFT + DEM_W * DEM_RES, TOP - DEM_H * DEM_RES

# Aerial imagery (NAIP): the whole area at 4 m, and the valley itself sharper, in two halves.
CONTEXT_RES = 4.0
VALLEY = (263_000.0, 4_183_600.0, 278_200.0, 4_176_400.0)   # left, top, right, bottom
VALLEY_RES = 1.5

DEM_RAW = DATA / "yosemite_dem_3m.tif"
DEM = DATA / "yosemite_dem_3m_smooth.tif"
NAIP_CONTEXT = DATA / "naip_context_4m.png"
NAIP_WEST = DATA / "naip_valley_west.png"
NAIP_EAST = DATA / "naip_valley_east.png"


def graded(path: Path) -> Path:
    return path.with_name(path.stem + "_graded.png")


def valley_halves():
    """(path, left, top, right, bottom) for the two sharp imagery tiles."""
    left, top, right, bottom = VALLEY
    mid = round((left + right) / 2 / VALLEY_RES) * VALLEY_RES
    return [(NAIP_WEST, left, top, mid, bottom), (NAIP_EAST, mid, top, right, bottom)]


# Evening light: June 20, 7:15 pm PDT. The sun is low in the west-north-west and
# lights Half Dome's north-west face while the valley floor falls into shade.
SUN_UTC = (2026, 6, 21, 2, 15, 0)
SUN_LATLON = (37.735, -119.59)

VIDEO = OUT / "yosemite_valley_flyover.mp4"
