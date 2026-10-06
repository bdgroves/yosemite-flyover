"""Step 2: prepare the render inputs from fetch_data.py's downloads.

  yosemite_dem_3m_smooth.tif  a light blur (under a pixel), so single noisy cells
                              don't become spikes in the viewer's mesh; the granite
                              walls stay sharp
  naip_*_graded.png           a little more saturation and contrast, offsetting the
                              renderer's sun/sky tint and tone mapping. On the cliffs the
                              aerial photo is useless (seen from straight above, a 900 m
                              wall is a few pixels wide, and at midday it's in shadow), so
                              faces steeper than ~50 degrees get a granite tone instead and
                              the renderer's evening sun lights them

    pixi run prep
"""
from __future__ import annotations

import sys

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage

import config

DEM_SIGMA_PX = 0.8
SATURATION = 1.22
GRANITE = np.array([0.80, 0.77, 0.71], dtype=np.float32)     # Yosemite granite, sunlit
CLIFF_DEG = (42.0, 62.0)                                     # photo -> granite blend over these slopes


def smooth_dem() -> None:
    with rasterio.open(config.DEM_RAW) as src:
        profile, h = src.profile, src.read(1).astype(np.float64)
    with rasterio.open(config.DEM, "w", **profile) as dst:
        dst.write(ndimage.gaussian_filter(h, DEM_SIGMA_PX).astype(np.float32), 1)
    print(f"wrote {config.DEM.name}")


def slope_deg() -> np.ndarray:
    with rasterio.open(config.DEM_RAW) as src:
        z = ndimage.gaussian_filter(src.read(1).astype(np.float32), 1.0)
    gy, gx = np.gradient(z, config.DEM_RES)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def image_box(path):
    """(left, top, res) of an imagery file."""
    if path == config.NAIP_CONTEXT:
        return config.LEFT, config.TOP, config.CONTEXT_RES
    for p, l, t, _, _ in config.valley_halves():
        if p == path:
            return l, t, config.VALLEY_RES
    raise KeyError(path)


def cliffs(rgb: np.ndarray, path, slope: np.ndarray) -> np.ndarray:
    left, top, res = image_box(path)
    h, w = rgb.shape[:2]
    rows = (top - (np.arange(h) + 0.5) * res - config.TOP) / -config.DEM_RES - 0.5
    cols = (left + (np.arange(w) + 0.5) * res - config.LEFT) / config.DEM_RES - 0.5
    # slope at every image pixel (bilinear from the DEM grid), a row at a time to save memory
    k = np.empty((h, w), dtype=np.float32)
    lo, hi = CLIFF_DEG
    for r0 in range(0, h, 512):
        rr = rows[r0:r0 + 512]
        RR, CC = np.meshgrid(rr, cols, indexing="ij")
        sl = ndimage.map_coordinates(slope, [RR, CC], order=1, mode="nearest")
        t = np.clip((sl - lo) / (hi - lo), 0, 1)
        k[r0:r0 + 512] = t * t * (3 - 2 * t)
    # keep a little of the photo's light and dark so the granite isn't flat paint
    lum = rgb.mean(axis=-1, keepdims=True)
    local = ndimage.uniform_filter(lum[..., 0], size=31)[..., None] + 1e-3
    tex = np.clip(0.85 + 0.30 * (lum / local - 1.0), 0.7, 1.15)
    rock = GRANITE * tex
    return rgb * (1 - k[..., None]) + rock * k[..., None]


def grade(path, slope: np.ndarray) -> None:
    Image.MAX_IMAGE_PIXELS = None
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    rgb = cliffs(rgb, path, slope)
    lum = (rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    rgb = np.clip(lum + (rgb - lum) * SATURATION, 0.0, 1.0)
    rgb += 0.12 * (rgb - 0.5) * (1.0 - np.abs(2.0 * rgb - 1.0))     # gentle S-curve
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(config.graded(path))
    print(f"wrote {config.graded(path).name}")


def main() -> int:
    missing = [p.name for p in (config.DEM_RAW, config.NAIP_CONTEXT, config.NAIP_WEST, config.NAIP_EAST)
               if not p.exists()]
    if missing:
        print("Missing", ", ".join(missing), "- run: pixi run data")
        return 1
    smooth_dem()
    slope = slope_deg()
    for p in (config.NAIP_CONTEXT, config.NAIP_WEST, config.NAIP_EAST):
        grade(p, slope)
    return 0


if __name__ == "__main__":
    sys.exit(main())
