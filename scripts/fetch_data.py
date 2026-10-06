"""Step 1: download the terrain and imagery for the Yosemite Valley flyover into data/.

  yosemite_dem_3m.tif    USGS 1 m lidar (Yosemite NP 2019) averaged to 3 m, with
                         USGS 3DEP 1/3" (10 m) filling anything the lidar misses
  naip_context_4m.png    NAIP aerial imagery over the whole area at 4 m
  naip_valley_*.png      NAIP over the valley at 1.5 m, in two halves

The lidar tiles (a few hundred MB each) are kept in data/cache/, and files that
already exist are skipped, so an interrupted run can simply be restarted.

    pixi run data
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy import ndimage

import config

TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
DEP_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n38w120/USGS_13_n38w120.tif"
NAIP_URL = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
PREFERRED = ["CA_YosemiteNP_2019", "CA_SouthernSierra_2020"]   # lidar projects, best first
FEATHER_PX = 20                                                # 60 m blend from lidar to 3DEP
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) yosemite-flyover (brooksgroves.com)"}


def fetch(url: str, timeout: int = 300) -> bytes:
    err = None
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return r.read()
        except Exception as e:                       # noqa: BLE001 - network hiccups, retry
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url.split('?')[0]}: {err}")


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300) as r, \
                    open(tmp, "wb") as f:
                total, got = int(r.headers.get("Content-Length") or 0), 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    print(f"    {got / 1e6:6.0f} / {total / 1e6:.0f} MB", end="\r")
            print()
            tmp.replace(dest)
            return dest
        except Exception as e:                       # noqa: BLE001
            print(f"    attempt {k + 1}: {e}")
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"couldn't download {url}")


def grid_profile(res: float, w: int, h: int) -> dict:
    return {"driver": "GTiff", "width": w, "height": h, "count": 1, "dtype": "float32", "crs": config.CRS,
            "transform": from_origin(config.LEFT, config.TOP, res, res), "nodata": np.nan,
            "compress": "deflate", "tiled": True}


def lidar_tiles() -> list[tuple[str, str]]:
    """(project, download URL) for every 1 m lidar tile touching the area, preferred projects first."""
    w, s, e, n = transform_bounds(config.CRS, "EPSG:4326", config.LEFT, config.BOTTOM, config.RIGHT, config.TOP)
    q = {"datasets": "Digital Elevation Model (DEM) 1 meter", "bbox": f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}",
         "prodFormats": "GeoTIFF", "max": 200, "outputFormat": "JSON"}
    items = json.loads(fetch(f"{TNM_API}?{urllib.parse.urlencode(q)}"))["items"]
    out = []
    for it in items:
        url = it.get("downloadURL") or ""
        proj = next((p for p in PREFERRED if p in url), None)
        if proj and url.endswith(".tif"):
            out.append((proj, url))
    out.sort(key=lambda pu: PREFERRED.index(pu[0]))
    return out


def build_dem() -> None:
    if config.DEM_RAW.exists():
        print(f"Terrain already present: {config.DEM_RAW.name}")
        return
    prof = grid_profile(config.DEM_RES, config.DEM_W, config.DEM_H)
    dst_tf = prof["transform"]
    lidar = np.full((config.DEM_H, config.DEM_W), np.nan, dtype=np.float32)
    tiles = lidar_tiles()
    print(f"Lidar: {len(tiles)} tiles touch the area")
    for k, (proj, url) in enumerate(tiles):
        name = url.rsplit("/", 1)[-1]
        print(f"  [{k + 1}/{len(tiles)}] {name}")
        path = download(url, config.CACHE / name)
        part = np.full_like(lidar, np.nan)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), part, dst_transform=dst_tf, dst_crs=config.CRS,
                      src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.average)
        fill = np.isnan(lidar) & np.isfinite(part)
        lidar[fill] = part[fill]                    # an earlier (preferred) project wins where both exist
    print(f"  lidar covers {np.isfinite(lidar).mean():.0%} of the area")

    print("3DEP 1/3 arc-second, for the rest...")
    dep = np.full_like(lidar, np.nan)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open("/vsicurl/" + DEP_URL) as src:
            reproject(rasterio.band(src, 1), dep, dst_transform=dst_tf, dst_crs=config.CRS,
                      src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.bilinear)
    have = np.isfinite(lidar)
    w = np.clip(ndimage.distance_transform_edt(have) / FEATHER_PX, 0, 1).astype(np.float32)
    dep = np.where(np.isnan(dep), lidar, dep)       # 3DEP gaps (if any) inside the lidar
    dem = np.where(have, w * np.nan_to_num(lidar) + (1 - w) * dep, dep)
    if not np.isfinite(dem).all():
        raise RuntimeError("the terrain still has gaps after filling with 3DEP")
    tmp = config.DEM_RAW.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(dem.astype(np.float32), 1)
    tmp.replace(config.DEM_RAW)
    print(f"Saved {config.DEM_RAW.name}: {config.DEM_W} x {config.DEM_H} at {config.DEM_RES:g} m, "
          f"{np.nanmin(dem):,.0f}-{np.nanmax(dem):,.0f} m")


def naip_chunk(left: float, top: float, right: float, bottom: float, w: int, h: int) -> Image.Image:
    q = {"bbox": f"{left},{bottom},{right},{top}", "bboxSR": 26911, "imageSR": 26911, "size": f"{w},{h}",
         "format": "png32", "pixelType": "U8", "bandIds": "0,1,2", "transparent": "true",
         "interpolation": "RSP_BilinearInterpolation", "f": "json"}
    meta = json.loads(fetch(f"{NAIP_URL}?{urllib.parse.urlencode(q)}"))
    if "error" in meta or "href" not in meta:
        raise RuntimeError(f"NAIP export failed: {meta}")
    img = Image.open(io.BytesIO(fetch(meta["href"]))).convert("RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return img


def naip(path: Path, left: float, top: float, right: float, bottom: float, res: float) -> None:
    if path.exists():
        print(f"Imagery already present: {path.name}")
        return
    W, H = round((right - left) / res), round((top - bottom) / res)
    print(f"NAIP {path.name}: {W} x {H} px at {res:g} m")
    out = Image.new("RGBA", (W, H))
    for step in (4000, 2000, 1000):                 # the service caps the export size; find what it allows
        try:
            for y0 in range(0, H, step):
                for x0 in range(0, W, step):
                    w, h = min(step, W - x0), min(step, H - y0)
                    l, t = left + x0 * res, top - y0 * res
                    out.paste(naip_chunk(l, t, l + w * res, t - h * res, w, h), (x0, y0))
                    print(f"  rows {min(y0 + h, H)}/{H}, columns {min(x0 + w, W)}/{W}", end="\r")
            break
        except RuntimeError as e:
            print(f"  ({step}-px pieces refused: {str(e)[:120]}; trying smaller)")
    else:
        raise RuntimeError("NAIP export failed at every piece size")
    a = np.asarray(out.getchannel("A"))
    if a.min() < 255:
        print(f"  note: {np.mean(a < 255):.1%} of {path.name} has no imagery (filled with neighbouring colour)")
        rgb = np.asarray(out.convert("RGB")).astype(np.float32)
        hole = a < 255
        idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        rgb = rgb[idx[0], idx[1]]
        out = Image.fromarray(rgb.astype(np.uint8))
    tmp = path.with_suffix(".part.png")
    out.convert("RGB").save(tmp)
    tmp.replace(path)
    print(f"Saved {path.name}")


def main() -> int:
    config.CACHE.mkdir(parents=True, exist_ok=True)
    build_dem()
    naip(config.NAIP_CONTEXT, config.LEFT, config.TOP, config.RIGHT, config.BOTTOM, config.CONTEXT_RES)
    for path, l, t, r, b in config.valley_halves():
        naip(path, l, t, r, b, config.VALLEY_RES)
    print("Data ready. Next: pixi run prep")
    return 0


if __name__ == "__main__":
    sys.exit(main())
