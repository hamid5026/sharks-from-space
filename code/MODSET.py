import os
import glob
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform
import matplotlib.pyplot as plt

# Optional Cartopy for pretty map; we’ll fallback if not available
try:
    import cartopy.crs as ccrs
    CARTOPY = True
except Exception:
    CARTOPY = False

# ---------------- CONFIG ----------------
# Folder with your *.hdf tiles (e.g., MCD15A3H…hdf)
DATA_DIR = r"C:/Users/h_mog/Downloads/MCD15A3H_061-20251003_184436"     # <- change this
# Which subdataset to extract from each HDF:
# For MCD15A3H common options include 'Lai_500m' or 'Fpar_500m'
SUBDATASET_NAME = "Lai_500m"                  # or "Fpar_500m"
# Apply scale/offset for MCD15A3H (per product docs)
SCALE = 0.1
OFFSET = 0.0
# Valid range for LAI (0–100 scaled → 0–10 LAI); MODIS fill is 249/255 etc.
FILL_VALUES = {0, 249, 250, 251, 252, 253, 254, 255}  # conservative set
# Output target CRS
TARGET_CRS = "EPSG:4326"
# Target resolution in degrees (approx ~0.0045 ~ 500m near equator; choose coarser if heavy)
TARGET_RES = 0.01

# ----------------------------------------

def find_subdataset_path(hdf_path, wanted_name):
    """Return the rasterio path string to a subdataset inside HDF that matches wanted_name."""
    # For GDAL/rasterio, HDF4 subdatasets are referenced as:
    # 'HDF4_EOS:...:file.hdf:GridName:FieldName'
    # Rasterio shows them in dataset.subdatasets
    with rasterio.open(hdf_path) as ds:
        for s in ds.subdatasets:
            # s looks like: 'HDF4_EOS:EOS_GRID:"...hdf":MOD_Grid_MCD15A2H:Lai_500m'
            if s.endswith(f":{wanted_name}"):
                return s
    return None

# Gather subdataset readers as WarpedVRTs to TARGET_CRS
vrt_list = []
src_profiles = []
paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf")))
if not paths:
    raise SystemExit("No HDF files found. Check DATA_DIR.")

for hdf in paths:
    sds_path = find_subdataset_path(hdf, SUBDATASET_NAME)
    if sds_path is None:
        print(f"[WARN] {SUBDATASET_NAME} not found in {os.path.basename(hdf)}; skipping.")
        continue

    # Open subdataset
    src = rasterio.open(sds_path)

    # Build a WarpedVRT to reproject on-the-fly to TARGET_CRS & target res
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds, resolution=TARGET_RES
    )
    vrt = WarpedVRT(
        src,
        crs=TARGET_CRS,
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        resampling=Resampling.nearest,  # keep categorical-ish values safe; LAI is numeric but scaled
    )
    vrt_list.append((vrt, hdf))
    src_profiles.append(src.profile)

if not vrt_list:
    raise SystemExit(f"No subdatasets '{SUBDATASET_NAME}' found in any HDFs.")

# Merge (mosaic) all reprojected tiles
rasters = [v[0] for v in vrt_list]
mosaic, out_transform = merge(rasters, nodata=np.nan)

# merge() returns shape (bands, rows, cols). We expect single band:
mosaic = mosaic[0, :, :]

# Convert data: set fill to NaN, apply scale/offset
# Note: The raw values are integers with specific fill codes.
if FILL_VALUES:
    mask_fill = np.isin(mosaic, list(FILL_VALUES))
    mosaic = mosaic.astype("float32")
    mosaic[mask_fill] = np.nan
else:
    mosaic = mosaic.astype("float32")

mosaic = mosaic * SCALE + OFFSET

# Compute simple robust limits for display
finite = np.isfinite(mosaic)
if finite.sum() > 0:
    vmin = np.nanpercentile(mosaic, 2)
    vmax = np.nanpercentile(mosaic, 98)
else:
    vmin, vmax = 0, 1

# Plot
if CARTOPY:
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    # rasterio expects transform in origin; we can draw with imshow using extent
    left, bottom, right, top = (
        out_transform.c,
        out_transform.f + out_transform.e * mosaic.shape[0],
        out_transform.c + out_transform.a * mosaic.shape[1],
        out_transform.f,
    )
    extent = (left, right, bottom, top)
    ax.imshow(mosaic, origin="upper", extent=extent, transform=ccrs.PlateCarree(),
              vmin=vmin, vmax=vmax)
    ax.coastlines()
    ax.set_title(f"MCD15A3H {SUBDATASET_NAME} (scaled) — mosaic")
    plt.tight_layout()
    plt.show()
else:
    # Fallback: no projection; just show lon/lat extent
    fig, ax = plt.subplots(figsize=(10, 5))
    left, bottom, right, top = (
        out_transform.c,
        out_transform.f + out_transform.e * mosaic.shape[0],
        out_transform.c + out_transform.a * mosaic.shape[1],
        out_transform.f,
    )
    extent = (left, right, bottom, top)
    im = ax.imshow(mosaic, origin="upper", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"MCD15A3H {SUBDATASET_NAME} (scaled) — mosaic (no proj)")
    plt.colorbar(im, ax=ax, label=SUBDATASET_NAME)
    plt.tight_layout()
    plt.show()
