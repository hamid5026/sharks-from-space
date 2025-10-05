import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

# =========================
# CONFIG
# =========================
MOANA_FILE = "./data/PACE_OCI.20250830.L4m.DAY.MOANA.V3_1.4km.nc"
VAR_NAME   = "picoeuk_moana"       # <-- pick one of the MOANA variables
ATL_BBOX   = (-100, 20, -60, 70)   # lon_min, lon_max, lat_min, lat_max

OUT_NPY = "./data/index_3.npy"
OUT_NPZ = "./data/index_3_with_coords.npz"
OUT_PDF = "./data/index_3_map.pdf"

# =========================
# HELPERS
# =========================
def normalize_lon(a): 
    """Wrap 0–360° longitudes to -180–180°."""
    return ((a + 180) % 360) - 180

# =========================
# LOAD & PREPROCESS
# =========================
print(f"🔹 Opening {MOANA_FILE}")
ds = xr.open_dataset(MOANA_FILE, chunks={"lat": 800, "lon": 800})

# fix coordinate names
if "longitude" in ds and "lon" not in ds:
    ds = ds.rename({"longitude": "lon"})
if "latitude" in ds and "lat" not in ds:
    ds = ds.rename({"latitude": "lat"})

# normalize longitudes to [-180, 180]
if ds.lon.max() > 180:
    ds = ds.assign_coords(lon=normalize_lon(ds.lon)).sortby("lon")

# select variable
if VAR_NAME not in ds:
    raise KeyError(f"{VAR_NAME!r} not found in {MOANA_FILE}. Available: {list(ds.data_vars)}")
mo = ds[VAR_NAME]

# crop early for speed
lo1, lo2, la1, la2 = ATL_BBOX
mo = mo.sel(lon=slice(lo1, lo2), lat=slice(la1, la2))

# keep only valid, positive values
mo = mo.where(np.isfinite(mo) & (mo > 0)).astype("float32")

if mo.size == 0:
    raise ValueError("Atlantic subset is empty — check ATL_BBOX or variable coverage.")

# =========================
# NORMALIZE → index_3 ∈ [0,1]
# =========================
vmin = float(mo.min())
vmax = float(mo.max())
print(f"Data range before normalization: min={vmin:.3e}, max={vmax:.3e}")

if np.isclose(vmax, vmin) or not np.isfinite(vmax):
    index_3 = xr.zeros_like(mo, dtype="float32")
else:
    index_3 = ((mo - vmin) / (vmax - vmin)).clip(0, 1).astype("float32")

index_3.name = "index_3"
index_3.attrs.update({
    "long_name": f"Normalized {VAR_NAME}",
    "definition": f"(x - {vmin:.3e}) / ({vmax:.3e} - {vmin:.3e})",
    "units": "1",
})
print("✅ index_3 created successfully.")

# =========================
# PLOT (Atlantic)
# =========================
fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
im = index_3.plot(
    ax=ax, x="lon", y="lat",
    transform=ccrs.PlateCarree(),
    cmap="viridis", vmin=0, vmax=1,
    cbar_kwargs={"label": "Index_3 (normalized MOANA, 0–1)"},
)
ax.coastlines(linewidth=0.6)
ax.set_extent([lo1, lo2, la1, la2], crs=ccrs.PlateCarree())
ax.set_title(f"Index_3 — Normalized {VAR_NAME} (Atlantic)")
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ Saved plot → {OUT_PDF}")

# =========================
# SAVE NumPy arrays
# =========================
os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
np.save(OUT_NPY, index_3.values)
np.savez(OUT_NPZ, lon=index_3.lon.values, lat=index_3.lat.values, index_3=index_3.values)
print(f"✅ Saved data → {OUT_NPY} and {OUT_NPZ}")
