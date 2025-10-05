# from matplotlib import pyplot as plt
# import cartopy
# import earthaccess
# import numpy as np
# import xarray as xr
# auth = earthaccess.login(persist=True)
# results = earthaccess.search_datasets(
#     keyword="L3m ocean color modis aqua chlorophyll",
#     instrument = "MODIS",
# )
# set((i.summary()["short-name"] for i in results))
# tspan = ("2025-08-20", "2025-09-6")
# results = earthaccess.search_data(
#     short_name="MODISA_L3m_CHL",
#     temporal=tspan,
# )
# results = earthaccess.search_data(
#     short_name="MODISA_L3m_CHL",
#     granule_name="*.8D*.9km*",
#     temporal=tspan,
# )

# paths = earthaccess.download(results, "data")
# dataset = xr.open_dataset(paths[0])
# # dataset
# array = np.log10(dataset["chlor_a"])

# # --- 1) Min–max normalize to [0, 1] ---
# vmin = float(array.min(skipna=True))
# vmax = float(array.max(skipna=True))
# norm = (array - vmin) / (vmax - vmin)
# norm = norm.clip(0, 1)  # just in case of any tiny numeric overshoot
# norm.name = "chl_norm"
# norm.attrs.update({
#     "long_name": "Normalized log10 chlorophyll-a",
#     "units": "1",
#     "comment": f"Min–max normalized from vmin={vmin:.6g}, vmax={vmax:.6g} on log10(chlor_a)."
# })

# # --- 2) Plot normalized map ---
# import cartopy.crs as ccrs  # add this if missing

# crs_proj = ccrs.Robinson()        # projection for plotting
# crs_data = ccrs.PlateCarree()     # projection of the data (lat/lon grid)
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(projection=crs_proj)
# norm.plot(x="lon", y="lat", cmap="viridis", ax=ax, robust=True, transform=crs_data)
# ax.coastlines()
# ttl = dataset.attrs.get("product_name", "MODIS-Aqua chlor_a (normalized)")
# ax.set_title(f"{ttl}\n(min–max on log10(chl_a))")
# plt.tight_layout()
# plt.savefig("data/chl_norm_map.png", dpi=200)
# plt.show()

# # --- 3) Save gridded data (lat/lon + values) as NetCDF ---
# # This keeps the geolocated grid intact and is lossless.
# ds_out = xr.Dataset(
#     data_vars={"chl_norm": norm.astype("float32")},
#     coords={"lat": dataset["lat"], "lon": dataset["lon"]},
#     attrs={
#         "source_product": dataset.attrs.get("product_name", ""),
#         "normalization": "min-max on log10(chlor_a)",
#         "vmin_log10": vmin,
#         "vmax_log10": vmax,
#     },
# )
# ds_out.to_netcdf("data/chl_norm_grid.nc")

# # --- 4) Also save a tidy CSV (lat, lon, value) for quick use ---
# # Drop NaNs to keep the file small.
# df = norm.to_dataframe().reset_index().dropna(subset=["chl_norm"])
# df.to_csv("data/chl_norm_points.csv", index=False)
# print(f"Saved {len(df):,} valid points to data/chl_norm_points.csv")



# array.attrs.update(
#     {
#         "units": f'log10({dataset["chlor_a"].attrs["units"]})',
#     }
# )
# crs_proj = cartopy.crs.Robinson()
# crs_data = cartopy.crs.PlateCarree()
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(projection=crs_proj)
# array.plot(x="lon", y="lat", cmap="jet", ax=ax, robust=True, transform=crs_data)
# ax.coastlines()
# ax.set_title(dataset.attrs["product_name"])
# plt.show()

# -*- coding: utf-8 -*-
# MODIS-Aqua chlorophyll -> log10 -> min-max normalize -> Atlantic subset -> plot & save

import os
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import earthaccess

# =================== SETTINGS ===================
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Time window (inclusive); fix your earlier typo to 2025-09-06
TSPAN = ("2025-08-20", "2025-09-06")

# Atlantic bounding box (tune if you want wider east: e.g., LON_MAX=40)
LAT_MIN, LAT_MAX = -70, 70
LON_MIN, LON_MAX = -100, 20

# =================== 1) DOWNLOAD DATA ===================
auth = earthaccess.login(persist=True)

results = earthaccess.search_data(
    short_name="MODISA_L3m_CHL",
    granule_name="*.8D*.9km*",
    temporal=TSPAN,
)
paths = earthaccess.download(results, str(DATA_DIR))
if not paths:
    raise RuntimeError("No MODISA_L3m_CHL granules found in this time window.")
ds = xr.open_dataset(paths[0])
print(f"Opened dataset: {paths[0]}")

# =================== 2) PREPARE chlor_a (robust numeric) ===================
# Normalize coordinate names
if "latitude" in ds:
    ds = ds.rename({"latitude": "lat"})
if "longitude" in ds:
    ds = ds.rename({"longitude": "lon"})

# Decode CF if needed (often automatic)
try:
    ds = xr.decode_cf(ds)
except Exception:
    pass

# Ensure coords are numeric float
ds = ds.assign_coords(
    lat=xr.DataArray(ds["lat"].astype("float64"), dims=("lat",)),
    lon=xr.DataArray(ds["lon"].astype("float64"), dims=("lon",)),
)

# Convert 0..360 -> -180..180 (if needed) and sort lon ascending
lon_vals = ds["lon"].values
if np.nanmax(lon_vals) > 180.0:
    lon_std = ((lon_vals + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_std)
    ds = ds.assign_coords(lon=lon_std).isel(lon=order)

# Sort both coords ascending (lat is often 90->-90 in L3m)
ds = ds.sortby("lat").sortby("lon")

# Extract chlor_a, make float, mask non-positive before log10
chl = ds["chlor_a"]
chl = xr.where(np.isfinite(chl), chl, np.nan).astype("float32")
chl = chl.where(chl > 0)

# log10 and min–max normalize to [0,1]
array = xr.apply_ufunc(np.log10, chl)
vmin = float(array.min(skipna=True).values)
vmax = float(array.max(skipna=True).values)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    raise ValueError(f"No numeric spread after masking/log10: vmin={vmin}, vmax={vmax}")

norm = ((array - vmin) / (vmax - vmin)).clip(0, 1).astype("float32")
norm.name = "chl_norm"
norm.attrs.update({
    "long_name": "Normalized log10 chlorophyll-a",
    "units": "1",
    "comment": f"Min–max normalized on log10(chlor_a); vmin={vmin:.6g}, vmax={vmax:.6g}",
})

# =================== 3) ROBUST ATLANTIC SUBSET (mask, not slice) ===================
lat = norm["lat"]
lon = norm["lon"]
mask = (
    (lat >= LAT_MIN) & (lat <= LAT_MAX) &
    (lon >= LON_MIN) & (lon <= LON_MAX)
)
atl = norm.where(mask, drop=True)

# Diagnostics (helpful if you ever hit empties)
print("lat range:", float(norm.lat.min()), "→", float(norm.lat.max()))
print("lon range:", float(norm.lon.min()), "→", float(norm.lon.max()))
print("Atlantic shape:", tuple(atl.shape), "finite count:", int(np.isfinite(atl.values).sum()))
if atl.size == 0 or int(np.isfinite(atl.values).sum()) == 0:
    raise ValueError(
        "Atlantic subset empty or all-NaN after coord fixes. "
        "Try widening LON_MIN/LON_MAX or verify granule coverage."
    )

# =================== 4) PLOT ATLANTIC ===================
fig = plt.figure(figsize=(10.8, 5.6))
ax = fig.add_subplot(projection=ccrs.Robinson())
pc = atl.plot(
    x="lon", y="lat",
    cmap="viridis", robust=True,
    ax=ax, transform=ccrs.PlateCarree(),
    cbar_kwargs={"label": "chl_norm (0–1)"},
)
ax.coastlines()
ttl = ds.attrs.get("product_name", "MODIS-Aqua chlor_a (normalized)")
ax.set_title(f"Atlantic Ocean – {ttl}")
plt.tight_layout()
plt.savefig(DATA_DIR / "atlantic_chl_norm_map.png", dpi=220)
plt.show()

# =================== 5) SAVE EXACTLY WHAT WAS PLOTTED ===================
# NetCDF (gridded)
ds_atl = xr.Dataset(
    data_vars={"chl_norm": atl.astype("float32")},
    coords={"lat": atl["lat"], "lon": atl["lon"]},
    attrs={
        "region": "Atlantic",
        "source_product": ds.attrs.get("product_name", ""),
        "normalization": "min-max on log10(chlor_a)",
        "vmin_log10": vmin,
        "vmax_log10": vmax,
        "bbox": f"lat[{LAT_MIN},{LAT_MAX}], lon[{LON_MIN},{LON_MAX}]",
        "source_file": Path(paths[0]).name,
    },
)
ds_atl.to_netcdf(DATA_DIR / "atlantic_chl_norm_grid.nc")

# CSV (lat, lon, chl_norm)
df_atl = atl.to_dataframe().reset_index().dropna(subset=["chl_norm"])
df_atl.to_csv(DATA_DIR / "atlantic_chl_norm_points.csv", index=False)

# NPZ (compact arrays for quick reload)
np.savez(
    DATA_DIR / "atlantic_chl_norm_with_coords.npz",
    lat=atl["lat"].values,
    lon=atl["lon"].values,
    chl_norm=atl.values,
)

print("Saved:")
print(f"  PNG -> {DATA_DIR / 'atlantic_chl_norm_map.png'}")
print(f"  NC  -> {DATA_DIR / 'atlantic_chl_norm_grid.nc'}")
print(f"  CSV -> {DATA_DIR / 'atlantic_chl_norm_points.csv'}  ({len(df_atl):,} rows)")
print(f"  NPZ -> {DATA_DIR / 'atlantic_chl_norm_with_coords.npz'}")
