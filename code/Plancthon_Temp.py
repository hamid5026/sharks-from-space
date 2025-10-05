# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs

# # ---------- FILES ----------
# moana_nc = "./data/PACE_OCI.20250830.L4m.DAY.MOANA.V3_1.4km.nc"
# mur_nc   = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"

# # ---------- LOAD MOANA ----------
# ds = xr.open_dataset(moana_nc)

# # Standardize coordinate names (MOANA sometimes uses lon/lat already)
# for a, b in (("longitude","lon"), ("latitude","lat")):
#     if a in ds and b not in ds:
#         ds = ds.rename({a:b})

# # Choose a MOANA variable (example)
# moana_var = "picoeuk_moana"
# arr = ds[moana_var]

# # Log for display (avoid -inf)
# log_arr = np.log10(arr.where(arr > 0))

# # Ensure lon in [-180, 180] and sorted
# if (ds.lon > 180).any():
#     ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
#     log_arr = log_arr.assign_coords(lon=ds.lon)

# # ---------- LOAD MUR ----------
# mur = xr.open_dataset(mur_nc)

# # MUR usual variable names
# mur_var = "analysed_sst" if "analysed_sst" in mur else list(mur.data_vars)[0]

# # Normalize MUR longitudes to [-180, 180] and sort
# if "lon" in mur:
#     mur = mur.assign_coords(lon=((mur.lon + 180) % 360) - 180).sortby("lon")
# elif "longitude" in mur:
#     mur = mur.rename({"longitude":"lon"})
#     mur = mur.assign_coords(lon=((mur.lon + 180) % 360) - 180).sortby("lon")

# if "latitude" in mur and "lat" not in mur:
#     mur = mur.rename({"latitude":"lat"})

# # Select the time slice if present (your file name looks single-time)
# mur_da = mur[mur_var]
# if "time" in mur_da.dims:
#     mur_da = mur_da.isel(time=0)

# # Convert from K to °C (typical for GHRSST); skip if already in °C
# mur_da_c = mur_da - 273.15

# # Mask obvious fill values
# fv = mur_da_c.attrs.get("_FillValue", None)
# if fv is not None:
#     mur_da_c = mur_da_c.where(~np.isclose(mur_da_c, fv))

# # Optional: thin the contour field so it doesn’t clutter
# mur_da_c_coarse = mur_da_c.coarsen(lat=4, lon=4, boundary="trim").mean()

# # ---------- PLOT ----------
# fig = plt.figure(figsize=(11, 5.5))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# # Background: MOANA (log10 cells/mL)
# p = log_arr.plot(
#     x="lon", y="lat",
#     ax=ax,
#     transform=ccrs.PlateCarree(),
#     cmap="viridis",
#     robust=True,
#     add_colorbar=True
# )
# p.colorbar.set_label(f"log10({moana_var})")

# # Overlay: MUR SST contours (°C)
# levels = [10, 15, 20, 25, 28, 30]
# cs = ax.contour(
#     mur_da_c_coarse["lon"], mur_da_c_coarse["lat"], mur_da_c_coarse,
#     levels=levels,
#     transform=ccrs.PlateCarree(),
#     linewidths=0.8,
#     alpha=0.8
# )
# ax.clabel(cs, fmt="%.0f°C", inline=True, fontsize=8)

# # (Optional) semi-transparent SST raster overlay — comment/uncomment
# # mur_img = mur_da_c.where(np.isfinite(mur_da_c))
# # mur_img.plot(
# #     x="lon", y="lat",
# #     ax=ax, transform=ccrs.PlateCarree(),
# #     cmap="coolwarm",
# #     alpha=0.25,
# #     add_colorbar=True
# # ).colorbar.set_label("SST (°C)")

# ax.coastlines(linewidth=0.6)
# ax.set_title(f"MOANA {moana_var} (log10) with MUR SST contours (°C)")
# plt.tight_layout()
# plt.show()


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ---------- FILES ----------
moana_nc = "./data/PACE_OCI.20250831.L4m.DAY.MOANA.V3_1.4km.nc"
mur_nc   = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"

# ---------- LOAD MOANA ----------
ds = xr.open_dataset(moana_nc)

# Standardize coord names
for a, b in (("longitude","lon"), ("latitude","lat")):
    if a in ds and b not in ds:
        ds = ds.rename({a:b})

# Choose a MOANA variable
moana_var = "picoeuk_moana"
moana = ds[moana_var]

# Keep positive values, compute log10 for display
moana_pos = moana.where(moana > 0)
moana_log = np.log10(moana_pos)

# Normalize lon to [-180, 180] and sort (safest for map overlay)
if (ds.lon > 180).any():
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    moana_pos = moana_pos.assign_coords(lon=ds.lon)
    moana_log = moana_log.assign_coords(lon=ds.lon)

# ---------- LOAD MUR ----------
mur = xr.open_dataset(mur_nc)
if "longitude" in mur: mur = mur.rename({"longitude":"lon"})
if "latitude"  in mur: mur = mur.rename({"latitude":"lat"})

# Normalize MUR lons too
mur = mur.assign_coords(lon=((mur.lon + 180) % 360) - 180).sortby("lon")

mur_var = "analysed_sst" if "analysed_sst" in mur else list(mur.data_vars)[0]
mur_da = mur[mur_var]
if "time" in mur_da.dims:
    mur_da = mur_da.isel(time=0)

# Kelvin -> °C (typical for GHRSST)
mur_sst_c = mur_da - 273.15

# Mask fill values if present
fv = mur_sst_c.attrs.get("_FillValue", None)
if fv is not None:
    mur_sst_c = mur_sst_c.where(~np.isclose(mur_sst_c, fv))

# ---------- PUT BOTH ON SAME GRID (MOANA GRID) ----------
# interpolate SST to MOANA grid for masking + vector math
mur_on_moana = mur_sst_c.interp(lon=ds.lon, lat=ds.lat, method="linear")

# ---------- DEFINE "MATCH" AREA (mask by MOANA presence) ----------
# Robust threshold: use 70th percentile of MOANA where positive.
# Tweak 'q' to make mask looser/tighter.
q = 0.70
thr = moana_pos.quantile(q)  # scalar DataArray
moana_mask = moana_pos >= thr

# Also keep finite SST
mask = moana_mask & np.isfinite(mur_on_moana)

# ---------- NORMALIZED VECTOR FIELD ----------
# We’ll create a 2D vector V = (MOANA_norm, SST_norm), then normalize to unit length.
# Use robust min-max via quantiles to reduce outlier impact.
def robust_minmax_norm(da, qlo=0.05, qhi=0.95):
    lo = float(da.quantile(qlo))
    hi = float(da.quantile(qhi))
    out = (da - lo) / (hi - lo)
    return out.clip(0, 1)

moana_n = robust_minmax_norm(moana_log.where(mask))
sst_n   = robust_minmax_norm(mur_on_moana.where(mask))

U = moana_n
V = sst_n
mag = np.sqrt(U**2 + V**2)
Uu = (U / mag).where(mag > 0)
Vv = (V / mag).where(mag > 0)

# Thin the vectors for readability (change factors to adjust density)
step_lat = 8   # every N grid points in latitude
step_lon = 8   # every N grid points in longitude
Uu_s = Uu.isel(lat=slice(None, None, step_lat), lon=slice(None, None, step_lon))
Vv_s = Vv.isel(lat=slice(None, None, step_lat), lon=slice(None, None, step_lon))
lon_s = ds.lon.isel(lon=slice(None, None, step_lon))
lat_s = ds.lat.isel(lat=slice(None, None, step_lat))

# ---------- PLOTTING ----------
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# Background: MOANA log10
p = moana_log.plot(
    x="lon", y="lat",
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    robust=True,
    add_colorbar=True
)
p.colorbar.set_label(f"log10({moana_var} cells/mL)")

# SST contours *only where MOANA mask is true*
sst_for_contours = mur_on_moana.where(mask)
# Choose neat levels; adjust to your region/season if desired
levels = [10, 15, 20, 25, 28, 30]
cs = ax.contour(
    ds.lon, ds.lat, sst_for_contours,
    levels=levels,
    transform=ccrs.PlateCarree(),
    linewidths=0.8,
    alpha=0.9
)
ax.clabel(cs, fmt="%.0f°C", inline=True, fontsize=8)

# Normalized vector (unit arrows) from merged fields
qv = ax.quiver(
    lon_s, lat_s, Uu_s, Vv_s,
    transform=ccrs.PlateCarree(),
    scale=30,   # change scale to adjust arrow length
    width=0.002,
    regrid_shape=20
)

ax.coastlines(linewidth=0.6)
ax.set_title("MOANA (log10) + masked MUR SST contours, with merged normalized vector field")
plt.tight_layout()
plt.show()
