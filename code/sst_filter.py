# import os
# import numpy as np
# import xarray as xr

# # ============================
# # CONFIG
# # ============================
# MOANA_NC = "./data/PACE_OCI.20250830.L4m.DAY.MOANA.V3_1.4km.nc"
# MUR_NC   = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"

# TARGET_DATE = "2025-08-30"          # pick the daily slice you want
# OUT_NC = f"./data/sst_on_moana_4km_{TARGET_DATE}.nc"

# # If you want to restrict to Atlantic only while regridding, set:
# ATLANTIC_ONLY = False
# ATL_BBOX = (-100, 20, -60, 70)      # (lon_min, lon_max, lat_min, lat_max)

# # Dask chunk sizes (tune for your RAM)
# MUR_CHUNKS = {"lat": 1000, "lon": 1000}

# # ============================
# # HELPERS
# # ============================
# def normalize_lon_vals(vals):
#     vals = np.asarray(vals)
#     return ((vals + 180) % 360) - 180

# def normalize_lon_coords(ds, lon_name="lon"):
#     if lon_name in ds.coords:
#         ds = ds.assign_coords({lon_name: normalize_lon_vals(ds[lon_name].values)}).sortby(lon_name)
#     return ds

# def find_lon_lat(da_or_ds):
#     names = list(getattr(da_or_ds, "coords", {})) + list(getattr(da_or_ds, "dims", ()))
#     lon = next((n for n in names if "lon" in n.lower()), None)
#     lat = next((n for n in names if "lat" in n.lower()), None)
#     if lon is None or lat is None:
#         raise KeyError("Could not find lon/lat names.")
#     return lon, lat

# def crop_bbox(ds, lon_name, lat_name, bbox):
#     lo1, lo2, la1, la2 = bbox
#     if float(ds[lon_name].max()) > 180:
#         ds = normalize_lon_coords(ds, lon_name)
#     return ds.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})

# def safe_interp(src_da, tgt_lon, tgt_lat, lon_name, lat_name, primary="linear"):
#     # drop all-NaN rows/cols to avoid empty reductions
#     src_da = src_da.where(np.isfinite(src_da))
#     src_da = src_da.dropna(dim=lat_name, how="all").dropna(dim=lon_name, how="all")
#     # try linear, fallback to nearest
#     try:
#         out = src_da.interp({lon_name: tgt_lon, lat_name: tgt_lat}, method=primary)
#         if np.isnan(out).all():
#             out = src_da.interp({lon_name: tgt_lon, lat_name: tgt_lat}, method="nearest")
#     except ValueError:
#         out = src_da.interp({lon_name: tgt_lon, lat_name: tgt_lat}, method="nearest")
#     return out

# # ============================
# # 1) Load MOANA grid (target 4 km)
# # ============================
# mo = xr.open_dataset(MOANA_NC)
# if "longitude" in mo: mo = mo.rename({"longitude": "lon"})
# if "latitude"  in mo: mo = mo.rename({"latitude": "lat"})
# mo = normalize_lon_coords(mo, "lon")

# # Optional: restrict MOANA grid to Atlantic
# if ATLANTIC_ONLY:
#     mo = crop_bbox(mo, "lon", "lat", ATL_BBOX)

# tgt_lon = mo["lon"]
# tgt_lat = mo["lat"]

# if tgt_lon.size == 0 or tgt_lat.size == 0:
#     raise ValueError("Target MOANA grid is empty after cropping. Check ATL_BBOX.")

# # ============================
# # 2) Load MUR SST lazily & prep
# # ============================
# mur = xr.open_dataset(MUR_NC, chunks=MUR_CHUNKS, decode_timedelta=False)
# # standardize coord names
# if "longitude" in mur: mur = mur.rename({"longitude": "lon"})
# if "latitude"  in mur: mur = mur.rename({"latitude": "lat"})
# mur = normalize_lon_coords(mur, "lon")

# # choose SST var
# sst_name = "analysed_sst" if "analysed_sst" in mur else list(mur.data_vars)[0]
# sst = mur[sst_name]

# # convert K->°C if needed
# if sst.attrs.get("units", "").lower() in ("k", "kelvin") or (sst.max().compute().item() > 200):
#     sst = sst - 273.15
# sst.attrs["units"] = "degC"
# sst.name = "sst"

# # time → daily slice
# if "time" in sst.dims:
#     sst_daily = sst.resample(time="1D").mean()
#     # choose the day (exact or nearest)
#     if TARGET_DATE in sst_daily.time.dt.strftime("%Y-%m-%d").values:
#         sst = sst_daily.sel(time=TARGET_DATE)
#     else:
#         sst = sst_daily.sel(time=TARGET_DATE, method="nearest")

# # Reduce MUR spatial domain to just the MOANA extent (big speed boost)
# lon_min = float(tgt_lon.min())
# lon_max = float(tgt_lon.max())
# lat_min = float(tgt_lat.min())
# lat_max = float(tgt_lat.max())
# sst = sst.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

# # ============================
# # 3) Interpolate MUR -> MOANA 4 km grid
# # ============================
# lon_name, lat_name = find_lon_lat(sst)
# sst_on_mo = safe_interp(sst, tgt_lon, tgt_lat, lon_name, lat_name, primary="linear")

# # ============================
# # 4) Save as compressed NetCDF
# # ============================
# comp = dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.float32(np.nan), chunksizes=(tgt_lat.size, tgt_lon.size))
# ds_out = xr.Dataset(
#     {"sst": sst_on_mo.astype("float32")},
#     coords={"lon": tgt_lon, "lat": tgt_lat},
#     attrs={
#         "title": "MUR SST regridded to MOANA 4 km grid",
#         "source_sst": os.path.basename(MUR_NC),
#         "target_grid": os.path.basename(MOANA_NC),
#         "units": "degC",
#         "history": "Interpolated with xarray.interp (linear->nearest fallback)",
#     },
# )
# encoding = {"sst": comp, "lon": {"dtype": "float32"}, "lat": {"dtype": "float32"}}

# os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
# ds_out.to_netcdf(OUT_NC, encoding=encoding)
# print(f"✅ Saved: {OUT_NC}")


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Path to your regridded file
SST_ON_MOANA = "./data/sst_on_moana_4km_2025-08-30.nc"

# ============================
# LOAD & INSPECT
# ============================
ds = xr.open_dataset(SST_ON_MOANA)
print(ds)

# The main variable should be "sst"
sst = ds["sst"]

# Optional: inspect min/max
print("SST range (°C):", float(sst.min()), "→", float(sst.max()))

# ============================
# PLOT
# ============================
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Filled color plot
im = sst.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="coolwarm",
    robust=True,
    cbar_kwargs={"label": "Sea Surface Temperature (°C)"},
)

# Coastlines and title
ax.coastlines(linewidth=0.6)
ax.set_title("MUR SST regridded to MOANA 4 km grid — 2025-08-30")

plt.tight_layout()
plt.show()
