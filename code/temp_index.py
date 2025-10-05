# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import os

# # ============================
# # CONFIG
# # ============================
# SST_FILE = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
# OUT_NC   = "./data/sst_on_moana_4km_2025-08-30.nc"
# OUT_NPY  = "./data/sst_4km.npy"
# OUT_NPZ  = "./data/sst_4km_with_coords.npz"
# OUT_PDF  = "./data/sst_4km_map.pdf"

# ATL_BBOX = (-100, 20, -60, 70)  # lon_min, lon_max, lat_min, lat_max
# COARSEN_FACTOR = 4              # 1 km -> ~4 km

# # ============================
# # HELPERS
# # ============================
# def find_lon_lat(ds):
#     # prefer coord names; fall back to dims
#     names = list(ds.coords) + list(ds.dims)
#     lon = next((n for n in names if "lon" in n.lower()), None)
#     lat = next((n for n in names if "lat" in n.lower()), None)
#     if lon is None and "longitude" in names: lon = "longitude"
#     if lat is None and "latitude"  in names: lat = "latitude"
#     if lon is None or lat is None:
#         raise KeyError("Couldn't find longitude/latitude coordinate names.")
#     return lon, lat

# def normalize_lon(arr):
#     return ((arr + 180) % 360) - 180

# # ============================
# # LOAD
# # ============================
# print("🔹 Opening MUR SST lazily...")
# ds = xr.open_dataset(SST_FILE, chunks={"lat": 1000, "lon": 1000})

# # Find SST variable
# sst_var = "analysed_sst" if "analysed_sst" in ds else list(ds.data_vars)[0]
# sst = ds[sst_var]

# # Detect coord names on the *dataset*
# lon_name, lat_name = find_lon_lat(ds)

# # If coords are longitude/latitude, rename on the DATASET (not the DataArray)
# rename_map = {}
# if lon_name == "longitude": rename_map["longitude"] = "lon"
# if lat_name == "latitude":  rename_map["latitude"]  = "lat"
# if rename_map:
#     ds = ds.rename(rename_map)
#     # refresh names
#     lon_name, lat_name = "lon", "lat"
#     sst = ds[sst_var]

# # Normalize longitudes to [-180, 180] and sort
# if ds[lon_name].max() > 180:
#     ds = ds.assign_coords({lon_name: normalize_lon(ds[lon_name])}).sortby(lon_name)
#     sst = ds[sst_var]

# # Crop to Atlantic
# lo1, lo2, la1, la2 = ATL_BBOX
# sst = sst.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})

# # Time slice (first if present)
# if "time" in sst.dims:
#     sst = sst.isel(time=0)

# # Kelvin -> Celsius
# if sst.attrs.get("units", "").lower() in ("k", "kelvin") or float(sst.mean()) > 200:
#     sst = sst - 273.15
# sst = sst.astype("float32")
# sst.attrs["units"] = "°C"

# # Ensure dims order is (lat, lon) for coarsen/pcolormesh
# if sst.dims != (lat_name, lon_name):
#     sst = sst.transpose(lat_name, lon_name)

# # ============================
# # COARSEN to ~4 km
# # ============================
# print(f"🔹 Coarsening to ~4 km (factor={COARSEN_FACTOR})...")
# sst_4km = sst.coarsen({lat_name: COARSEN_FACTOR, lon_name: COARSEN_FACTOR}, boundary="trim").mean()
# sst_4km = sst_4km.astype("float32").rename({lat_name: "lat", lon_name: "lon"})  # standardize names

# print(f"✅ Coarsened shape: {sst_4km.shape}, dtype: {sst_4km.dtype}")

# # ============================
# # PLOT
# # ============================
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# im = sst_4km.plot(
#     ax=ax, x="lon", y="lat",
#     transform=ccrs.PlateCarree(),
#     cmap="coolwarm", robust=True,
#     cbar_kwargs={"label": "SST (°C)"},
# )
# ax.coastlines(linewidth=0.6)
# ax.set_extent([lo1, lo2, la1, la2], crs=ccrs.PlateCarree())
# ax.set_title("MUR SST (Atlantic, ~4 km grid) — 2025-08-30")
# plt.tight_layout()

# os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
# plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
# plt.show()
# print(f"✅ Map saved → {OUT_PDF}")

# # ============================
# # SAVE
# # ============================
# os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
# sst_4km.to_dataset(name="sst").to_netcdf(OUT_NC)
# np.save(OUT_NPY, sst_4km.values)
# np.savez(OUT_NPZ, lon=sst_4km.lon.values, lat=sst_4km.lat.values, sst=sst_4km.values)
# print(f"✅ Saved NetCDF → {OUT_NC}")
# print(f"✅ Saved NumPy arrays → {OUT_NPY} and {OUT_NPZ}")


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

# ========= INPUTS (edit paths if needed) =========
SST_NC = "./data/sst_on_moana_4km_2025-08-30.nc"   # from your previous step
# If you only have NPZ, set e.g. SST_NPZ = "./data/sst_4km_with_coords.npz"
SST_NPZ = None   # or path string

# ========= OUTPUTS =========
OUT_NPY = "./data/index_2.npy"
OUT_NPZ = "./data/index_2_with_coords.npz"
OUT_PDF = "./data/index_2_map.pdf"

# ========= Gaussian settings =========
MU = 17.0                 # °C: peak suitability (index_2 = 1)
LOW, HIGH = 12.0, 22.0    # °C: ~0 at the edges
EDGE_VALUE = 0.01         # index value at LOW/HIGH (near-zero)

# ========= Load SST (°C) on 4-km grid =========
if SST_NPZ is None:
    ds = xr.open_dataset(SST_NC)
    if "sst" not in ds:
        raise KeyError(f"'sst' not found in {SST_NC}")
    sst = ds["sst"].astype("float32")
    # standardize coord names if needed
    if "longitude" in sst.coords: sst = sst.rename({"longitude":"lon"})
    if "latitude"  in sst.coords: sst = sst.rename({"latitude":"lat"})
else:
    pack = np.load(SST_NPZ)
    sst_vals = pack["sst"].astype("float32")
    lon = xr.DataArray(pack["lon"].astype("float32"), dims=("lon",))
    lat = xr.DataArray(pack["lat"].astype("float32"), dims=("lat",))
    sst = xr.DataArray(sst_vals, coords={"lat": lat, "lon": lon}, dims=("lat","lon"), name="sst")

# quick sanity
if sst.sizes.get("lat",0)==0 or sst.sizes.get("lon",0)==0:
    raise ValueError("SST grid is empty.")

# ========= Build Gaussian index_2 =========
# Choose sigma so index≈EDGE_VALUE at bounds (assuming symmetric MU)
delta = float(min(abs(MU-LOW), abs(HIGH-MU)))   # = 5 for 12–22 around 17
sigma = delta / np.sqrt(2.0 * np.log(1.0/EDGE_VALUE))

index_2 = np.exp(-((sst - MU) ** 2) / (2.0 * sigma ** 2))
index_2 = index_2.where(np.isfinite(sst)).clip(0,1).astype("float32")
index_2.name = "index_2"
index_2.attrs.update({
    "long_name": "Gaussian thermal suitability index",
    "formula": f"exp(-((T-{MU})^2)/(2*{sigma:.3f}^2)) with ~{EDGE_VALUE} at {LOW}°C & {HIGH}°C",
    "units": "1"
})
print(f"σ = {sigma:.3f} °C; index_2({LOW}°C) ≈ index_2({HIGH}°C) ≈ {EDGE_VALUE}")

# ========= Plot =========
fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
try:
    im = index_2.plot(
        ax=ax, x="lon", y="lat",
        transform=ccrs.PlateCarree(),
        cmap="plasma", vmin=0, vmax=1,
        cbar_kwargs={"label":"Index_2 (thermal suitability, 0–1)"}
    )
except Exception as e:
    # Fallback to pcolormesh if xarray's plot complains
    pcm = ax.pcolormesh(index_2.lon, index_2.lat, index_2, transform=ccrs.PlateCarree(),
                        cmap="plasma", shading="auto", vmin=0, vmax=1)
    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)
    cb.set_label("Index_2 (thermal suitability, 0–1)")
ax.coastlines(linewidth=0.6)
ax.set_title("Shark Index_2 — Gaussian Thermal Suitability (peak 17 °C; ~0 at 12/22 °C)")
plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"✅ Saved map → {OUT_PDF}")

# ========= Save NumPy (keeps 4-km resolution) =========
os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)
np.save(OUT_NPY, index_2.values)
np.savez(OUT_NPZ, lon=index_2.lon.values, lat=index_2.lat.values, index_2=index_2.values)
print(f"✅ Saved arrays → {OUT_NPY} and {OUT_NPZ}")



# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import dask
# import os

# # --- make dask lean & threaded
# dask.config.set(scheduler="threads", num_workers=4)

# SST_FILE = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"
# ATL_BBOX = (-100, 20, -60, 70)  # lon_min, lon_max, lat_min, lat_max
# COARSEN = 4                      # ~4 km from ~1 km

# def normalize_lon(a): return ((a + 180) % 360) - 180

# # 1) Open lazily + only the SST var (drop everything else)
# ds = xr.open_dataset(SST_FILE, chunks={"lat": 2000, "lon": 2000})
# sst_var = "analysed_sst" if "analysed_sst" in ds else list(ds.data_vars)[0]
# ds = ds[[sst_var]]  # drop other variables/ancillaries early
# # standardize coord names
# if "longitude" in ds: ds = ds.rename({"longitude":"lon"})
# if "latitude"  in ds: ds = ds.rename({"latitude":"lat"})
# # normalize 0–360 -> -180–180
# if ds.lon.max() > 180:
#     ds = ds.assign_coords(lon=normalize_lon(ds.lon)).sortby("lon")

# # 2) Crop Atlantic BEFORE any compute
# lo1, lo2, la1, la2 = ATL_BBOX
# sst = ds[sst_var].sel(lon=slice(lo1, lo2), lat=slice(la1, la2))

# # 3) Choose one time step (does not compute yet)
# if "time" in sst.dims: sst = sst.isel(time=0)

# # 4) Kelvin -> °C (still lazy), cast to float32
# if sst.attrs.get("units","").lower() in ("k","kelvin"):
#     sst = sst - 273.15
# sst = sst.astype("float32")
# sst.attrs["units"] = "°C"

# # 5) Coarsen to ~4 km BEFORE loading
# sst4 = sst.coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()

# # 6) Pull the much smaller array to memory once
# sst4 = sst4.load()

# # ========== FAST PLOT (imshow) ==========
# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())

# # imshow is much faster than pcolormesh on rectilinear grids
# im = ax.imshow(
#     sst4.values,
#     extent=[float(sst4.lon.min()), float(sst4.lon.max()),
#             float(sst4.lat.min()), float(sst4.lat.max())],
#     origin="lower",
#     transform=ccrs.PlateCarree(),
#     cmap="coolwarm",
#     vmin=np.nanpercentile(sst4, 2),
#     vmax=np.nanpercentile(sst4, 98),
#     interpolation="nearest"   # keeps it crisp
# )

# cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
# cb.set_label("SST (°C)")
# ax.coastlines(linewidth=0.6)
# ax.set_extent([lo1, lo2, la1, la2], crs=ccrs.PlateCarree())
# ax.set_title("MUR SST — Atlantic (~4 km, fast path)")
# plt.tight_layout()
# plt.show()
