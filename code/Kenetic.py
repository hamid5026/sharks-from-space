# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import os
# # ============================================================
# # CONFIG
# # ============================================================
# FILE = "./data/dataset-uv-nrt-daily_20250830T1200Z_P20250905T0000.nc"
# OUT_PDF = "./data/ocean_currents_2025-08-30.pdf"
# TIME_INDEX = 0
# STEP = 5  # quiver thinning (higher -> fewer arrows)

# # depth dimension candidates in CMEMS
# DEPTH_NAMES = ("depth", "depthu", "depthv", "deptho", "lev", "z")

# # ============================================================
# # LOAD
# # ============================================================
# ds = xr.open_dataset(FILE)

# # Pick variable names
# if "uo" in ds and "vo" in ds:
#     u = ds["uo"]
#     v = ds["vo"]
# elif "eastward_current_velocity" in ds and "northward_current_velocity" in ds:
#     u = ds["eastward_current_velocity"]
#     v = ds["northward_current_velocity"]
# else:
#     raise KeyError("Couldn't find 'uo/vo' or 'eastward_current_velocity/northward_current_velocity'.")

# # Select a single time step if present
# if "time" in u.dims:
#     u = u.isel(time=TIME_INDEX)
#     v = v.isel(time=TIME_INDEX)

# # Select a single depth (prefer depth closest to 0 m if depth axis exists)
# depth_dim = next((d for d in u.dims if d in DEPTH_NAMES), None)
# if depth_dim is not None:
#     depth_coord = u[depth_dim]
#     # choose index of depth closest to 0.0
#     idx0 = int(np.abs(depth_coord - 0.0).argmin())
#     u = u.isel({depth_dim: idx0})
#     v = v.isel({depth_dim: idx0})

# # Identify lon/lat coordinate names
# lon_name = next((c for c in u.coords if "lon" in c.lower()), None)
# lat_name = next((c for c in u.coords if "lat" in c.lower()), None)
# if lon_name is None or lat_name is None:
#     # sometimes lon/lat are dims rather than coords
#     lon_name = next((d for d in u.dims if "lon" in d.lower()), None)
#     lat_name = next((d for d in u.dims if "lat" in d.lower()), None)
# if lon_name is None or lat_name is None:
#     raise KeyError("Longitude/latitude names not found among coords/dims.")

# lon = u[lon_name].values
# lat = u[lat_name].values

# # Normalize longitudes to [-180, 180] for map continuity
# lon = ((lon + 180) % 360) - 180

# # Ensure (lat, lon) order (most CMEMS are [lat, lon], but be safe)
# # Get axis indices
# lat_axis = u.dims.index(lat_name)
# lon_axis = u.dims.index(lon_name)
# if (lat_axis, lon_axis) != (0, 1):
#     u = u.transpose(lat_name, lon_name)
#     v = v.transpose(lat_name, lon_name)

# # Compute speed (2D now)
# speed = np.sqrt(u**2 + v**2)

# # Downsample for quiver readability
# u_sub = u.values[::STEP, ::STEP]
# v_sub = v.values[::STEP, ::STEP]
# lon_sub = lon[::STEP]
# lat_sub = lat[::STEP]

# # ============================================================
# # PLOT
# # ============================================================
# fig = plt.figure(figsize=(12, 6))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# # Filled color background: current speed (m/s)
# pc = ax.pcolormesh(
#     lon, lat, speed.values,           # 1D lon/lat + 2D C (lat, lon)
#     transform=ccrs.PlateCarree(),
#     cmap="turbo",
#     shading="auto",                   # handles cell edges nicely
# )
# cb = plt.colorbar(pc, orientation="horizontal", pad=0.05)
# cb.set_label("Surface current speed (m/s)")

# # Quiver arrows (direction)
# Q = ax.quiver(
#     lon_sub, lat_sub, u_sub, v_sub,
#     transform=ccrs.PlateCarree(),
#     scale=45, width=0.002, color="k", alpha=0.3
# )
# ax.quiverkey(Q, 0.9, -0.05, 1, "1 m/s", labelpos="E")

# ax.coastlines(linewidth=0.6)
# ax.set_global()
# ax.set_title("Copernicus Marine Surface Currents (surface, daily)")

# plt.tight_layout()
# # ============================================================
# # SAVE AS VECTOR PDF
# # ============================================================
# # os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
# # plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
# # print(f"✅ Saved vector PDF: {OUT_PDF}")
# plt.show()


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

# ============================================================
# CONFIG
# ============================================================
FILE = "./data/dataset-uv-nrt-daily_20250830T1200Z_P20250905T0000.nc"
OUT_PDF = "./data/ocean_currents_ATL_2025-08-30.pdf"

# Atlantic bbox (lon_min, lon_max, lat_min, lat_max)
ATL_BBOX = (-100, 20, -60, 70)

TIME_INDEX = 0       # pick a time
STEP = 5             # quiver thinning

DEPTH_NAMES = ("depth", "depthu", "depthv", "deptho", "lev", "z")

# ============================================================
# HELPERS
# ============================================================
def normalize_lon(vals):
    vals = np.asarray(vals)
    return ((vals + 180) % 360) - 180

def find_lon_lat(da):
    lon_name = next((c for c in list(da.coords) + list(da.dims) if "lon" in c.lower()), None)
    lat_name = next((c for c in list(da.coords) + list(da.dims) if "lat" in c.lower()), None)
    if lon_name is None or lat_name is None:
        # common alt names
        if "longitude" in da.coords or "longitude" in da.dims: lon_name = "longitude"
        if "latitude"  in da.coords or "latitude"  in da.dims: lat_name = "latitude"
    if lon_name is None or lat_name is None:
        raise KeyError("Longitude/latitude names not found.")
    return lon_name, lat_name

# ============================================================
# LOAD (chunked so cropping happens before compute)
# ============================================================
ds = xr.open_dataset(FILE, chunks={"latitude": 600, "longitude": 600})

# Pick variable names
if "uo" in ds and "vo" in ds:
    u = ds["uo"]; v = ds["vo"]
elif "eastward_current_velocity" in ds and "northward_current_velocity" in ds:
    u = ds["eastward_current_velocity"]; v = ds["northward_current_velocity"]
else:
    raise KeyError("Couldn't find 'uo/vo' or 'eastward_current_velocity/northward_current_velocity'.")

# Time slice
if "time" in u.dims:
    u = u.isel(time=TIME_INDEX)
    v = v.isel(time=TIME_INDEX)

# Surface (depth closest to 0 m)
depth_dim = next((d for d in u.dims if d in DEPTH_NAMES), None)
if depth_dim is not None:
    idx0 = int(np.abs(u[depth_dim] - 0.0).argmin())
    u = u.isel({depth_dim: idx0})
    v = v.isel({depth_dim: idx0})

# Identify lon/lat; normalize to [-180, 180] and sort
lon_name, lat_name = find_lon_lat(u)
if u[lon_name].max() > 180:
    new_lon = normalize_lon(u[lon_name].values)
    u = u.assign_coords({lon_name: new_lon}).sortby(lon_name)
    v = v.assign_coords({lon_name: new_lon}).sortby(lon_name)

# Crop to Atlantic early (speed!)
lo1, lo2, la1, la2 = ATL_BBOX
u = u.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})
v = v.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})

# Ensure dims order (lat, lon)
if u.dims.index(lat_name) != 0 or u.dims.index(lon_name) != 1:
    u = u.transpose(lat_name, lon_name)
    v = v.transpose(lat_name, lon_name)

# Pull coord vectors
lon = u[lon_name].values
lat = u[lat_name].values

# Compute speed (2D)
speed = np.sqrt(u**2 + v**2)

# Downsample for quiver
u_sub = u.values[::STEP, ::STEP]
v_sub = v.values[::STEP, ::STEP]
lon_sub = lon[::STEP]
lat_sub = lat[::STEP]

# ============================================================
# PLOT (Atlantic only)
# ============================================================
fig = plt.figure(figsize=(11, 6.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

pc = ax.pcolormesh(
    lon, lat, speed.values,
    transform=ccrs.PlateCarree(),
    cmap="turbo",
    shading="auto",
)
cb = plt.colorbar(pc, orientation="horizontal", pad=0.05)
cb.set_label("Surface current speed (m/s)")

Q = ax.quiver(
    lon_sub, lat_sub, u_sub, v_sub,
    transform=ccrs.PlateCarree(),
    scale=45, width=0.002, color="k", alpha=0.3
)
ax.quiverkey(Q, 0.9, -0.05, 1, "1 m/s", labelpos="E")

ax.coastlines(linewidth=0.6)
ax.set_extent([lo1, lo2, la1, la2], crs=ccrs.PlateCarree())
ax.set_title("Copernicus Marine Surface Currents — Atlantic (surface, daily)")

plt.tight_layout()

# Optional: save vector PDF
# os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
# plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
# print(f"✅ Saved vector PDF: {OUT_PDF}")

plt.show()


