import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ---------- FILE ----------
mur_nc = "./data/20250830090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc"

# ---------- SETTINGS ----------
# Reduce resolution: plot every N×N block as its mean
DECIMATE = 20   # try 10–40; higher = lighter memory
# Optional crop (lon_min, lon_max, lat_min, lat_max); set to None for global
CROP = None  # e.g., (-90, 0, 0, 60)

# ---------- LOAD (chunked) ----------
mur = xr.open_dataset(mur_nc, chunks={"lat": 1000, "lon": 1000})

# Normalize coordinate names
if "longitude" in mur: mur = mur.rename({"longitude": "lon"})
if "latitude"  in mur: mur = mur.rename({"latitude": "lat"})

# Normalize longitude to [-180, 180] and sort
mur = mur.assign_coords(lon=((mur.lon + 180) % 360) - 180).sortby("lon")

# Pick SST var and time slice
mur_var = "analysed_sst" if "analysed_sst" in mur else list(mur.data_vars)[0]
sst = mur[mur_var]
if "time" in sst.dims:
    sst = sst.isel(time=0)

# Kelvin -> Celsius
sst_c = sst - 273.15

# Optional crop BEFORE decimating (saves a lot of memory)
if CROP is not None:
    lon_min, lon_max, lat_min, lat_max = CROP
    sst_c = sst_c.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

# Downsample/coarsen to reduce pixels massively
# boundary="trim" to keep whole blocks only
sst_c_ds = sst_c.coarsen(lat=DECIMATE, lon=DECIMATE, boundary="trim").mean()

# Compute the smaller array into memory now
sst_c_small = sst_c_ds.compute()

# ---------- PLOT (imshow is lighter than pcolormesh) ----------
fig = plt.figure(figsize=(11, 5.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

# xarray's imshow uses faster image path; robust=True avoids outliers in color scale
im = sst_c_small.plot.imshow(
    x="lon", y="lat",
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="coolwarm",
    robust=True,
    add_colorbar=True,
    rasterized=True,
)

ax.coastlines(linewidth=0.6)
ax.set_title("MUR GHRSST Sea Surface Temperature (°C) — downsampled")
if CROP is not None:
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
else:
    ax.set_global()

plt.tight_layout()
plt.show()
