import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Path to your MOANA netCDF file (downloaded or via OPeNDAP)
fname = "./data/PACE_OCI.20250831.L4m.DAY.MOANA.V3_1.4km.nc"

ds = xr.open_dataset(fname)

# (Optional) set the lon/lat arrays as coordinates if not already
if "longitude" in ds and "latitude" in ds:
    ds = ds.set_coords(("longitude", "latitude"))

# Pick one variable to view — e.g. picoeukaryotes
var = "picoeuk_moana"
arr = ds[var]

# Log scale for display (cells mL⁻1)
log_arr = np.log10(arr)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection=ccrs.Robinson())
# Use PlateCarree as data coord transform
log_arr.plot(
    x="lon", y="lat",
    ax=ax, transform=ccrs.PlateCarree(),
    cmap="viridis", robust=True
)
ax.coastlines()
ax.set_title(f"MOANA {var} (log₁₀ cells/mL)")
plt.show()
