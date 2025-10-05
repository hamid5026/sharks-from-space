from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from IPython.display import JSON
import cartopy
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# print("\r\nOK with:", sys.executable)

# user: 
#pass: 
# --- CONFIG ---
AOI_BBOX = (-30.0, 30.0, -10.0, 40.0)     # (min_lon, min_lat, max_lon, max_lat) – change to your region
auth = earthaccess.login(persist=True)
results = earthaccess.search_datasets(
    keyword="L2 ocean color",
    instrument="MODIS",
)
for item in results:
    summary = item.summary()
    print(summary["short-name"])
tspan = ("2020-10-15", "2020-10-23")
bbox = (-76.75, 36.97, -75.74, 39.01)
cc = (0, 50)

results = earthaccess.search_data(
    short_name="MODISA_L2_OC",
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=cc,
)
# print (results[0])
paths = earthaccess.download(results, "data")
prod = xr.open_dataset(paths[0])
obs = xr.open_dataset(paths[0], group="geophysical_data")
nav = xr.open_dataset(paths[0], group="navigation_data")

nav = (
    nav
    .set_coords(("longitude", "latitude"))
    .rename({"pixel_control_points": "pixels_per_line"})
)
dataset = xr.merge((prod, obs, nav.coords))

array = np.log10(dataset["chlor_a"])
array.attrs.update(
    {
        "units": f'log10({dataset["chlor_a"].attrs["units"]})',
    }
)
plot = array.plot(
    x="longitude", y="latitude", aspect=2, size=4, cmap="jet", robust=True
)
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection=cartopy.crs.PlateCarree())
array.plot(x="longitude", y="latitude", cmap="jet", robust=True, ax=ax)
ax.gridlines(draw_labels={"bottom": "x", "left": "y"})
ax.add_feature(cartopy.feature.STATES, linewidth=0.5)
ax.set_title(dataset.attrs["product_name"], loc="center")
plt.show()