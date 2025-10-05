import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

# ============================
# CONFIG
# ============================
FILE = "./data/dataset-uv-nrt-daily_20250830T1200Z_P20250905T0000.nc"
TIME_INDEX = 0
DEPTH_NAMES = ("depth", "depthu", "depthv", "deptho", "lev", "z")
ATL_BBOX = (-100, 20, -60, 70)  # lon_min, lon_max, lat_min, lat_max

# quantile targets (robust scaling)
R_QLO, R_QHI = 0.10, 0.90
SAMPLE_STEP = 8   # use 8–16 for lighter memory; or set COARSEN = {"lat":4,"lon":4}
COARSEN = None

R_EARTH = 6371000.0
OMEGA   = 7.292115e-5
EPS     = 1e-12

# ============================
# HELPERS
# ============================
def normalize_lon(a): return ((a + 180) % 360) - 180

def find_lon_lat(da):
    lon = next((c for c in list(da.coords)+list(da.dims) if "lon" in c.lower()), None)
    lat = next((c for c in list(da.coords)+list(da.dims) if "lat" in c.lower()), None)
    if lon is None and "longitude" in da.coords: lon = "longitude"
    if lat is None and "latitude"  in da.coords: lat = "latitude"
    if lon is None or lat is None: raise KeyError("Could not find lon/lat names.")
    return lon, lat

def crop_bbox(ds, lon_name, lat_name, bbox):
    lo1, lo2, la1, la2 = bbox
    # normalize longitudes if 0-360
    if float(ds[lon_name].max()) > 180:
        ds = ds.assign_coords({lon_name: normalize_lon(ds[lon_name])}).sortby(lon_name)
    return ds.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})

def okubo_weiss_and_vorticity(u, v, lon, lat):
    """Compute Okubo–Weiss (W) and relative vorticity (omega) on a lat-lon grid."""
    latv = lat.values; lonv = lon.values
    lat_rad = np.deg2rad(latv)
    dphi = np.deg2rad(np.gradient(latv))                 # 1D
    dlam = np.deg2rad(np.gradient(lonv))                 # 1D
    dy = (R_EARTH * dphi).astype(np.float32)             # 1D (m)
    dx = (R_EARTH * np.cos(lat_rad)[:, None] * dlam[None, :]).astype(np.float32)  # 2D (m)

    U = u.astype("float32").values
    V = v.astype("float32").values
    dudy = np.full_like(U, np.nan, dtype=np.float32); dvdy = np.full_like(U, np.nan, dtype=np.float32)
    dudx = np.full_like(U, np.nan, dtype=np.float32); dvdx = np.full_like(U, np.nan, dtype=np.float32)

    i = slice(1, -1); j = slice(1, -1)
    dudy[i,:] = (U[2:,:] - U[:-2,:]) / (2.0 * dy[1:-1, None])
    dvdy[i,:] = (V[2:,:] - V[:-2,:]) / (2.0 * dy[1:-1, None])
    dudx[:,j] = (U[:,2:] - U[:,:-2]) / (2.0 * dx[:,1:-1])
    dvdx[:,j] = (V[:,2:] - V[:,:-2]) / (2.0 * dx[:,1:-1])

    s_n = dudx - dvdy
    s_s = dvdx + dudy
    omega = dvdx - dudy
    W = s_n**2 + s_s**2 - omega**2

    # dims & coord names MUST match the arrays: use 'lat','lon'
    return (
        xr.DataArray(W,     coords={"lat": lat, "lon": lon}, dims=("lat","lon")),
        xr.DataArray(omega, coords={"lat": lat, "lon": lon}, dims=("lat","lon"))
    )

def strength_from_radius(radius_km, qlo=R_QLO, qhi=R_QHI, step=SAMPLE_STEP, coarsen=COARSEN):
    """Memory-safe scaling: percentiles from a small subset, apply to full."""
    r = radius_km.where(np.isfinite(radius_km))
    if coarsen:
        r_small = r.coarsen(**coarsen, boundary="trim").mean()
    else:
        r_small = r.isel(lat=slice(None, None, step), lon=slice(None, None, step))
    r_vals = r_small.astype("float32").values
    rlo = np.nanpercentile(r_vals, qlo*100.0)
    rhi = np.nanpercentile(r_vals, qhi*100.0)
    if not np.isfinite(rlo) or not np.isfinite(rhi) or np.isclose(rhi, rlo):
        return xr.zeros_like(radius_km, dtype="float32")
    strength = (rhi - radius_km) / max(rhi - rlo, 1e-6)
    return strength.clip(0, 1).astype("float32")

# ============================
# LOAD & STANDARDIZE NAMES
# ============================
ds = xr.open_dataset(FILE, chunks={"latitude": 600, "longitude": 600})

# pick u,v
if "uo" in ds and "vo" in ds:
    u = ds["uo"]; v = ds["vo"]
elif "eastward_current_velocity" in ds and "northward_current_velocity" in ds:
    u = ds["eastward_current_velocity"]; v = ds["northward_current_velocity"]
else:
    raise KeyError("u/v not found.")

# pick time
if "time" in u.dims:
    u = u.isel(time=TIME_INDEX); v = v.isel(time=TIME_INDEX)

# surface
depth_dim = next((d for d in u.dims if d in DEPTH_NAMES), None)
if depth_dim is not None:
    idx0 = int(np.abs(u[depth_dim] - 0.0).argmin())
    u = u.isel({depth_dim: idx0}); v = v.isel({depth_dim: idx0})

# find current lon/lat names
lon_name, lat_name = find_lon_lat(u)

# ---- CROP to Atlantic early
u = crop_bbox(u, lon_name, lat_name, ATL_BBOX)
v = crop_bbox(v, lon_name, lat_name, ATL_BBOX)

# ---- ENSURE dims order and RENAME BOTH DIMS & COORDS to 'lat','lon'
# After this line, everything uses 'lat'/'lon' consistently.
if u.dims.index(lat_name) != 0 or u.dims.index(lon_name) != 1:
    u = u.transpose(lat_name, lon_name)
    v = v.transpose(lat_name, lon_name)
u = u.rename({lat_name: "lat", lon_name: "lon"})
v = v.rename({lat_name: "lat", lon_name: "lon"})
# if coord variables are still named 'latitude'/'longitude', rename them too:
if "latitude" in u.coords:  u = u.rename({"latitude": "lat"})
if "longitude" in u.coords: u = u.rename({"longitude": "lon"})
if "latitude" in v.coords:  v = v.rename({"latitude": "lat"})
if "longitude" in v.coords: v = v.rename({"longitude": "lon"})

lon = u["lon"]; lat = u["lat"]

# ============================
# EDDY-BASED SHARK INDEX_1
# ============================
# speed
speed = np.sqrt((u.astype("float32")**2) + (v.astype("float32")**2))

# vorticity & Okubo–Weiss (now coords are lat/lon -> no mismatch)
W, omega = okubo_weiss_and_vorticity(u, v, lon, lat)

# hemisphere-aware polarity: sign(omega * f)
phi = np.deg2rad(lat.values)
f_1d = (2 * OMEGA * np.sin(phi)).astype(np.float32)
f_2d = xr.DataArray(np.repeat(f_1d[:, None], lon.size, axis=1), coords=omega.coords, dims=omega.dims)
polarity = xr.apply_ufunc(np.sign, (omega * f_2d)).clip(-1, 1).astype("float32")

# rotation radius r ≈ 2|V|/|ζ|
radius_m  = (2.0 * speed / (np.abs(omega).astype("float32") + EPS)).where(np.isfinite(speed))
radius_km = (radius_m / 1000.0).astype("float32")

# strength: small radii -> 1, large -> 0  (memory-safe)
strength = strength_from_radius(radius_km)

# optional: keep only eddy cores
# strength = strength.where(W < 0)

# final index in [-1, 1]
shark_index_1 = (polarity * strength).astype("float32")

# ============================
# PLOT
# ============================
fig = plt.figure(figsize=(11, 6.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
im = shark_index_1.plot(
    ax=ax, x="lon", y="lat",
    transform=ccrs.PlateCarree(),
    cmap="coolwarm", vmin=-1, vmax=1,
    cbar_kwargs={"label": "Shark Index 1 (cyclonic + / anticyclonic −)"},
)
ax.coastlines(linewidth=0.6)
ax.set_extent([ATL_BBOX[0], ATL_BBOX[1], ATL_BBOX[2], ATL_BBOX[3]], crs=ccrs.PlateCarree())
ax.set_title("Shark Index_1 from Eddy Polarity & Rotation Radius (Atlantic)")
plt.tight_layout()
plt.show()

# ============================
# SAVE AS NUMPY ARRAY
# ============================
OUT_NPY = "./data/edd_index.npy"
os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)

# Save as NumPy .npy (only the index values)
np.save(OUT_NPY, shark_index_1.values.astype("float32"))
print(f"✅ Saved Shark Index_1 array → {OUT_NPY}")

# Optional: save coordinates too if you’ll need them for reloading
np.savez("./data/edd_index_with_coords.npz",
         lon=shark_index_1.lon.values,
         lat=shark_index_1.lat.values,
         index=shark_index_1.values.astype("float32"))
print("✅ Saved eddy index + lat/lon → ./data/edd_index_with_coords.npz")
