import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# ============================
# CONFIG
# ============================
DATE = "2025-08-30"
ATL_BBOX = (-100, 20, -60, 70)  # lon_min, lon_max, lat_min, lat_max

MOANA_NC = "./data/PACE_OCI.20250830.L4m.DAY.MOANA.V3_1.4km.nc"
MOANA_VAR = "picoeuk_moana"

SST_ON_MOANA_NC = "./data/sst_on_moana_4km_2025-08-30.nc"  # from your previous step
CURR_NC  = "./data/dataset-uv-nrt-daily_20250830T1200Z_P20250905T0000.nc"

DEPTH_NAMES = ("depth", "depthu", "depthv", "deptho", "lev", "z")

OUT_NC  = "./data/index_atlantic_eddy_2025-08-30.nc"
OUT_PDF = "./data/index_atlantic_eddy_2025-08-30.pdf"

warnings.filterwarnings(
    "ignore",
    message=".*will not decode the variable 'dt_1km_data'.*",
    category=FutureWarning
)

R_EARTH = 6371000.0  # meters

# ============================
# HELPERS
# ============================
def normalize_lon_vals(vals):
    vals = np.asarray(vals)
    return ((vals + 180) % 360) - 180

def normalize_lon_coords(ds, lon_name="lon"):
    if lon_name in ds.coords:
        ds = ds.assign_coords({lon_name: normalize_lon_vals(ds[lon_name].values)}).sortby(lon_name)
    return ds

def find_lon_lat(da_or_ds):
    names = list(getattr(da_or_ds, "coords", {})) + list(getattr(da_or_ds, "dims", ()))
    lon = next((n for n in names if "lon" in n.lower()), None)
    lat = next((n for n in names if "lat" in n.lower()), None)
    if lon is None or lat is None:
        raise KeyError("Could not find lon/lat names.")
    return lon, lat

def crop_bbox(ds_or_da, lon_name="lon", lat_name="lat", bbox=ATL_BBOX):
    lo1, lo2, la1, la2 = bbox
    if float(ds_or_da[lon_name].max()) > 180:
        ds_or_da = normalize_lon_coords(ds_or_da, lon_name)
    return ds_or_da.sel({lon_name: slice(lo1, lo2), lat_name: slice(la1, la2)})

def robust_minmax_norm(da, qlo=0.05, qhi=0.95):
    lo = float(da.quantile(qlo))
    hi = float(da.quantile(qhi))
    if np.isclose(hi, lo):
        return xr.zeros_like(da)
    return ((da - lo) / (hi - lo)).clip(0, 1)

def safe_overlap_interp(src_da, tgt_lon, tgt_lat, lon_name, lat_name, primary="linear"):
    """
    Interpolate src_da to the exact target lon/lat, but **never** crash:
      - normalize and sort
      - drop all-NaN bands
      - compute src/target overlap; if none -> return all-NaN on target grid
      - if overlap <2 in any dim and method==linear -> switch to nearest
    """
    # ensure 1D coords increasing
    src_da = src_da.sortby(lat_name).sortby(lon_name)

    # drop all-NaN rows/cols to avoid empty reductions
    src_da = src_da.where(np.isfinite(src_da))
    if lat_name in src_da.dims:
        src_da = src_da.dropna(dim=lat_name, how="all")
    if lon_name in src_da.dims:
        src_da = src_da.dropna(dim=lon_name, how="all")

    # if empty after drop, return NaNs on target
    if src_da.sizes.get(lon_name, 0) == 0 or src_da.sizes.get(lat_name, 0) == 0:
        return xr.DataArray(
            np.full((tgt_lat.size, tgt_lon.size), np.nan, dtype=np.float32),
            coords={"lat": tgt_lat, "lon": tgt_lon},
            dims=("lat", "lon")
        )

    # compute overlap window with target (assuming both in [-180,180])
    src_lon = src_da[lon_name].values
    src_lat = src_da[lat_name].values
    lon_min =-100 #max(np.nanmin(src_lon), float(tgt_lon.min()))
    lon_max = 20#min(np.nanmax(src_lon), float(tgt_lon.max()))
    lat_min = -60#max(np.nanmin(src_lat), float(tgt_lat.min()))
    lat_max = 70#min(np.nanmax(src_lat), float(tgt_lat.max()))

    -100, 20, -60, 70
    # no overlap → return NaNs
    if not (lon_min < lon_max and lat_min < lat_max):
        return xr.DataArray(
            np.full((tgt_lat.size, tgt_lon.size), np.nan, dtype=np.float32),
            coords={"lat": tgt_lat, "lon": tgt_lon},
            dims=("lat", "lon")
        )

    # subset to overlap (slight pad to avoid edge effects)
    eps_lon = (lon_max - lon_min) * 1e-6 + 1e-9
    eps_lat = (lat_max - lat_min) * 1e-6 + 1e-9
    src_sub = src_da.sel(
        {lon_name: slice(lon_min - eps_lon, lon_max + eps_lon),
         lat_name: slice(lat_min - eps_lat, lat_max + eps_lat)}
    )

    # if too thin for linear, force nearest
    method = primary
    if src_sub.sizes.get(lon_name, 0) < 2 or src_sub.sizes.get(lat_name, 0) < 2:
        method = "nearest"

    # finally interpolate; if everything NaN, return NaNs on target
    try:
        out = src_sub.interp({lon_name: tgt_lon, lat_name: tgt_lat}, method=method)
        if np.isnan(out).all():
            return xr.DataArray(
                np.full((tgt_lat.size, tgt_lon.size), np.nan, dtype=np.float32),
                coords={"lat": tgt_lat, "lon": tgt_lon},
                dims=("lat", "lon")
            )
    except Exception:
        return xr.DataArray(
            np.full((tgt_lat.size, tgt_lon.size), np.nan, dtype=np.float32),
            coords={"lat": tgt_lat, "lon": tgt_lon},
            dims=("lat", "lon")
        )

    # ensure dims are (lat, lon)
    if out.dims != ("lat", "lon"):
        out = out.transpose("lat", "lon")
    return out

def okubo_weiss(u, v, lon, lat):
    """
    Okubo–Weiss parameter W = s_n^2 + s_s^2 - ω^2
    central differences on geographic grid in meters.
    """
    lat_rad = np.deg2rad(lat.values)
    dphi = np.deg2rad(np.gradient(lat.values))             # 1D
    dlam = np.deg2rad(np.gradient(lon.values))             # 1D

    dy = R_EARTH * dphi                                    # meters (lat)
    dx = R_EARTH * np.cos(lat_rad)[:, None] * dlam[None,:] # meters (2D)

    U = u.values; V = v.values
    dudy = np.full_like(U, np.nan); dvdy = np.full_like(U, np.nan)
    dudx = np.full_like(U, np.nan); dvdx = np.full_like(U, np.nan)

    i = slice(1, -1); j = slice(1, -1)
    dudy[i,:] = (U[2:,:] - U[:-2,:]) / (2.0 * dy[1:-1, None])
    dvdy[i,:] = (V[2:,:] - V[:-2,:]) / (2.0 * dy[1:-1, None])
    dudx[:,j] = (U[:,2:] - U[:,:-2]) / (2.0 * dx[:,1:-1])
    dvdx[:,j] = (V[:,2:] - V[:,:-2]) / (2.0 * dx[:,1:-1])

    s_n = dudx - dvdy
    s_s = dvdx + dudy
    omega = dvdx - dudy
    W = s_n**2 + s_s**2 - omega**2

    return xr.DataArray(W, coords={"lat": lat, "lon": lon}, dims=("lat","lon")), \
           xr.DataArray(omega, coords={"lat": lat, "lon": lon}, dims=("lat","lon"))

# ============================
# 1) MOANA (ref grid) & SST-on-MOANA
# ============================
mo = xr.open_dataset(MOANA_NC)
if "longitude" in mo: mo = mo.rename({"longitude": "lon"})
if "latitude"  in mo: mo = mo.rename({"latitude": "lat"})
mo = normalize_lon_coords(mo, "lon")
mo = crop_bbox(mo, "lon", "lat", ATL_BBOX)

if MOANA_VAR not in mo:
    raise KeyError(f"{MOANA_VAR} not in MOANA file.")
mo_var = mo[MOANA_VAR].where(mo[MOANA_VAR] > 0)

# load SST already on MOANA grid, then crop to Atlantic and align
sst_on_mo = xr.open_dataset(SST_ON_MOANA_NC)["sst"]
sst_on_mo = crop_bbox(sst_on_mo.to_dataset(name="sst"), "lon", "lat", ATL_BBOX)["sst"]
if not (np.array_equal(sst_on_mo.lon, mo.lon) and np.array_equal(sst_on_mo.lat, mo.lat)):
    sst_on_mo = sst_on_mo.interp(lon=mo.lon, lat=mo.lat)

# ============================
# 2) Currents → eddy index (Okubo–Weiss)
# ============================
cu = xr.open_dataset(CURR_NC, chunks={"latitude": 600, "longitude": 600})
if "uo" in cu and "vo" in cu:
    u = cu["uo"]; v = cu["vo"]
elif "eastward_current_velocity" in cu and "northward_current_velocity" in cu:
    u = cu["eastward_current_velocity"]; v = cu["northward_current_velocity"]
else:
    raise KeyError("Couldn't find u/v in currents file.")

ulon, ulat = find_lon_lat(u)
cu = normalize_lon_coords(cu, ulon)
u = cu[u.name]; v = cu[v.name]

# crop early to Atlantic
u = crop_bbox(u, ulon, ulat, ATL_BBOX)
v = crop_bbox(v, ulon, ulat, ATL_BBOX)

# choose time near DATE
if "time" in u.dims:
    u = u.sel(time=DATE, method="nearest")
    v = v.sel(time=DATE, method="nearest")

# choose surface
depth_dim = next((d for d in u.dims if d in DEPTH_NAMES), None)
if depth_dim is not None:
    idx0 = int(np.abs(u[depth_dim] - 0.0).argmin())
    u = u.isel({depth_dim: idx0})
    v = v.isel({depth_dim: idx0})

# Interp u,v -> exact MOANA grid (safe)
u_on_mo = safe_overlap_interp(u, mo.lon, mo.lat, ulon, ulat, primary="linear")
v_on_mo = safe_overlap_interp(v, mo.lon, mo.lat, ulon, ulat, primary="linear")

# If both came back NaN (no overlap), stop early with a clear message
if np.isnan(u_on_mo).all() or np.isnan(v_on_mo).all():
    raise RuntimeError("Currents subset has no usable overlap with MOANA Atlantic grid. "
                       "Check ATL_BBOX and DATE coverage.")

# Compute Okubo–Weiss and vorticity
W, omega = okubo_weiss(u_on_mo, v_on_mo, mo.lon, mo.lat)
eddy_raw = xr.where(W < 0, -W, 0.0)  # eddy signal

# ============================
# 3) Strict mask & normalization
# ============================
mask = (mo_var.notnull()) & (sst_on_mo.notnull()) & (np.isfinite(eddy_raw))

mo_n   = robust_minmax_norm(mo_var.where(mask))
sst_n  = robust_minmax_norm(sst_on_mo.where(mask))
eddy_n = robust_minmax_norm(eddy_raw.where(mask))

index = ((mo_n + sst_n + eddy_n) / 3.0).where(mask)

# ============================
# 4) Visualize & Save
# ============================
fig = plt.figure(figsize=(10.5, 6.5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
h = index.plot(
    ax=ax, x="lon", y="lat",
    transform=ccrs.PlateCarree(),
    cmap="plasma", vmin=0, vmax=1,
    cbar_kwargs={"label": "Composite Index (MOANA + SST + Eddy; 0–1)"},
)
ax.coastlines(linewidth=0.6)
ax.set_extent([ATL_BBOX[0], ATL_BBOX[1], ATL_BBOX[2], ATL_BBOX[3]], crs=ccrs.PlateCarree())
ax.set_title(f"Atlantic Composite Index with Eddy Signal — {DATE}")
plt.tight_layout()

os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
xr.Dataset(
    {
        "index": index.astype("float32"),
        "moana_norm": mo_n.astype("float32"),
        "sst_norm": sst_n.astype("float32"),
        "eddy_norm": eddy_n.astype("float32"),
        "eddy_raw": eddy_raw.astype("float32"),
        "okubo_weiss": W.astype("float32"),
        "vorticity": omega.astype("float32"),
        "mask": mask.astype("uint8"),
    },
    coords={"lon": mo.lon, "lat": mo.lat}
).to_netcdf(OUT_NC)

plt.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
print(f"✅ Saved NetCDF: {OUT_NC}")
print(f"✅ Saved PDF:    {OUT_PDF}")
plt.show()
