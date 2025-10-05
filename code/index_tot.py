import os
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# ---------------- USER SETTINGS ----------------
# Input files (existing)
FP_CHL = "./data/atlantic_chl_norm_with_coords.npz"
FP_EDD = "./data/edd_index_with_coords.npz"
FP_SST = "./data/sst_4km_with_coords.npz"

# Output
FP_OUT = "atlantic_merged_index.npz"

# Interp & smoothing
GAUSS_SIGMA = 1.0   # in grid cells; set 0 to disable Gaussian smoothing per-layer
FINAL_DO_KNN3 = True  # apply 3-nearest spatial average to the composite index

# ------------------------------------------------

def load_npz_autokey(path):
    """Load an NPZ and try to guess the data key (besides lon/lat). Returns lon, lat, data."""
    d = np.load(path)
    keys = list(d.keys())
    # common coordinate names
    lon_key = next((k for k in keys if k.lower() in ("lon", "lons", "longitude")), None)
    lat_key = next((k for k in keys if k.lower() in ("lat", "lats", "latitude")), None)
    if lon_key is None or lat_key is None:
        # try more flexible guesses
        lon_key = next((k for k in keys if "lon" in k.lower()), None)
        lat_key = next((k for k in keys if "lat" in k.lower()), None)
    if lon_key is None or lat_key is None:
        raise ValueError(f"{path}: Could not find lon/lat keys. Found keys={keys}")

    # data key is anything not lon/lat
    data_keys = [k for k in keys if k not in (lon_key, lat_key)]
    if not data_keys:
        raise ValueError(f"{path}: No data variable found besides lon/lat. Keys={keys}")
    if len(data_keys) > 1:
        # pick the first nontrivial 2D array
        cand = [k for k in data_keys if np.ndim(d[k]) >= 2]
        data_key = cand[0] if cand else data_keys[0]
    else:
        data_key = data_keys[0]

    lon = d[lon_key]
    lat = d[lat_key]
    data = d[data_key]
    # Ensure 2D grid (lat x lon or lon x lat) → we’ll try to standardize to (lat, lon)
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    data = np.asarray(data)

    # If lon/lat are 1D and data is 2D, we assume meshgrid(lat, lon) style.
    if data.ndim == 2 and lon.ndim == 1 and lat.ndim == 1:
        # we don't need to build full mesh; just remember orientation
        # Try to infer orientation by matching shapes
        if data.shape == (lat.size, lon.size):
            orient = "latlon"
        elif data.shape == (lon.size, lat.size):
            orient = "lonlat"
            data = data.T  # transpose to (lat, lon)
        else:
            # fallback: assume (lat, lon)
            orient = "latlon"
    else:
        # If they already gave full lon/lat 2D grids matching data
        if data.shape == lon.shape == lat.shape:
            # We will flatten later for interpolation
            orient = "fullgrid"
        else:
            raise ValueError(f"{path}: Shapes not understood. "
                             f"lon.shape={lon.shape}, lat.shape={lat.shape}, data.shape={data.shape}")

    return lon, lat, data

def build_target_grid(lon_sst, lat_sst, data_sst, others):
    """
    Use SST grid if it's regular lat/lon 1D; otherwise infer a regular grid that spans all datasets.
    `others` is list of tuples (lon, lat, data).
    Returns lon2d, lat2d (2D meshgrid) and 1D lon/lat vectors as well.
    """
    # Case 1: SST provided as 1D lon/lat and 2D data
    if data_sst.ndim == 2 and lon_sst.ndim == 1 and lat_sst.ndim == 1 \
       and data_sst.shape == (lat_sst.size, lon_sst.size):
        lon_vec = lon_sst
        lat_vec = lat_sst
    else:
        # Build a grid from min/max across all datasets; infer step from median spacing if lon/lat 1D
        # Collect all lon/lat vectors if available
        lons_all = []
        lats_all = []
        for (lo, la, da) in [(lon_sst, lat_sst, data_sst)] + list(others):
            if da.ndim == 2 and lo.ndim == 1 and la.ndim == 1 and da.shape == (la.size, lo.size):
                lons_all.append(lo)
                lats_all.append(la)
        if lons_all and lats_all:
            # infer step as median of diffs from first one
            lon_step = np.median(np.diff(lons_all[0]))
            lat_step = np.median(np.diff(lats_all[0]))
        else:
            # default ~4km (0.04°) grid if nothing better
            lon_step = 0.04
            lat_step = 0.04

        # span across min/max from all arrays (using coarse estimates)
        def get_min_max(lo, la, da):
            if da.ndim == 2 and lo.ndim == 1 and la.ndim == 1 and da.shape == (la.size, lo.size):
                return lo.min(), lo.max(), la.min(), la.max()
            else:
                # if full grids, use mins/maxes
                return lo.min(), lo.max(), la.min(), la.max()

        lmins = []
        lmaxs = []
        bmins = []
        bmaxs = []
        for (lo, la, da) in [(lon_sst, lat_sst, data_sst)] + list(others):
            mnlo, mxlo, mnla, mxla = get_min_max(lo, la, da)
            lmins.append(mnlo); lmaxs.append(mxlo); bmins.append(mnla); bmaxs.append(mxla)

        lon_min, lon_max = float(np.min(lmins)), float(np.max(lmaxs))
        lat_min, lat_max = float(np.min(bmins)), float(np.max(bmaxs))

        lon_vec = np.arange(lon_min, lon_max + lon_step*0.5, lon_step)
        lat_vec = np.arange(lat_min, lat_max + lat_step*0.5, lat_step)

    lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)
    return lon2d, lat2d, lon_vec, lat_vec

def interp_to_grid(lon, lat, data, lon2d, lat2d):
    """
    Interpolate an input layer to the target grid (lon2d, lat2d).
    Supports either (lat,lon) 2D data with lon,lat 1D OR full 2D lon/lat.
    Returns 2D array on target grid.
    """
    # Prepare scattered points and values
    if data.ndim == 2 and lon.ndim == 1 and lat.ndim == 1 and data.shape == (lat.size, lon.size):
        # build mesh for source
        src_lon2d, src_lat2d = np.meshgrid(lon, lat)
    else:
        # already full grids matching data
        src_lon2d, src_lat2d = lon, lat

    pts = np.column_stack([src_lon2d.ravel(), src_lat2d.ravel()])
    vals = data.ravel()

    # mask out NaNs
    m = np.isfinite(vals)
    pts = pts[m]
    vals = vals[m]

    # First linear interpolation
    out = griddata(pts, vals, (lon2d, lat2d), method="linear")
    # Fill holes with nearest
    if np.isnan(out).any():
        out_near = griddata(pts, vals, (lon2d, lat2d), method="nearest")
        out = np.where(np.isnan(out), out_near, out)
    return out

def knn3_average(index2d, lon2d, lat2d):
    """For each grid cell, replace value by the average of its 3 nearest **spatial** neighbors (including itself)."""
    ny, nx = index2d.shape
    # Flatten valid points
    vals = index2d.ravel()
    valid = np.isfinite(vals)
    xy = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    xy_valid = xy[valid]
    vals_valid = vals[valid]

    tree = cKDTree(xy_valid)
    # Query 3 nearest for all nodes (use valid set so we don't average NaNs)
    d, idx = tree.query(xy, k=min(3, xy_valid.shape[0]))
    # idx shape: (N,) if k==1 else (N, k)
    if idx.ndim == 1:
        avg = vals_valid[idx]
    else:
        avg = np.mean(vals_valid[idx], axis=1)

    out = np.full_like(vals, np.nan, dtype=float)
    out[:] = avg
    return out.reshape(index2d.shape)

def main():
    # Load layers
    lon_chl, lat_chl, chl = load_npz_autokey(FP_CHL)
    lon_edd, lat_edd, edd = load_npz_autokey(FP_EDD)
    lon_sst, lat_sst, sst = load_npz_autokey(FP_SST)

    # Build target grid (prefer SST grid if regular)
    lon2d, lat2d, lon_vec, lat_vec = build_target_grid(lon_sst, lat_sst, sst,
                                                       others=[(lon_chl, lat_chl, chl),
                                                               (lon_edd, lat_edd, edd)])

    # Interpolate
    chl_g = interp_to_grid(lon_chl, lat_chl, chl, lon2d, lat2d)
    edd_g = interp_to_grid(lon_edd, lat_edd, edd, lon2d, lat2d)
    sst_g = interp_to_grid(lon_sst, lat_sst, sst, lon2d, lat2d)

    # Optional Gaussian smoothing per-layer
    if GAUSS_SIGMA and GAUSS_SIGMA > 0:
        # Only smooth finite portions; keep NaNs where all invalid
        def smooth_nan(a):
            m = np.isfinite(a)
            if not m.any():
                return a
            # Fill NaNs with local mean to avoid bleeding zeros
            fill = a.copy()
            fill[~m] = np.nanmean(a[m])
            sm = gaussian_filter(fill, sigma=GAUSS_SIGMA, mode="nearest")
            # Reapply NaNs where there was no original data absolutely anywhere (rare)
            return np.where(m, sm, np.nan)
        chl_g = smooth_nan(chl_g)
        edd_g = smooth_nan(edd_g)
        sst_g = smooth_nan(sst_g)

    # Composite index: simple mean of the 3 layers at each gridpoint
    # (You can change to weighted mean if needed)
    stack = np.stack([chl_g, edd_g, sst_g], axis=0)
    with np.errstate(invalid='ignore'):
        index_raw = np.nanmean(stack, axis=0)

    # Final KNN-3 spatial average (what you asked: average of 3 nearest point data)
    if FINAL_DO_KNN3:
        index_knn3 = knn3_average(index_raw, lon2d, lat2d)
    else:
        index_knn3 = index_raw.copy()

    # Save
    np.savez_compressed(
        FP_OUT,
        lon=lon_vec,
        lat=lat_vec,
        chl=chl_g,
        edd=edd_g,
        sst=sst_g,
        index_raw=index_raw,
        index_knn3=index_knn3
    )
    print(f"Done. Saved: {FP_OUT}")
    print("Variables: lon (1D), lat (1D), chl (2D), edd (2D), sst (2D), index_raw (2D), index_knn3 (2D)")

if __name__ == "__main__":
    main()
