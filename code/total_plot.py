# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # =============== SETTINGS ===============
# FP_IN   = "atlantic_merged_index.npz"
# SAVE_PNGS = True
# PNG_DIR   = "plots_atlantic"
# # Atlantic view (adjust if needed)
# EXTENT = (-100, 20, -60, 60)  # (lon_min, lon_max, lat_min, lat_max)

# # Robust color scaling: use percentiles to avoid outliers
# ROBUST_LOW_PCT  = 2
# ROBUST_HIGH_PCT = 98

# # Colormaps per layer (change if you prefer)
# CMAP_CHL   = "viridis"
# CMAP_EDD   = "plasma"
# CMAP_SST   = "coolwarm"
# CMAP_INDEX = "magma"

# # ========================================

# def robust_vmin_vmax(a, low=2, high=98):
#     a = a[np.isfinite(a)]
#     if a.size == 0:
#         return (0.0, 1.0)
#     return (np.percentile(a, low), np.percentile(a, high))

# def meshgrid_from_1d(lon, lat):
#     # NPZ saved lon, lat as 1D vectors
#     lon2d, lat2d = np.meshgrid(lon, lat)
#     return lon2d, lat2d

# def add_geo(ax, title=None):
#     ax.coastlines(linewidth=0.6)
#     ax.add_feature(cfeature.LAND, facecolor="0.9")
#     ax.gridlines(draw_labels=True, linewidth=0.2, x_inline=False, y_inline=False)
#     if title:
#         ax.set_title(title, fontsize=11, pad=8)

# def imshow_ll(ax, lon, lat, data, cmap, vmin=None, vmax=None):
#     # Use pcolormesh for curvilinear-friendly plotting
#     lon2d, lat2d = np.meshgrid(lon, lat)
#     h = ax.pcolormesh(lon2d, lat2d, data, cmap=cmap, vmin=vmin, vmax=vmax,
#                       transform=ccrs.PlateCarree(), shading="auto")
#     return h

# def main():
#     d = np.load(FP_IN)
#     lon = d["lon"]           # 1D
#     lat = d["lat"]           # 1D
#     chl = d["chl"]           # 2D
#     edd = d["edd"]           # 2D
#     sst = d["sst"]           # 2D
#     idx_raw  = d["index_raw"]
#     idx_knn3 = d["index_knn3"]

#     if SAVE_PNGS and not os.path.isdir(PNG_DIR):
#         os.makedirs(PNG_DIR, exist_ok=True)

#     # Precompute robust ranges
#     chl_vmin, chl_vmax = robust_vmin_vmax(chl, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)
#     edd_vmin, edd_vmax = robust_vmin_vmax(edd, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)
#     sst_vmin, sst_vmax = robust_vmin_vmax(sst, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)
#     idx_vmin, idx_vmax = robust_vmin_vmax(idx_raw, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)
#     idxk_vmin, idxk_vmax = robust_vmin_vmax(idx_knn3, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)

#     layers = [
#         ("Chl (interp/smoothed)", chl, CMAP_CHL, chl_vmin, chl_vmax, "chl"),
#         ("Eddy Index (interp/smoothed)", edd, CMAP_EDD, edd_vmin, edd_vmax, "edd"),
#         ("SST (interp/smoothed)", sst, CMAP_SST, sst_vmin, sst_vmax, "sst"),
#         ("Composite Index (raw mean)", idx_raw, CMAP_INDEX, idx_vmin, idx_vmax, "index_raw"),
#         ("Composite Index (KNN-3)", idx_knn3, CMAP_INDEX, idxk_vmin, idxk_vmax, "index_knn3"),
#     ]

#     # ---- Figure: 2x3 panel ----
#     proj = ccrs.PlateCarree()
#     fig = plt.figure(figsize=(14, 8))
#     axes = []
#     for i in range(6):
#         ax = plt.subplot(2, 3, i+1, projection=proj)
#         ax.set_extent(EXTENT, crs=proj)
#         axes.append(ax)

#     for ax, (title, data, cmap, vmin, vmax, tag) in zip(axes[:5], layers):
#         add_geo(ax, title)
#         h = imshow_ll(ax, lon, lat, data, cmap, vmin, vmax)
#         cb = plt.colorbar(h, ax=ax, shrink=0.8, pad=0.03)
#         cb.ax.tick_params(labelsize=8)

#     # Legend/info panel
#     ax_legend = axes[5]
#     ax_legend.axis("off")
#     text = (
#         "Atlantic Merge Overview\n"
#         "• Inputs: chl, eddy, sst → interpolated to common grid\n"
#         "• Per-layer optional smoothing\n"
#         "• Composite index = mean(chl, eddy, sst)\n"
#         "• KNN-3 = spatial average of 3 nearest points"
#     )
#     ax_legend.text(0.02, 0.95, text, va="top", ha="left", fontsize=11)
#     ax_legend.text(0.02, 0.10, f"Lon range: {lon.min():.2f} .. {lon.max():.2f}\n"
#                                 f"Lat range: {lat.min():.2f} .. {lat.max():.2f}",
#                    va="top", ha="left", fontsize=10)

#     fig.suptitle("Atlantic Merged Layers & Composite Index", fontsize=14)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if SAVE_PNGS:
#         panel_path = os.path.join(PNG_DIR, "atlantic_merged_panel.png")
#         fig.savefig(panel_path, dpi=200)
#         print(f"Saved: {panel_path}")

#     # ---- Also save individual full-size maps (optional) ----
#     if SAVE_PNGS:
#         for title, data, cmap, vmin, vmax, tag in layers:
#             f = plt.figure(figsize=(8, 6))
#             ax = plt.axes(projection=proj)
#             ax.set_extent(EXTENT, crs=proj)
#             add_geo(ax, title)
#             h = imshow_ll(ax, lon, lat, data, cmap, vmin, vmax)
#             cb = plt.colorbar(h, ax=ax, shrink=0.8, pad=0.03)
#             cb.ax.tick_params(labelsize=8)
#             out = os.path.join(PNG_DIR, f"{tag}.png")
#             f.tight_layout()
#             f.savefig(out, dpi=200)
#             plt.close(f)
#             print(f"Saved: {out}")

#     plt.show()

# if __name__ == "__main__":
#     main()


import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ====== Settings ======
FP_IN     = "atlantic_merged_index.npz"
SAVE_PNG  = True
OUT_PNG   = "index_knn3_atlantic.png"

# Atlantic extent (lon_min, lon_max, lat_min, lat_max) — tweak if needed
EXTENT = (-100, 20, -60, 60)

# Robust color scaling to avoid outliers
ROBUST_LOW_PCT  = 2
ROBUST_HIGH_PCT = 98

# Colormap for the index
CMAP_INDEX = "magma_r"
# ======================

def normalize_to_01(a):
    """Normalize array to 0–1 ignoring NaNs."""
    valid = np.isfinite(a)
    if not valid.any():
        return a
    amin, amax = np.nanmin(a), np.nanmax(a)
    if amax == amin:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)

def robust_vmin_vmax(a, low=2, high=98):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    return (np.percentile(a, low), np.percentile(a, high))

def main():
    d = np.load(FP_IN)
    lon = d["lon"]           # 1D
    lat = d["lat"]           # 1D
    idx_knn3 = d["index_knn3"]  # 2D

    vmin, vmax = robust_vmin_vmax(idx_knn3, ROBUST_LOW_PCT, ROBUST_HIGH_PCT)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.5, 7))
    ax = plt.axes(projection=proj)
    ax.set_extent(EXTENT, crs=proj)

    # Base features
    ax.coastlines(linewidth=0.7)
    ax.gridlines(draw_labels=True, linewidth=0.3, x_inline=False, y_inline=False)

    # # Plot data
    # lon2d, lat2d = np.meshgrid(lon, lat)
    # im = ax.pcolormesh(
    #     lon2d, lat2d, idx_knn3,
    #     cmap=CMAP_INDEX, vmin=vmin, vmax=vmax,
    #     transform=proj, shading="auto", zorder=1
    # )

        # Plot data

    # --- Normalize to [0, 1] ---
    idx_norm = normalize_to_01(idx_knn3)

    lon2d, lat2d = np.meshgrid(lon, lat)
    im = ax.pcolormesh(
        lon2d, lat2d, idx_norm,
        cmap=CMAP_INDEX, vmin=0, vmax=1,
        transform=proj, shading="auto", zorder=1
    )

    # ---- Land "mask": paint land in solid white over the plot ----
    # (Covers any data over land without needing a separate land/sea mask raster)
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=5)

    # Optional: make coastlines stand out on top
    ax.coastlines(linewidth=0.7, zorder=6)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.ax.tick_params(labelsize=9)
    ax.set_title("Composite Index (KNN-3) — Atlantic", fontsize=12, pad=8)

    fig.tight_layout()

    if SAVE_PNG:
        fig.savefig(OUT_PNG, dpi=220)
        print(f"Saved: {OUT_PNG}")

    plt.show()

if __name__ == "__main__":
    main()
