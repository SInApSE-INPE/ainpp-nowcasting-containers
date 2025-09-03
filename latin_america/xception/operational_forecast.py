# --- Existing imports ---
import ftplib
import os
from datetime import datetime, timedelta
import pathlib
import tempfile
import gzip
import numpy as np
from skimage.morphology import binary_closing, remove_small_objects, square
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch


from xception import (
    XceptionUNet,
    INPUT_SEQ_LEN,
    OUTPUT_SEQ_LEN,
    MODEL_INPUT_CHANNELS_PER_STEP,
    MODEL_OUTPUT_CHANNELS_PER_STEP,
    XCEPTION_INIT_FEATURES,
    XCEPTION_DEPTH,
    XCEPTION_BLOCKS_PER_STAGE,
)

# --- New imports for GeoJSON ---
import json
from affine import Affine
try:
    from rasterio.features import shapes as rasterio_shapes
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False

# ==============================
# GENERAL CONFIG
# ==============================
FTP_JAXA_HOST   = os.environ['FTP_JAXA_HOST']
user = os.environ['FTP_JAXA_USER']
password = os.environ['FTP_JAXA_PASSWORD']
BASE_DIR   = '/now/latest'
product = 'gsmap_now'
HORIZON = 6  # hours

# Global coordinates
glat_min, glat_max = -60.0, 60.0
glon_min, glon_max = -180.0, 180.0

# Latin America subdomain
rlat_min, rlat_max = -55.0, 33.0
rlon_min, rlon_max = -120.0, -23.0

NROW = 880
NCOL = 970

# --- Palette/Classes (kept from your plot) ---
PRECIP_BOUNDS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
PRECIP_COLORS = [
    '#9056b1',
    '#1a00db',
    '#0342a6',
    '#10b50e',
    '#b3e804',
    '#ffdd00',
    '#ff9900',
    '#ff2c00',
]
cmap_precip = mcolors.ListedColormap(PRECIP_COLORS)
cmap_precip.set_under('none')
cmap_precip.set_over('#831600')
norm_precip = mcolors.BoundaryNorm(PRECIP_BOUNDS, cmap_precip.N - 1, clip=False)

# --- Output folders (BASE without date) ---
ARCH = "xception"
OUTPUT_ROOT = os.environ['OUTPUT_ROOT']
FORECASTS_DIR_BASE = os.path.join(OUTPUT_ROOT, "forecasts")
GEOJSON_DIR_BASE   = os.path.join(OUTPUT_ROOT, "geometries")
PANELS_DIR_BASE    = os.path.join(OUTPUT_ROOT, "panels")

# --- Execution timestamp (use local time; for UTC, use datetime.now(timezone.utc)) ---
EXEC_DT = datetime.now()

def dated_subdir(base_dir: str, dt: datetime) -> str:
    """Build /base/YYYY/MM/DD"""
    return os.path.join(base_dir, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))

# --- Final dated folders for this run ---
FORECASTS_DIR = dated_subdir(FORECASTS_DIR_BASE, EXEC_DT)
GEOJSON_DIR   = dated_subdir(GEOJSON_DIR_BASE,   EXEC_DT)
PANELS_DIR    = dated_subdir(PANELS_DIR_BASE,    EXEC_DT)

# Ensure they exist
pathlib.Path(FORECASTS_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(GEOJSON_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(PANELS_DIR).mkdir(parents=True, exist_ok=True)

# Output pattern (.dat.gz) now inside the dated folder
OUT_PATTERN = f"{FORECASTS_DIR}/gsmap_{ARCH}.%Y%m%d.%H%M.v1.0600.dat.gz"

# ==============================
# Utility functions
# ==============================
def read_data(filepath):
    """Read global GSMaP NOW 0.1° grid, roll longitudes to [-180, 180], return 2D array (1200x3600)."""
    with gzip.open(filepath, mode='rb') as handle:
        data = np.frombuffer(handle.read(), dtype=np.float32).reshape(1200, 3600)
        data = np.roll(data, shift=1800, axis=1)
    return data

def crop_data(data, lat_min, lat_max, lon_min, lon_max):
    """Crop global grid to [lat_min, lat_max] x [lon_min, lon_max] using simple index math."""
    lat_resolution = (glat_max - glat_min) / data.shape[0]
    lat_idx_min = int((glat_max - lat_max) / lat_resolution)
    lat_idx_max = int((glat_max - lat_min) / lat_resolution)

    lon_resolution = (glon_max - glon_min) / data.shape[1]
    lon_idx_min = int((lon_min - glon_min) / lon_resolution)
    lon_idx_max = int((lon_max - glon_min) / lon_resolution)

    cropped_data = data[lat_idx_min:lat_idx_max, lon_idx_min:lon_idx_max]
    return cropped_data

def class_edges_from_bounds(bounds):
    """
    Build (min, max) tuples per class from PRECIP_BOUNDS,
    including the last open interval (>= last bound).
    """
    edges = []
    for i in range(1, len(bounds)):
        edges.append((bounds[i-1], bounds[i]))
    edges.append((bounds[-1], float("inf")))
    return edges

def build_affine_transform(lon_min, lon_max, lat_min, lat_max, ncol, nrow):
    """
    Create north-up Affine transform:
    x = lon_min + col*dx
    y = lat_max - row*dy
    """
    dx = (lon_max - lon_min) / ncol
    dy = (lat_max - lat_min) / nrow
    transform = Affine.translation(lon_min, lat_max) * Affine.scale(dx, -dy)
    return transform, dx, dy

def quantize_to_classes(prec2d, bounds):
    """
    Convert continuous precipitation to integer classes:
    0 = below the smallest threshold
    1..N = classes for each interval defined in bounds
    """
    extended = bounds + [float("inf")]
    classes = np.digitize(prec2d, extended, right=False)
    return classes.astype(np.int16)

def export_geojson_for_step(prec2d, run_dt, lead_dt, lon_min, lon_max, lat_min, lat_max,
                            ncol, nrow, bounds, out_dir):
    """
    Save one GeoJSON per forecast step. Prefer polygonization by class with rasterio.
    If rasterio is not available, fallback to point features at cell centroids where precip >= bounds[0].
    """
    transform, dx, dy = build_affine_transform(lon_min, lon_max, lat_min, lat_max, ncol, nrow)
    classes = quantize_to_classes(prec2d, bounds)
    class_ranges = class_edges_from_bounds(bounds)

    run_str  = run_dt.strftime("%Y%m%d%H%M")
    lead_str = lead_dt.strftime("%Y%m%d%H%M")
    out_fn = os.path.join(out_dir, f"gsmap_{ARCH}.{run_str}.{lead_str}.v1.0600.geojson")

    features = []
    if RASTERIO_AVAILABLE:
        # Polygons per class (ignore class 0 = below threshold)
        for geom, val in rasterio_shapes(classes, transform=transform):
            cls = int(val)
            if cls <= 0:
                continue
            idx = cls - 1
            vmin, vmax = class_ranges[idx]
            props = {
                "run_time": run_dt.strftime("%Y-%m-%d %H:%M"),
                "lead_time": lead_dt.strftime("%Y-%m-%d %H:%M"),
                "class": int(cls),
                "min_mmph": float(vmin),
                "max_mmph": (None if np.isinf(vmax) else float(vmax))
            }
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": props
            })
    else:
        # Fallback: points at centroids where precip >= bounds[0]
        thr = bounds[0]
        rr, cc = np.where(prec2d >= thr)
        for r, c in zip(rr, cc):
            lon = rlon_min + (c + 0.5) * ((lon_max - lon_min) / ncol)
            lat = rlat_max - (r + 0.5) * ((lat_max - lat_min) / nrow)
            val = float(prec2d[r, c])
            cls = int(quantize_to_classes(np.array([[val]]), bounds)[0, 0])
            if cls <= 0:
                continue
            idx = cls - 1
            vmin, vmax = class_ranges[idx]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "run_time": run_dt.strftime("%Y-%m-%d %H:%M"),
                    "lead_time": lead_dt.strftime("%Y-%m-%d %H:%M"),
                    "value_mmph": val,
                    "class": cls,
                    "min_mmph": float(vmin),
                    "max_mmph": (None if np.isinf(vmax) else float(vmax))
                }
            })

    fc = {
        "type": "FeatureCollection",
        "name": "gsmap-now_unet",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": features
    }

    with open(out_fn, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"[GeoJSON] saved: {out_fn}")

# ==============================
# DOWNLOAD/READ + INFERENCE
# ==============================
remote_dir = f"{BASE_DIR}"
with ftplib.FTP(FTP_JAXA_HOST) as ftp:
    ftp.login(user, password)
    ftp.cwd(remote_dir)
    files = []
    ftp.retrlines('LIST', lambda x: files.append(x.split()[-1]))
    files = sorted(set(filter(lambda x: x.startswith(product) and x.endswith('00.dat.gz'), files)))[-HORIZON:]
    input_data = []
    datetime_labels = []
    for file in files:
        datetime_label = datetime.strptime(file, f"{product}.%Y%m%d.%H00.dat.gz")
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = pathlib.Path(tmp_dir) / file
            with open(local_path, 'wb') as f:
                ftp.retrbinary(f"RETR {file}", f.write)
            print(f"Downloaded: {local_path}")
            data = read_data(local_path)
            data = crop_data(data, rlat_min, rlat_max, rlon_min, rlon_max)
            input_data.append(data)
            datetime_labels.append(datetime_label)
            print(f"Processed: {file} at {datetime_label}")
    input_data = np.array(input_data).astype(np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = os.environ['CHECKPOINT_PATH']

model = XceptionUNet(
    input_seq_len=INPUT_SEQ_LEN,
    output_seq_len=OUTPUT_SEQ_LEN,
    input_channels_per_step=MODEL_INPUT_CHANNELS_PER_STEP,
    output_channels_per_step=MODEL_OUTPUT_CHANNELS_PER_STEP,
    init_features=XCEPTION_INIT_FEATURES,
    depth=XCEPTION_DEPTH,
    blocks_per_stage=XCEPTION_BLOCKS_PER_STAGE,
)
state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)
model.to(device).eval()

INFER_START = datetime_labels[0] + timedelta(hours=HORIZON)

with torch.no_grad():
    print(f"Forecasting for {INFER_START}")
    # (batch=1, time=HORIZON, 1, H, W)
    input_data = np.expand_dims(input_data, axis=0)
    inp = torch.from_numpy(input_data).unsqueeze(2).to(device)
    out = model(inp)                         # (1, 6, 1, H, W)
    forecast = out.squeeze(0).cpu().numpy()  # (6, 1, H, W)
    forecast = forecast[:, 0, ...]           # (6, H, W)

    # Save binary array inside dated folder
    out_fn = INFER_START.strftime(OUT_PATTERN)
    with gzip.open(out_fn, 'wb') as gzout:
        gzout.write(forecast.astype(np.float32).tobytes())
    if device == "cuda":
        torch.cuda.empty_cache()

# ==============================
# 2x3 PANEL PLOT + GEOJSON
# ==============================
lats_1d = np.linspace(rlat_min, rlat_max, NROW)
lons_1d = np.linspace(rlon_min, rlon_max, NCOL)
lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)

N_STEPS = min(6, forecast.shape[0])
proj = ccrs.PlateCarree()
PANEL_BACKGROUND = os.environ.get('PANEL_BACKGROUND')
bg = mpimg.imread(PANEL_BACKGROUND)

def clean_precip(P, thr=0.1, min_pix=20, k=3):
    """Basic morphological cleanup to reduce speckle/noise."""
    mask = P >= thr
    mask = remove_small_objects(mask, min_size=min_pix)
    mask = binary_closing(mask, square(k))
    return np.where(mask, P, 0.0)

fig, axes = plt.subplots(
    2, 3, figsize=(16, 10),
    subplot_kw={'projection': proj},
    constrained_layout=False
)
axes = axes.ravel()

meshes = []
letters = [chr(i) for i in range(97, 103)]  # 'a'..'f'

for i in range(N_STEPS):
    ax = axes[i]
    lead_dt = INFER_START + timedelta(hours=i)
    ax.set_extent([rlon_min, rlon_max, rlat_min, rlat_max], crs=proj)

    # background
    ax.imshow(bg, origin='upper', transform=proj, extent=[-180, 180, -90, 90], zorder=0)

    # cartography and grid
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.6, color='gray', alpha=0.4, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': 'gray'}
    gl.ylabel_style = {'size': 9, 'color': 'gray'}

    # data (flip latitude to match mesh orientation)
    prec = forecast[i, ::-1, :]
    prec = clean_precip(prec, thr=0.1, min_pix=20, k=3)

    m = ax.pcolormesh(lons_2d, lats_2d, prec, cmap=cmap_precip, norm=norm_precip, shading='auto', zorder=2)
    meshes.append(m)

    # titles
    ax.set_title(f"+{i+1}h • {lead_dt.strftime('%Y-%m-%d %H:%M')}", fontweight='bold', fontsize=11)
    ax.set_title(f"({letters[i]})", loc='left', fontweight='bold', fontsize=11, color='gray', pad=10)

    # U-Net watermark
    ax.text(0.02, 0.02, f"{ARCH}", transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    # === EXPORT GEOJSON FOR THIS STEP ===
    export_geojson_for_step(
        prec2d=prec[::-1, :],  # revert to row->increasing lat to be consistent with transform (top = rlat_max)
        run_dt=INFER_START,
        lead_dt=lead_dt,
        lon_min=rlon_min, lon_max=rlon_max,
        lat_min=rlat_min, lat_max=rlat_max,
        ncol=NCOL, nrow=NROW,
        bounds=PRECIP_BOUNDS,
        out_dir=GEOJSON_DIR
    )

# any remaining empty panels
for j in range(N_STEPS, 6):
    axes[j].axis('off')

# single colorbar
cbar = fig.colorbar(meshes[-1], ax=axes, orientation='horizontal', fraction=0.04, pad=0.02, extend='both')
cbar.set_label("Precipitation (mm/h)", fontsize=14)

# move colorbar slightly downward
cbar.ax.set_position([
    cbar.ax.get_position().x0,
    cbar.ax.get_position().y0 - 0.05,
    cbar.ax.get_position().width,
    cbar.ax.get_position().height
])

plt.subplots_adjust(wspace=0.08, hspace=0.15)

# === SAVE PANEL INTO DATED FOLDER ===
panel_name = f"{PANELS_DIR}/panel_gsmap_{ARCH}.{INFER_START.strftime('%Y%m%d_%H%M')}.png"
plt.savefig(panel_name, dpi=300, bbox_inches='tight')
print(f"[Panel] saved: {panel_name}")
