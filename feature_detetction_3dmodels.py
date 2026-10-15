"""
3D Feature Detection Comparison
-------------------------------
Compares MSER, SIFT, FAST, Harris, and ORB on the same rendered views
of a selected 3D model.

Outputs:
- car_mser_keypoints.glb, car_sift_keypoints.glb, ...
- car_all_algorithms_keypoints.glb
- per_view_metrics.csv
- summary_metrics.csv
- keypoint_coordinates.csv

Important:
These are 2D detections on rendered images projected into 3D using the
renderer's depth buffer. They are not native 3D feature detectors, and
keypoint count alone does not measure accuracy.
"""

import os
import csv
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import trimesh
import pyrender
import csv
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
import psutil
import math
from sklearn.neighbors import NearestNeighbors


# ============================================================
# SETTINGS — keep these fixed for reproducible comparisons
# ============================================================

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
FOCAL_LENGTH = 700
NUMBER_OF_VIEWS = 12

# Same maximum number of points used for 3D projection per algorithm/view.
# Raw detector counts are also recorded BEFORE this limit is applied.
MAX_POINTS_PER_VIEW = 250

# Same final visualization cap and same 3D deduplication distance for all.
MAX_UNIQUE_POINTS_PER_ALGORITHM = 700
DEDUPLICATION_DISTANCE = 0.012
MARKER_SIZE_RATIO = 0.008

ALGORITHMS = ["MSER", "SIFT", "FAST", "HARRIS", "ORB"]

# RGB marker colours
COLORS = {
    "MSER":   [255, 60, 60],
    "SIFT":   [60, 255, 80],
    "FAST":   [50, 140, 255],
    "HARRIS": [255, 220, 40],
    "ORB":    [220, 60, 255],
}


# ============================================================
# SELECT MODEL
# ============================================================

root = tk.Tk()
root.withdraw()

model_path = filedialog.askopenfilename(
    title="Select your 3D model",
    filetypes=[
        ("3D models", "*.obj *.glb *.gltf *.stl *.ply"),
        ("All files", "*.*")
    ]
)

if not model_path:
    messagebox.showinfo("Cancelled", "No model was selected.")
    root.destroy()
    raise SystemExit

output_folder = os.path.join(
    os.path.dirname(model_path),
    "feature_detection_analysis"
)
os.makedirs(output_folder, exist_ok=True)


# ============================================================
# LOAD AND NORMALIZE MODEL
# ============================================================

print("Loading model:", model_path)
loaded = trimesh.load(model_path, force="scene")
mesh = loaded.dump(concatenate=True)

if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
    root.destroy()
    raise RuntimeError("Could not read mesh geometry from this file.")

bounds = mesh.bounds
center = (bounds[0] + bounds[1]) / 2.0
mesh.apply_translation(-center)

largest_dimension = float(np.max(mesh.extents))
if largest_dimension <= 0:
    root.destroy()
    raise RuntimeError("The model has invalid dimensions.")

# Normalize largest dimension to 2 model units.
mesh.apply_scale(2.0 / largest_dimension)
mesh.fix_normals()

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))


# ============================================================
# PREPARE RENDERER
# ============================================================

render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
render_scene = pyrender.Scene(
    bg_color=[245, 245, 245, 255],
    ambient_light=[0.35, 0.35, 0.35]
)
render_scene.add(render_mesh, name="model")

camera = pyrender.IntrinsicsCamera(
    fx=FOCAL_LENGTH,
    fy=FOCAL_LENGTH,
    cx=IMAGE_WIDTH / 2.0,
    cy=IMAGE_HEIGHT / 2.0,
    znear=0.01,
    zfar=100.0
)
camera_node = render_scene.add(camera, pose=np.eye(4))

light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
light_node = render_scene.add(light, pose=np.eye(4))

renderer = pyrender.OffscreenRenderer(
    viewport_width=IMAGE_WIDTH,
    viewport_height=IMAGE_HEIGHT
)


# ============================================================
# CAMERA HELPERS
# ============================================================

def camera_pose(eye, target=np.zeros(3)):
    """Create a camera-to-world pose. Camera looks along local -Z."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("Camera eye and target cannot be identical.")
    forward /= norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def render_view(view_index):
    """Render a view and return RGB image, depth map, and camera pose."""
    radius = 4.5
    azimuth = 2.0 * np.pi * view_index / NUMBER_OF_VIEWS
    elevation = 0.25 * np.sin(azimuth * 2.0) + 0.12

    eye = np.array([
        radius * np.cos(azimuth) * np.cos(elevation),
        radius * np.sin(azimuth) * np.cos(elevation),
        radius * np.sin(elevation)
    ])

    pose = camera_pose(eye)
    render_scene.set_pose(camera_node, pose)
    render_scene.set_pose(light_node, pose)
    color, depth = renderer.render(render_scene)
    return color, depth, pose


# ============================================================
# DETECTORS
# ============================================================

def detect_mser(gray):
    """
    MSER regions are represented by their centroids for visualization.
    Parameters are explicitly set and recorded; they are not guaranteed
    to be optimal for every model.
    """
    detector = cv2.MSER_create(
        5,       # delta
        30,      # min_area
        14400,   # max_area
        0.25,    # max_variation
        0.20     # min_diversity
    )

    regions, _ = detector.detectRegions(gray)
    keypoints = []

    for region in regions:
        if len(region) == 0:
            continue

        x, y = np.mean(region, axis=0)
        area = float(len(region))

        # Store region area as response for deterministic sorting.
        keypoints.append(
            cv2.KeyPoint(
                float(x), float(y),
                float(max(3.0, np.sqrt(area))),
                -1, area
            )
        )

    return keypoints


def detect_harris(gray):
    """
    Harris corner response followed by local-maximum selection.
    This avoids treating every adjacent above-threshold pixel as
    a separate corner.
    """
    gray_float = np.float32(gray)
    response = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    if response.size == 0 or float(response.max()) <= 0:
        return []

    threshold = 0.01 * float(response.max())
    dilated = cv2.dilate(response, None)

    maxima_mask = (response == dilated) & (response > threshold)
    ys, xs = np.where(maxima_mask)

    keypoints = [
        cv2.KeyPoint(
            float(x), float(y), 5.0, -1,
            float(response[y, x])
        )
        for y, x in zip(ys, xs)
    ]
    return keypoints


def detect_all(gray):
    """
    All algorithms receive the exact same grayscale image.
    No algorithm gets special contrast enhancement.
    Returns raw keypoints and elapsed detector time in milliseconds.
    """
    results = {}
    times_ms = {}

    start = time.perf_counter()
    results["MSER"] = detect_mser(gray)
    times_ms["MSER"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    # nfeatures=0 asks SIFT to return all detected features.
    sift = cv2.SIFT_create(nfeatures=0)
    results["SIFT"] = sift.detect(gray, None) or []
    times_ms["SIFT"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    fast = cv2.FastFeatureDetector_create(
        threshold=20,
        nonmaxSuppression=True
    )
    results["FAST"] = fast.detect(gray, None) or []
    times_ms["FAST"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    results["HARRIS"] = detect_harris(gray)
    times_ms["HARRIS"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    # Use a high internal limit; apply the shared cap below.
    orb = cv2.ORB_create(nfeatures=10000)
    results["ORB"] = orb.detect(gray, None) or []
    times_ms["ORB"] = (time.perf_counter() - start) * 1000.0

    # Sort deterministically by response, then apply the SAME cap.
    # Raw counts are retained separately for analysis.
    for name in ALGORITHMS:
        results[name] = sorted(
            results[name],
            key=lambda kp: kp.response,
            reverse=True
        )

    return results, times_ms


# ============================================================
# PROJECT IMAGE PIXELS TO 3D USING THE DEPTH BUFFER
# ============================================================

def back_project(u, v, depth, pose):
    """Convert a rendered image pixel and its depth to world coordinates."""
    x = int(round(u))
    y = int(round(v))

    if x < 0 or x >= IMAGE_WIDTH or y < 0 or y >= IMAGE_HEIGHT:
        return None

    z = float(depth[y, x])
    if not np.isfinite(z) or z <= 0:
        return None

    camera_point = np.array([
        (x - IMAGE_WIDTH / 2.0) * z / FOCAL_LENGTH,
        -(y - IMAGE_HEIGHT / 2.0) * z / FOCAL_LENGTH,
        -z,
        1.0
    ])

    world_point = pose @ camera_point
    point = world_point[:3]

    if not np.all(np.isfinite(point)):
        return None

    return point


# ============================================================
# EFFICIENT 3D DEDUPLICATION
# ============================================================

def deduplicate(points, distance=DEDUPLICATION_DISTANCE):
    """
    Remove nearby 3D points using a spatial grid.
    The same distance is used for every algorithm.
    """
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)

    cell_size = float(distance)
    if cell_size <= 0:
        return np.asarray(points, dtype=float)

    unique = []
    grid = {}

    for point in points:
        p = np.asarray(point, dtype=float)
        cell = tuple(np.floor(p / cell_size).astype(int))

        is_duplicate = False

        # Check this voxel and its 26 neighbours.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                    for index in grid.get(neighbour, []):
                        if np.linalg.norm(p - unique[index]) <= distance:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break

        if not is_duplicate:
            index = len(unique)
            unique.append(p)
            grid.setdefault(cell, []).append(index)

    return np.asarray(unique, dtype=float)


# ============================================================
# PROCESS VIEWS AND COLLECT METRICS
# ============================================================

all_points = {name: [] for name in ALGORITHMS}
per_view_rows = []
total_raw_counts = {name: 0 for name in ALGORITHMS}
total_projected_counts = {name: 0 for name in ALGORITHMS}
total_selected_counts = {name: 0 for name in ALGORITHMS}
total_detection_time_ms = {name: 0.0 for name in ALGORITHMS}

# NEW ANALYSIS DATA STRUCTURES
selected_keypoints_per_view = {name: [] for name in ALGORITHMS}   # list per view
projected_points_per_view = {name: [] for name in ALGORITHMS}    # list per view of 3D points
descriptor_times_ms = {name: [] for name in ALGORITHMS}
memory_peaks_mb = {name: [] for name in ALGORITHMS}

process = psutil.Process()

try:
    for view_index in range(NUMBER_OF_VIEWS):
        print(f"\nProcessing view {view_index + 1}/{NUMBER_OF_VIEWS}...")

        color, depth, pose = render_view(view_index)

        if color.ndim == 3 and color.shape[2] == 4:
            gray = cv2.cvtColor(color, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)

        detected, times_ms = detect_all(gray)

        for name in ALGORITHMS:
            raw_keypoints = detected[name]
            raw_count = len(raw_keypoints)
            selected = raw_keypoints[:MAX_POINTS_PER_VIEW]

            # store selected keypoints for later descriptor analysis
            selected_keypoints_per_view[name].append(selected)

            projected_count = 0
            view_projected_points = []
            for kp in selected:
                point = back_project(kp.pt[0], kp.pt[1], depth, pose)
                if point is not None:
                    all_points[name].append(point)
                    view_projected_points.append(point)
                    projected_count += 1
            projected_points_per_view[name].append(view_projected_points)

            total_raw_counts[name] += raw_count
            total_selected_counts[name] += len(selected)
            total_projected_counts[name] += projected_count
            total_detection_time_ms[name] += times_ms[name]

            # Descriptor computation for algorithms that have descriptors
            desc_time = 0.0
            peak_mem_mb = 0.0
            if name in ("SIFT", "ORB"):
                mem_before = process.memory_info().rss
                start_desc = time.perf_counter()
                if name == "SIFT":
                    sift = cv2.SIFT_create(nfeatures=0)
                    _, descs = sift.compute(gray, selected)
                else:  # ORB
                    orb = cv2.ORB_create(nfeatures=10000)
                    _, descs = orb.compute(gray, selected)
                desc_time = (time.perf_counter() - start_desc) * 1000.0
                mem_after = process.memory_info().rss
                peak_mem_mb = max(mem_before, mem_after) / (1024 * 1024)
            descriptor_times_ms[name].append(desc_time)
            memory_peaks_mb[name].append(peak_mem_mb)

            per_view_rows.append({
                "view": view_index + 1,
                "algorithm": name,
                "raw_2d_detections": raw_count,
                "points_selected_for_projection": len(selected),
                "successful_3d_projections": projected_count,
                "detection_time_ms": round(times_ms[name], 4),
                "descriptor_time_ms": round(desc_time, 4),
                "projection_success_percent": (
                    round(100.0 * projected_count / len(selected), 2)
                    if selected else 0.0
                )
            })

            print(
                f"  {name:6s}: raw={raw_count:5d}, "
                f"selected={len(selected):3d}, "
                f"projected={projected_count:3d}, "
                f"time={times_ms[name]:.2f} ms, "
                f"desc_time={desc_time:.2f} ms"
            )

finally:
    renderer.delete()


# ============================================================
# DEDUPLICATE AND APPLY THE SAME FINAL EXPORT CAP
# ============================================================

unique_points = {}

for name in ALGORITHMS:
    points = deduplicate(
        all_points[name],
        distance=DEDUPLICATION_DISTANCE
    )

    unique_before_export_cap = len(points)

    if unique_before_export_cap > MAX_UNIQUE_POINTS_PER_ALGORITHM:
        # Evenly sample the sorted spatial insertion order to cap file size.
        indices = np.linspace(
            0,
            unique_before_export_cap - 1,
            MAX_UNIQUE_POINTS_PER_ALGORITHM,
            dtype=int
        )
        points = points[indices]

    unique_points[name] = points

    print(
        f"\n{name}: raw 2D detections={total_raw_counts[name]}, "
        f"projected={total_projected_counts[name]}, "
        f"unique 3D points={unique_before_export_cap}, "
        f"exported markers={len(points)}"
    )


# ============================================================
# EXPORT COLOURED 3D MARKERS
# ============================================================

marker_radius = MARKER_SIZE_RATIO


def make_marker(point, rgb, radius):
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    sphere.apply_translation(point)

    rgba = np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)
    sphere.visual.vertex_colors = np.tile(
        rgba, (len(sphere.vertices), 1)
    )
    return sphere


def export_model(algorithm_names, filename):
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh.copy(),
        node_name="base_model",
        geom_name="base_model"
    )

    for name in algorithm_names:
        points = unique_points[name]
        if len(points) == 0:
            continue

        print(f"Creating {len(points)} markers for {name}...")
        marker_meshes = [
            make_marker(point, COLORS[name], marker_radius)
            for point in points
        ]
        combined = trimesh.util.concatenate(marker_meshes)

        scene.add_geometry(
            combined,
            node_name=f"{name.lower()}_markers",
            geom_name=f"{name.lower()}_markers"
        )

    output_path = os.path.join(output_folder, filename)
    scene.export(output_path)
    print("Saved:", output_path)


for algorithm in ALGORITHMS:
    export_model(
        [algorithm],
        f"car_{algorithm.lower()}_keypoints.glb"
    )

export_model(
    ALGORITHMS,
    "car_all_algorithms_keypoints.glb"
)


# ============================================================
# SAVE CSV REPORTS
# ============================================================

per_view_path = os.path.join(output_folder, "per_view_metrics.csv")
with open(per_view_path, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
    "view",
    "algorithm",
    "raw_2d_detections",
    "points_selected_for_projection",
    "successful_3d_projections",
    "detection_time_ms",
    "descriptor_time_ms",  # ADD THIS LINE
    "projection_success_percent"
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_view_rows)


summary_path = os.path.join(output_folder, "summary_metrics.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "algorithm",
        "views_processed",
        "total_raw_2d_detections",
        "average_raw_detections_per_view",
        "total_points_selected_for_projection",
        "successful_3d_projections",
        "projection_success_percent",
        "total_detection_time_ms",
        "average_detection_time_ms",
        "unique_3d_points_before_export_cap",
        "exported_3d_markers",
        "mser_delta",
        "mser_min_area",
        "mser_max_area",
        "mser_max_variation",
        "mser_min_diversity",
        "shared_max_points_per_view",
        "shared_deduplication_distance"
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for name in ALGORITHMS:
        raw_count = total_raw_counts[name]
        projected_count = total_projected_counts[name]
        total_time = total_detection_time_ms[name]

        # Recreate the pre-export unique count for the report.
        # This count is derived from the collected projected points.
        pre_cap_unique = len(
            deduplicate(
                all_points[name],
                distance=DEDUPLICATION_DISTANCE
            )
        )

        row = {
            "algorithm": name,
            "views_processed": NUMBER_OF_VIEWS,
            "total_raw_2d_detections": raw_count,
            "average_raw_detections_per_view": round(
                raw_count / NUMBER_OF_VIEWS, 3
            ),
            "total_points_selected_for_projection": total_selected_counts[name],
            "successful_3d_projections": projected_count,
            "projection_success_percent": round(
                100.0 * projected_count /
                max(1, total_selected_counts[name]),
                2
            ),
            "total_detection_time_ms": round(total_time, 4),
            "average_detection_time_ms": round(
                total_time / NUMBER_OF_VIEWS, 4
            ),
            "unique_3d_points_before_export_cap": pre_cap_unique,
            "exported_3d_markers": len(unique_points[name]),
            "mser_delta": 5 if name == "MSER" else "",
            "mser_min_area": 30 if name == "MSER" else "",
            "mser_max_area": 14400 if name == "MSER" else "",
            "mser_max_variation": 0.25 if name == "MSER" else "",
            "mser_min_diversity": 0.20 if name == "MSER" else "",
            "shared_max_points_per_view": MAX_POINTS_PER_VIEW,
            "shared_deduplication_distance": DEDUPLICATION_DISTANCE
        }
        writer.writerow(row)


coordinates_path = os.path.join(output_folder, "keypoint_coordinates.csv")
with open(coordinates_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["algorithm", "x", "y", "z"])

    for name in ALGORITHMS:
        for x, y, z in unique_points[name]:
            writer.writerow([name, float(x), float(y), float(z)])


# Save detector settings and experiment settings for the blog.
settings_path = os.path.join(output_folder, "experiment_settings.txt")
with open(settings_path, "w", encoding="utf-8") as file:
    file.write("3D Feature Detection Comparison — Experiment Settings\n")
    file.write("=====================================================\n")
    file.write(f"Input model: {model_path}\n")
    file.write(f"Image size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}\n")
    file.write(f"Focal length: {FOCAL_LENGTH}\n")
    file.write(f"Number of views: {NUMBER_OF_VIEWS}\n")
    file.write(f"Shared max points per view: {MAX_POINTS_PER_VIEW}\n")
    file.write(f"Shared 3D deduplication distance: {DEDUPLICATION_DISTANCE}\n")
    file.write(f"Shared final export cap per algorithm: {MAX_UNIQUE_POINTS_PER_ALGORITHM}\n")
    file.write("All algorithms receive the same grayscale render; no CLAHE is applied.\n")
    file.write("MSER: delta=5, min_area=30, max_area=14400, "
               "max_variation=0.25, min_diversity=0.20\n")
    file.write("FAST: threshold=20, nonmaxSuppression=True\n")
    file.write("Harris: blockSize=2, apertureSize=3, k=0.04, "
               "threshold=1% of maximum response, local maxima selected\n")
    file.write("ORB: nfeatures=10000 before the shared projection cap\n")
    file.write("SIFT: nfeatures=0 (all detected keypoints before shared cap)\n")
    file.write("\nInterpretation note:\n")
    file.write("Counts measure detections, not accuracy. Accuracy/repeatability "
               "requires ground truth or a defined evaluation protocol.\n")
    file.write("3D coordinates are obtained by projecting detected 2D pixels "
               "using the renderer depth buffer.\n")


# ============================================================
# NEW ANALYSIS SECTION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter
from sklearn.neighbors import NearestNeighbors
import math
import os

# ---------- Gather per-view detection times ----------
detection_times_per_view = {name: [] for name in ALGORITHMS}
descriptor_times_per_view = {name: [] for name in ALGORITHMS}
for row in per_view_rows:
    detection_times_per_view[row["algorithm"]].append(row["detection_time_ms"])
    descriptor_times_per_view[row["algorithm"]].append(row.get("descriptor_time_ms", 0.0))

# ---------- 1. Detection Speed ----------
det_speed_rows = []
for name in ALGORITHMS:
    times = detection_times_per_view[name]
    if times:
        total = sum(times)
        avg = np.mean(times)
        mn = np.min(times)
        mx = np.max(times)
        std = np.std(times)
    else:
        total = avg = mn = mx = std = 0.0
    det_speed_rows.append({
        "algorithm": name,
        "total_detection_time_ms": round(total, 4),
        "avg_detection_time_ms": round(avg, 4),
        "min_detection_time_ms": round(mn, 4),
        "max_detection_time_ms": round(mx, 4),
        "std_detection_time_ms": round(std, 4)
    })

# ---------- 2. Descriptor Performance ----------
desc_rows = []
for name in ALGORITHMS:
    if name in ("SIFT", "ORB"):
        dtimes = descriptor_times_per_view[name]
        total_desc = sum(dtimes)
        avg_desc = np.mean(dtimes) if dtimes else 0.0
        # descriptor count approximated by total selected keypoints for those views
        total_desc_count = sum(len(selected_keypoints_per_view[name][i]) for i in range(NUMBER_OF_VIEWS))
        dim = 128 if name == "SIFT" else 32
        dtype = "float32" if name == "SIFT" else "uint8"
        desc_rows.append({
            "algorithm": name,
            "total_descriptor_time_ms": round(total_desc, 4),
            "avg_descriptor_time_ms": round(avg_desc, 4),
            "descriptor_count": total_desc_count,
            "descriptor_dim": dim,
            "descriptor_dtype": dtype
        })
    else:
        desc_rows.append({
            "algorithm": name,
            "total_descriptor_time_ms": "N/A",
            "avg_descriptor_time_ms": "N/A",
            "descriptor_count": "N/A",
            "descriptor_dim": "N/A",
            "descriptor_dtype": "N/A"
        })

# ---------- 3. Keypoint Statistics ----------
kp_stat_rows = []
for name in ALGORITHMS:
    raw = total_raw_counts[name]
    selected = total_selected_counts[name]
    projected = total_projected_counts[name]
    success_rate = round(100.0 * projected / max(1, selected), 2)
    # unique before export cap
    pre_cap_unique = len(
        deduplicate(all_points[name], distance=DEDUPLICATION_DISTANCE)
    )
    exported = len(unique_points[name])
    duplicate_rate = round(100.0 * (projected - pre_cap_unique) / max(1, projected), 2)
    kp_stat_rows.append({
        "algorithm": name,
        "raw_detections": raw,
        "selected_keypoints": selected,
        "successful_projections": projected,
        "projection_success_percent": success_rate,
        "unique_3d_points": pre_cap_unique,
        "exported_markers": exported,
        "duplicate_rate_percent": duplicate_rate
    })

# ---------- 4. Repeatability ----------
repeat_rows = []
repeat_summary = []
for name in ALGORITHMS:
    view_points = projected_points_per_view[name]  # list length NUMBER_OF_VIEWS
    pair_repeat = []
    for i in range(NUMBER_OF_VIEWS - 1):
        pts_a = np.array(view_points[i])
        pts_b = np.array(view_points[i + 1])
        if len(pts_a) == 0 or len(pts_b) == 0:
            rep = 0.0
        else:
            # Build kd-tree for pts_b
            nbrs = NearestNeighbors(n_neighbors=1, radius=DEDUPLICATION_DISTANCE).fit(pts_b)
            distances, _ = nbrs.kneighbors(pts_a)
            matches = np.sum(distances <= DEDUPLICATION_DISTANCE)
            rep = 100.0 * matches / min(len(pts_a), len(pts_b))
        pair_repeat.append(rep)
        repeat_rows.append({
            "algorithm": name,
            "view_pair": f"{i+1}-{i+2}",
            "repeatability_percent": round(rep, 2)
        })
    if pair_repeat:
        avg_rep = np.mean(pair_repeat)
        min_rep = np.min(pair_repeat)
        max_rep = np.max(pair_repeat)
    else:
        avg_rep = min_rep = max_rep = 0.0
    repeat_summary.append({
        "algorithm": name,
        "avg_repeatability_percent": round(avg_rep, 2),
        "min_repeatability_percent": round(min_rep, 2),
        "max_repeatability_percent": round(max_rep, 2)
    })

# ---------- 5. Spatial Distribution ----------
spatial_rows = []
# mesh surface area
mesh_area = mesh.area if hasattr(mesh, "area") else None
for name in ALGORITHMS:
    pts = unique_points[name]
    if len(pts) >= 2:
        nbrs = NearestNeighbors(n_neighbors=2).fit(pts)
        distances, _ = nbrs.kneighbors(pts)
        nn_dist = distances[:, 1]  # exclude self
        mean_nn = float(np.mean(nn_dist))
        median_nn = float(np.median(nn_dist))
        min_nn = float(np.min(nn_dist))
        max_nn = float(np.max(nn_dist))
        std_nn = float(np.std(nn_dist))
    else:
        mean_nn = median_nn = min_nn = max_nn = std_nn = 0.0
    unique_cnt = len(pts)
    feature_density = unique_cnt / mesh_area if mesh_area and mesh_area > 0 else float('nan')
    # Spatial coverage: sample mesh vertices
    coverage = 0.0
    if len(mesh.vertices) > 0 and len(pts) > 0:
        # Use a radius equal to marker_radius * 2 as coverage radius
        cov_radius = MARKER_SIZE_RATIO * 2.0
        nbrs_cov = NearestNeighbors(radius=cov_radius).fit(pts)
        # count vertices within radius of any keypoint
        # query_radius returns list of neighbor indices per vertex
        # We'll approximate by checking each vertex
        covered = 0
        for v in mesh.vertices:
            inds = nbrs_cov.radius_neighbors([v], radius=cov_radius, return_distance=False)[0]
            if len(inds) > 0:
                covered += 1
        coverage = 100.0 * covered / len(mesh.vertices)
    spatial_rows.append({
        "algorithm": name,
        "mean_nn_distance": round(mean_nn, 6),
        "median_nn_distance": round(median_nn, 6),
        "min_nn_distance": round(min_nn, 6),
        "max_nn_distance": round(max_nn, 6),
        "std_nn_distance": round(std_nn, 6),
        "feature_density_per_area": round(feature_density, 6) if not math.isnan(feature_density) else "N/A",
        "spatial_coverage_percent": round(coverage, 2)
    })

# ---------- 6. Memory Usage ----------
mem_rows = []
for name in ALGORITHMS:
    peaks = memory_peaks_mb[name]
    peak = max(peaks) if peaks else 0.0
    mem_rows.append({
        "algorithm": name,
        "peak_memory_mb": round(peak, 2)
    })

# ---------- 7. Per-view Statistics (already in per_view_rows, ensure descriptor_time column) ----------
# Already saved in per_view_rows with descriptor_time_ms.

# ---------- Save additional CSV files ----------
# comparison_table.csv
comp_rows = []
for i, name in enumerate(ALGORITHMS):
    ds = det_speed_rows[i]
    dsc = desc_rows[i]
    ks = kp_stat_rows[i]
    rs = repeat_summary[i]
    sp = spatial_rows[i]
    mr = mem_rows[i]
    comp_rows.append({
        "Algorithm": name,
        "Total_Detection_Time_ms": ds["total_detection_time_ms"],
        "Avg_Detection_Time_ms": ds["avg_detection_time_ms"],
        "Total_Descriptor_Time_ms": dsc["total_descriptor_time_ms"],
        "Avg_Descriptor_Time_ms": dsc["avg_descriptor_time_ms"],
        "Raw_Detections": ks["raw_detections"],
        "Selected_Keypoints": ks["selected_keypoints"],
        "Projected_Keypoints": ks["successful_projections"],
        "Projection_Success_%": ks["projection_success_percent"],
        "Unique_3D_Points": ks["unique_3d_points"],
        "Duplicate_Rate_%": ks["duplicate_rate_percent"],
        "Repeatability_%": rs["avg_repeatability_percent"],
        "Mean_NN_Distance": sp["mean_nn_distance"],
        "Feature_Density": sp["feature_density_per_area"],
        "Spatial_Coverage_%": sp["spatial_coverage_percent"],
        "Peak_Memory_MB": mr["peak_memory_mb"]
    })

comp_df = pd.DataFrame(comp_rows)
comp_csv_path = os.path.join(output_folder, "comparison_table.csv")
comp_df.to_csv(comp_csv_path, index=False)

# comparison_table.xlsx with formatting
wb = Workbook()
ws = wb.active
ws.title = "Comparison"
# headers
headers = list(comp_df.columns)
for col_idx, h in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col_idx, value=h)
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal="center")
for row_idx, row in comp_df.iterrows():
    for col_idx, val in enumerate(row, 1):
        ws.cell(row=row_idx+2, column=col_idx, value=val)
# auto column width
for col_idx in range(1, len(headers)+1):
    ws.column_dimensions[get_column_letter(col_idx)].width = 20
xlsx_path = os.path.join(output_folder, "comparison_table.xlsx")
wb.save(xlsx_path)

# descriptor_metrics.csv
pd.DataFrame(desc_rows).to_csv(os.path.join(output_folder, "descriptor_metrics.csv"), index=False)

# repeatability_metrics.csv
pd.DataFrame(repeat_rows).to_csv(os.path.join(output_folder, "repeatability_metrics.csv"), index=False)

# spatial_metrics.csv
pd.DataFrame(spatial_rows).to_csv(os.path.join(output_folder, "spatial_metrics.csv"), index=False)

# timing_metrics.csv (detection speed)
pd.DataFrame(det_speed_rows).to_csv(os.path.join(output_folder, "timing_metrics.csv"), index=False)

# memory_metrics.csv
pd.DataFrame(mem_rows).to_csv(os.path.join(output_folder, "memory_metrics.csv"), index=False)

# ---------- Generate Graphs ----------
def save_bar_chart(data_dict, ylabel, title, filename):
    names = list(data_dict.keys())
    values = list(data_dict.values())
    plt.figure(figsize=(8,5))
    bars = plt.bar(names, values, color=[np.array(COLORS[n])/255.0 for n in names])
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# Detection Time
save_bar_chart({r["algorithm"]: r["total_detection_time_ms"] for r in det_speed_rows},
               "Time (ms)", "Total Detection Time per Algorithm", "detection_time.png")
# Average Detection Time
save_bar_chart({r["algorithm"]: r["avg_detection_time_ms"] for r in det_speed_rows},
               "Time (ms)", "Average Detection Time per View", "avg_detection_time.png")
# Raw Keypoints
save_bar_chart({r["algorithm"]: r["raw_detections"] for r in kp_stat_rows},
               "Count", "Raw 2D Detections", "raw_keypoints.png")
# Unique 3D Keypoints
save_bar_chart({r["algorithm"]: r["unique_3d_points"] for r in kp_stat_rows},
               "Count", "Unique 3D Keypoints", "unique_3d_keypoints.png")
# Projection Success
save_bar_chart({r["algorithm"]: r["projection_success_percent"] for r in kp_stat_rows},
               "Percent", "Projection Success Rate", "projection_success.png")
# Repeatability
save_bar_chart({r["algorithm"]: r["avg_repeatability_percent"] for r in repeat_summary},
               "Percent", "Average Repeatability", "repeatability.png")
# Nearest-Neighbour Distance
save_bar_chart({r["algorithm"]: r["mean_nn_distance"] for r in spatial_rows},
               "Distance", "Mean Nearest-Neighbour Distance", "nn_distance.png")
# Feature Density
save_bar_chart({r["algorithm"]: (r["feature_density_per_area"] if isinstance(r["feature_density_per_area"], float) else 0) for r in spatial_rows},
               "Density", "Feature Density (pts/area)", "feature_density.png")
# Memory Usage
save_bar_chart({r["algorithm"]: r["peak_memory_mb"] for r in mem_rows},
               "MB", "Peak Memory Usage", "memory_usage.png")

# ---------- Print Comparison Table to Terminal ----------
print("\n" + "=" * 120)
print("COMPARISON TABLE")
print("=" * 120)
header = ["Algorithm", "TotDet(ms)", "AvgDet(ms)", "TotDesc(ms)", "AvgDesc(ms)",
          "Raw", "Sel", "Proj", "Proj%", "Unq3D", "Dup%", "Rep%", "MeanNN", "FeatDens", "Cov%", "MemMB"]
print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>8} {:>9} {:>6} {:>7}".format(*header))
for r in comp_rows:
    vals = [
        r["Algorithm"],
        r["Total_Detection_Time_ms"],
        r["Avg_Detection_Time_ms"],
        r["Total_Descriptor_Time_ms"],
        r["Avg_Descriptor_Time_ms"],
        r["Raw_Detections"],
        r["Selected_Keypoints"],
        r["Projected_Keypoints"],
        r["Projection_Success_%"],
        r["Unique_3D_Points"],
        r["Duplicate_Rate_%"],
        r["Repeatability_%"],
        r["Mean_NN_Distance"],
        r["Feature_Density"],
        r["Spatial_Coverage_%"],
        r["Peak_Memory_MB"]
    ]
    print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6} {:>6} {:>6.2f} {:>7} {:>6.2f} {:>6.2f} {:>8.6f} {:>9} {:>6.2f} {:>7.2f}".format(*vals))
print("=" * 120)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("Output folder:", output_folder)
print("\nFiles created:")
print("- car_mser_keypoints.glb")
print("- car_sift_keypoints.glb")
print("- car_fast_keypoints.glb")
print("- car_harris_keypoints.glb")
print("- car_orb_keypoints.glb")
print("- car_all_algorithms_keypoints.glb")
print("- per_view_metrics.csv")
print("- summary_metrics.csv")
print("- keypoint_coordinates.csv")
print("- experiment_settings.txt")

try:
    messagebox.showinfo(
        "Analysis complete",
        "3D models and analysis CSV files have been exported.\n\n"
        f"Output folder:\n{output_folder}\n\n"
        "Use summary_metrics.csv and per_view_metrics.csv for your blog."
    )
finally:
    root.destroy()
