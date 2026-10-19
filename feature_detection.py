import sys
import time
from pathlib import Path
     
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # we only save figures, never show them
import matplotlib.pyplot as plt


RUNS = 10                # timing repetitions (median reported)
MAX_KEYPOINTS = 1000     # common cap for every algorithm
MAX_DIM = 1200           # resize so the longest side is at most this (None = off)
MATCH_THRESHOLD = 3      # pixels, for repeatability
BORDER_MARGIN = 20       # ignore keypoints this close to the border in repeatability
SEED = 0                 # makes the noise test reproducible

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FOLDER = BASE_DIR / "feature_benchmark_results"
VISUAL_FOLDER = OUTPUT_FOLDER / "visualizations"
GRAPH_FOLDER = OUTPUT_FOLDER / "graphs"

for folder in [OUTPUT_FOLDER, VISUAL_FOLDER, GRAPH_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# ================================================================
# SELECT IMAGES
# ================================================================

def select_images():
    # Paths given on the command line skip the file picker
    if len(sys.argv) > 1:
        return [Path(p) for p in sys.argv[1:]]

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    print()
    print("=" * 70)
    print("SELECT YOUR TEST IMAGES")
    print("=" * 70)
    print("A file window will open. Select one or more JPG/PNG images.")
    print()

    files = filedialog.askopenfilenames(
        title="Select images for feature detection benchmark",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not files:
        raise SystemExit("\nNo images selected. Program stopped.")

    return [Path(f) for f in files]


image_files = select_images()

print(f"OpenCV version: {cv2.__version__}")


# ================================================================
# DETECTORS (created ONCE, outside any timer)
#
# Each entry has:
#   run        - the core OpenCV call (this is what gets timed)
#   to_kp      - converts raw output to a list of cv2.KeyPoint (not timed)
#   descriptor - object with .compute() for descriptor timing, or None
# ================================================================

def build_detectors():
    detectors = {}

    def try_build(name, builder):
        try:
            detectors[name] = builder()
        except Exception as error:
            print(f"Skipping {name}: {type(error).__name__}: {error}")

    # ---- FAST ----
    def build_fast():
        fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        return {
            "run": lambda g: fast.detect(g, None),
            "to_kp": lambda g, out: list(out),
            "descriptor": None,
        }

    # ---- Harris (response map, then threshold + non-max suppression) ----
    def build_harris():
        def run(g):
            return cv2.cornerHarris(np.float32(g), blockSize=2, ksize=3, k=0.04)

        def to_kp(g, response):
            threshold = 0.01 * response.max()
            local_max = response == cv2.dilate(response, np.ones((5, 5), np.uint8))
            ys, xs = np.where((response > threshold) & local_max)
            return [
                cv2.KeyPoint(float(x), float(y), 7.0, -1, float(response[y, x]))
                for x, y in zip(xs, ys)
            ]

        return {"run": run, "to_kp": to_kp, "descriptor": None}

    # ---- SIFT ----
    def build_sift():
        sift = cv2.SIFT_create()
        return {
            "run": lambda g: sift.detect(g, None),
            "to_kp": lambda g, out: list(out),
            "descriptor": sift,
        }

    # ---- ORB ----
    def build_orb():
        orb = cv2.ORB_create(nfeatures=MAX_KEYPOINTS)
        return {
            "run": lambda g: orb.detect(g, None),
            "to_kp": lambda g, out: list(out),
            "descriptor": orb,
        }

    # ---- MSER ----
    def build_mser():
        mser = cv2.MSER_create()

        def to_kp(g, out):
            regions, _ = out
            keypoints = []
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                keypoints.append(
                    cv2.KeyPoint(float(x + w / 2), float(y + h / 2), float(max(w, h, 1)))
                )
            return keypoints

        return {"run": lambda g: mser.detectRegions(g), "to_kp": to_kp, "descriptor": None}

    try_build("FAST", build_fast)
    try_build("Harris", build_harris)
    try_build("SIFT", build_sift)
    try_build("ORB", build_orb)
    try_build("MSER", build_mser)

    return detectors


DETECTORS = build_detectors()
ALGORITHM_ORDER = list(DETECTORS.keys())

if not DETECTORS:
    raise SystemExit("No detectors could be created in this OpenCV build.")


# ================================================================
# HELPERS
# ================================================================

def cap_keypoints(keypoints, limit=MAX_KEYPOINTS):
    """Keep the strongest `limit` keypoints (by response, or by size if no response)."""
    if len(keypoints) <= limit:
        return keypoints

    responses = np.array([kp.response for kp in keypoints])

    if responses.max() > 0:
        key = responses
    else:
        key = np.array([kp.size for kp in keypoints])

    order = np.argsort(-key)[:limit]
    return [keypoints[i] for i in order]


def benchmark_detector(detector, gray):
    """Warm-up, then RUNS timed calls of the core function only."""
    detector["run"](gray)

    times = []
    output = None

    for _ in range(RUNS):
        start = time.perf_counter()
        output = detector["run"](gray)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "output": output,
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def benchmark_descriptor(detector, gray, keypoints):
    """Time descriptor computation for algorithms that have one."""
    empty = {
        "count": np.nan,
        "dimensions": np.nan,
        "dtype": None,
        "memory": np.nan,
        "median_ms": np.nan,
    }

    descriptor = detector["descriptor"]

    if descriptor is None or len(keypoints) == 0:
        return empty

    descriptor.compute(gray, keypoints)  # warm-up

    times = []
    descriptors = None

    for _ in range(RUNS):
        start = time.perf_counter()
        _, descriptors = descriptor.compute(gray, keypoints)
        times.append((time.perf_counter() - start) * 1000)

    if descriptors is None:
        return empty

    return {
        "count": len(descriptors),
        "dimensions": descriptors.shape[1],
        "dtype": str(descriptors.dtype),
        "memory": int(descriptors.nbytes),
        "median_ms": float(np.median(times)),
    }


def calculate_statistics(keypoints, width, height):
    count = len(keypoints)

    if count == 0:
        return {
            "count": 0,
            "density": 0.0,
            "average_size": np.nan,
            "average_response": np.nan,
            "spatial_spread": np.nan,
        }

    sizes = np.array([kp.size for kp in keypoints], dtype=float)
    responses = np.array([kp.response for kp in keypoints], dtype=float)
    x = np.array([kp.pt[0] for kp in keypoints], dtype=float)
    y = np.array([kp.pt[1] for kp in keypoints], dtype=float)

    return {
        "count": count,
        "density": count / (width * height) * 1_000_000,
        "average_size": float(sizes.mean()),
        # MSER has no response, so report NaN instead of a misleading 0
        "average_response": float(responses.mean()) if responses.max() > 0 else np.nan,
        "spatial_spread": float(np.sqrt(x.std() ** 2 + y.std() ** 2)),
    }


def save_visualization(image, keypoints, algorithm_name, image_name):
    output = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    filename = f"{Path(image_name).stem}_{algorithm_name.replace('-', '_')}.jpg"
    cv2.imwrite(str(VISUAL_FOLDER / filename), output)


# ================================================================
# IMAGE TRANSFORMATIONS
# Each returns (transformed_image, 2x3 matrix mapping original -> transformed).
# Photometric changes keep the geometry, so they use the identity matrix.
# ================================================================

IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def create_transformations(image):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rng = np.random.default_rng(SEED)

    transformations = {}

    # Rotation 30 degrees
    M = cv2.getRotationMatrix2D(center, 30, 1.0)
    transformations["rotation_30"] = (cv2.warpAffine(image, M, (width, height)), M)

    # Scale to 70 percent (about the image centre)
    M = cv2.getRotationMatrix2D(center, 0, 0.7)
    transformations["scale_0.7"] = (cv2.warpAffine(image, M, (width, height)), M)

    # Blur
    transformations["blur"] = (cv2.GaussianBlur(image, (9, 9), 0), IDENTITY)

    # Brightness / darkness (values are clipped at 0 and 255)
    transformations["bright"] = (cv2.convertScaleAbs(image, alpha=1.0, beta=50), IDENTITY)
    transformations["dark"] = (cv2.convertScaleAbs(image, alpha=1.0, beta=-50), IDENTITY)

    # Gaussian noise (seeded)
    noisy = image.astype(np.float32) + rng.normal(0, 20, image.shape).astype(np.float32)
    transformations["noise"] = (np.clip(noisy, 0, 255).astype(np.uint8), IDENTITY)

    return transformations


# ================================================================
# REPEATABILITY
# ================================================================

def calculate_repeatability(original_keypoints, transformed_keypoints, M, width, height):
    """
    % of original keypoints that are re-detected after the transform.
    Original points are first moved through M, and only points that stay
    well inside the image (both before and after) are counted.
    """
    if not original_keypoints or not transformed_keypoints:
        return np.nan

    original = np.array([kp.pt for kp in original_keypoints], dtype=np.float64)
    mapped = original @ M[:, :2].T + M[:, 2]
    detected = np.array([kp.pt for kp in transformed_keypoints], dtype=np.float64)

    limit = np.array([width - BORDER_MARGIN, height - BORDER_MARGIN])

    valid = (
        ((original > BORDER_MARGIN) & (original < limit)).all(axis=1)
        & ((mapped > BORDER_MARGIN) & (mapped < limit)).all(axis=1)
    )

    if not valid.any():
        return np.nan

    mapped = mapped[valid]

    distances = np.empty(len(mapped))
    chunk = 500
    for i in range(0, len(mapped), chunk):
        block = mapped[i:i + chunk]
        distances[i:i + chunk] = np.linalg.norm(
            block[:, None, :] - detected[None, :, :], axis=2
        ).min(axis=1)

    return 100.0 * float((distances <= MATCH_THRESHOLD).mean())


# ================================================================
# RUN BENCHMARK
# ================================================================

results = []

print()
print("=" * 70)
print("STARTING FEATURE DETECTION BENCHMARK")
print("=" * 70)

for image_number, image_path in enumerate(image_files, start=1):

    print()
    print("=" * 70)
    print(f"IMAGE {image_number}/{len(image_files)}: {image_path.name}")
    print("=" * 70)

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read {image_path.name}. Skipping.")
        continue

    if MAX_DIM is not None and max(image.shape[:2]) > MAX_DIM:
        factor = MAX_DIM / max(image.shape[:2])
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(f"Working size: {width} x {height}")

    original_keypoints = {}

    # ------------------------------------------------------------
    # ORIGINAL IMAGE
    # ------------------------------------------------------------

    print()
    print("Testing original image...")

    for name, detector in DETECTORS.items():

        print(f"  {name:<10}", end="", flush=True)

        try:
            benchmark = benchmark_detector(detector, gray)

            keypoints = detector["to_kp"](gray, benchmark["output"])
            raw_count = len(keypoints)
            keypoints = cap_keypoints(keypoints)

            statistics = calculate_statistics(keypoints, width, height)
            descriptor = benchmark_descriptor(detector, gray, keypoints)

            total_time = benchmark["median_ms"] + (
                0 if np.isnan(descriptor["median_ms"]) else descriptor["median_ms"]
            )

            save_visualization(image, keypoints, name, image_path.name)
            original_keypoints[name] = keypoints

            results.append({
                "Image": image_path.name,
                "Transformation": "Original",
                "Algorithm": name,
                "Keypoints (raw)": raw_count,
                "Keypoints": statistics["count"],
                "Keypoint Density": statistics["density"],
                "Detection Time Median (ms)": benchmark["median_ms"],
                "Detection Time Std (ms)": benchmark["std_ms"],
                "Detection Time Min (ms)": benchmark["min_ms"],
                "Detection Time Max (ms)": benchmark["max_ms"],
                "Descriptor Time Median (ms)": descriptor["median_ms"],
                "Total Time Detect+Describe (ms)": total_time,
                "Average Keypoint Size": statistics["average_size"],
                "Average Response": statistics["average_response"],
                "Spatial Spread": statistics["spatial_spread"],
                "Descriptor Count": descriptor["count"],
                "Descriptor Dimensions": descriptor["dimensions"],
                "Descriptor Type": descriptor["dtype"],
                "Descriptor Memory (bytes)": descriptor["memory"],
                "Repeatability (%)": np.nan,
            })

            print(
                f"{raw_count} raw -> {statistics['count']} used, "
                f"{benchmark['median_ms']:.2f} ms"
            )

        except Exception as error:
            print(f"FAILED: {type(error).__name__}: {error}")

    # ------------------------------------------------------------
    # TRANSFORMATIONS
    # ------------------------------------------------------------

    print()
    print("Testing image transformations...")

    transformations = create_transformations(image)

    for transformation_name, (transformed_image, M) in transformations.items():

        transformed_gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        for name, detector in DETECTORS.items():

            if name not in original_keypoints:
                continue

            try:
                transformed_keypoints = cap_keypoints(
                    detector["to_kp"](transformed_gray, detector["run"](transformed_gray))
                )

                repeatability = calculate_repeatability(
                    original_keypoints[name], transformed_keypoints, M, width, height
                )

                statistics = calculate_statistics(transformed_keypoints, width, height)

                results.append({
                    "Image": image_path.name,
                    "Transformation": transformation_name,
                    "Algorithm": name,
                    "Keypoints": statistics["count"],
                    "Keypoint Density": statistics["density"],
                    "Average Keypoint Size": statistics["average_size"],
                    "Average Response": statistics["average_response"],
                    "Spatial Spread": statistics["spatial_spread"],
                    "Repeatability (%)": repeatability,
                })

            except Exception as error:
                print(
                    f"  {name} / {transformation_name} failed: "
                    f"{type(error).__name__}: {error}"
                )

        print(f"  {transformation_name} done")


# ================================================================
# DATAFRAME
# ================================================================

if not results:
    print(
        "No benchmark results were generated. "
        "Check that you selected readable image files (not .py files)."
    )
    raise SystemExit(1)

df = pd.DataFrame(results)

number_columns = [
    "Keypoint Density",
    "Detection Time Median (ms)",
    "Detection Time Std (ms)",
    "Detection Time Min (ms)",
    "Detection Time Max (ms)",
    "Descriptor Time Median (ms)",
    "Total Time Detect+Describe (ms)",
    "Average Keypoint Size",
    "Average Response",
    "Spatial Spread",
    "Repeatability (%)",
]

for column in number_columns:
    if column in df.columns:
        df[column] = df[column].round(4)


# ================================================================
# SAVE COMPLETE RESULTS
# ================================================================

csv_file = OUTPUT_FOLDER / "complete_benchmark_results.csv"
excel_file = OUTPUT_FOLDER / "complete_benchmark_results.xlsx"

df.to_csv(csv_file, index=False)

try:
    df.to_excel(excel_file, index=False)
except ImportError:
    excel_file = None
    print("\nSkipped Excel export: run  pip install openpyxl")


# ================================================================
# SUMMARIES
# ================================================================

original = df[df["Transformation"] == "Original"]

summary = (
    original.groupby("Algorithm")
    .agg({
        "Keypoints (raw)": "mean",
        "Keypoints": "mean",
        "Keypoint Density": "mean",
        "Detection Time Median (ms)": "mean",
        "Detection Time Std (ms)": "mean",
        "Descriptor Time Median (ms)": "mean",
        "Total Time Detect+Describe (ms)": "mean",
        "Average Keypoint Size": "mean",
        "Spatial Spread": "mean",
        "Descriptor Memory (bytes)": "mean",
    })
    .reindex(ALGORITHM_ORDER)
    .round(3)
)

summary_file = OUTPUT_FOLDER / "algorithm_summary.csv"
summary.to_csv(summary_file)

# Repeatability per transformation: rows = algorithm, columns = transformation
transform_order = [
    t for t in ["rotation_30", "scale_0.7", "blur", "bright", "dark", "noise"]
    if t in set(df["Transformation"])
]

repeatability_table = (
    df[df["Transformation"] != "Original"]
    .pivot_table(
        index="Algorithm",
        columns="Transformation",
        values="Repeatability (%)",
        aggfunc="mean",
    )
    .reindex(index=ALGORITHM_ORDER, columns=transform_order)
    .round(1)
)

repeatability_file = OUTPUT_FOLDER / "repeatability_by_transformation.csv"
repeatability_table.to_csv(repeatability_file)


# ================================================================
# GRAPHS
# ================================================================

def create_bar_graph(series, title, y_label, filename, log_scale=False):
    series = series.dropna()

    if series.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(series.index, series.values)
    plt.title(title)
    plt.xlabel("Feature Detection Algorithm")
    plt.ylabel(y_label)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(GRAPH_FOLDER / filename, dpi=300)
    plt.close()


create_bar_graph(
    summary["Keypoints"],
    f"Keypoints Detected (capped at {MAX_KEYPOINTS})",
    "Number of Keypoints",
    "01_keypoint_count.png",
)

create_bar_graph(
    summary["Keypoint Density"],
    "Keypoint Density",
    "Keypoints per Million Pixels",
    "02_keypoint_density.png",
)

create_bar_graph(
    summary["Detection Time Median (ms)"],
    "Median Detection Time (log scale)",
    "Time (ms)",
    "03_detection_time.png",
    log_scale=True,
)

create_bar_graph(
    summary["Total Time Detect+Describe (ms)"],
    "Median Detection + Description Time (log scale)",
    "Time (ms)",
    "04_total_time.png",
    log_scale=True,
)

create_bar_graph(
    summary["Spatial Spread"],
    "Spatial Distribution of Keypoints",
    "Spatial Spread (pixels)",
    "05_spatial_spread.png",
)

# Grouped repeatability chart (one group per transformation)
ax = repeatability_table.T.plot(kind="bar", figsize=(12, 6), width=0.8)
ax.set_title("Repeatability by Transformation")
ax.set_xlabel("Transformation")
ax.set_ylabel("Keypoints re-detected (%)")
ax.set_ylim(0, 100)
plt.xticks(rotation=0)
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig(GRAPH_FOLDER / "06_repeatability.png", dpi=300)
plt.close()

create_bar_graph(
    summary["Descriptor Memory (bytes)"],
    "Descriptor Memory Usage",
    "Memory (bytes)",
    "07_descriptor_memory.png",
)


# ================================================================
# PRINT FINAL RESULTS
# ================================================================

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

with pd.option_context("display.max_columns", None, "display.width", 200):
    print()
    print("ORIGINAL-IMAGE SUMMARY (mean over images)")
    print(summary.fillna("N/A").to_string())

    print()
    print("REPEATABILITY (%) BY TRANSFORMATION")
    print(repeatability_table.fillna("N/A").to_string())

print()
print("Algorithms tested:", ", ".join(ALGORITHM_ORDER))
print("OpenCV version   :", cv2.__version__)

print()
print("Images tested:")
for image_path in image_files:
    print(f"  - {image_path.name}")

print()
print("RESULT FILES")
print("-" * 70)
print(f"Complete CSV  : {csv_file.resolve()}")
if excel_file:
    print(f"Complete Excel: {excel_file.resolve()}")
print(f"Summary CSV   : {summary_file.resolve()}")
print(f"Repeatability : {repeatability_file.resolve()}")
print(f"Visualizations: {VISUAL_FOLDER.resolve()}")
print(f"Graphs        : {GRAPH_FOLDER.resolve()}")

print()
print("Notes:")
print(f" - Timing = median of {RUNS} runs of the core call only (after a warm-up).")
print("   Harris time is the response map only; SIFT/ORB descriptor time is separate.")
print(f" - All algorithms are capped to the strongest {MAX_KEYPOINTS} keypoints.")
print(f" - Repeatability: re-detected within {MATCH_THRESHOLD}px after mapping through the transform.")
print(" - Average Response is not comparable between algorithms.")
print()
print("=" * 70)
