
import cv2
import numpy as np
import os
import csv
import time
import tkinter as tk

from tkinter import filedialog, messagebox

try:
    import psutil
    PROCESS = psutil.Process(os.getpid())
except ImportError:
    PROCESS = None


# ============================================================
# SETTINGS
# ============================================================

MAX_SIFT_KEYPOINTS = 800
MAX_FAST_KEYPOINTS = 1200
MAX_ORB_KEYPOINTS = 1000
MAX_MSER_REGIONS = 300
MAX_HARRIS_CORNERS = 1200

MAX_REPEAT_POINTS = 1000
REPEATABILITY_RADIUS = 3.0

HARRIS_BLOCK_SIZE = 2
HARRIS_APERTURE_SIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01

BLUR_SIZE = 3

ALGORITHMS = ["MSER", "SIFT", "FAST", "HARRIS", "ORB"]

# Qualitative algorithm properties, not measured scores.
INVARIANCE = {
    "MSER": ("Partial / region-based", "Limited"),
    "SIFT": ("Strong", "Strong"),
    "FAST": ("Limited", "Limited"),
    "HARRIS": ("Limited", "Limited"),
    "ORB": ("Multi-scale pyramid", "Orientation-aware"),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def strongest_keypoints(keypoints, limit):
    """Return the strongest keypoints without mutating input."""
    return sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True
    )[:limit]


def keypoints_to_points(keypoints, limit=MAX_REPEAT_POINTS):
    """Convert OpenCV keypoints to an Nx2 array."""
    points = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float32
    )

    if len(points) > limit:
        points = points[:limit]

    return points.reshape(-1, 2)


def regions_to_points(regions):
    """Represent each MSER region by its centroid."""
    points = []

    for region in regions:
        if len(region) > 0:
            center = np.mean(region, axis=0)
            points.append(center)

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)

    # Limit tracking work on highly textured frames.
    return points[:MAX_REPEAT_POINTS]


def harris_points(gray):
    """Find Harris corners using local maxima."""
    response = cv2.cornerHarris(
        np.float32(gray),
        HARRIS_BLOCK_SIZE,
        HARRIS_APERTURE_SIZE,
        HARRIS_K
    )

    maximum = float(response.max())

    if not np.isfinite(maximum) or maximum <= 0:
        return response, np.empty((0, 2), dtype=np.float32)

    threshold = HARRIS_THRESHOLD * maximum

    dilated = cv2.dilate(
        response,
        np.ones((3, 3), dtype=np.uint8)
    )

    mask = (response > threshold) & (response == dilated)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return response, np.empty((0, 2), dtype=np.float32)

    scores = response[ys, xs]

    if len(xs) > MAX_HARRIS_CORNERS:
        indices = np.argpartition(
            scores,
            -MAX_HARRIS_CORNERS
        )[-MAX_HARRIS_CORNERS:]

        xs = xs[indices]
        ys = ys[indices]

    points = np.column_stack((xs, ys)).astype(np.float32)

    return response, points


def repeatability_estimate(
    previous_gray,
    current_gray,
    previous_points,
    current_points
):
    """
    Estimate repeatability using optical flow.

    Track previous detections into the current frame and
    check whether a current detection lies within a radius.

    This is a practical video estimate, not a standardized
    repeatability benchmark.
    """
    if (
        previous_gray is None
        or len(previous_points) == 0
        or len(current_points) == 0
    ):
        return None

    old = previous_points.reshape(-1, 1, 2).astype(np.float32)

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        old,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )

    if tracked is None or status is None:
        return None

    tracked = tracked.reshape(-1, 2)
    valid = status.ravel().astype(bool)

    h, w = current_gray.shape

    valid &= (
        (tracked[:, 0] >= 0) &
        (tracked[:, 0] < w) &
        (tracked[:, 1] >= 0) &
        (tracked[:, 1] < h)
    )

    tracked = tracked[valid]

    if len(tracked) == 0:
        return None

    # Compare tracked points with current detections.
    radius_squared = REPEATABILITY_RADIUS ** 2
    repeated = 0

    # Chunk the calculation to control memory usage.
    chunk_size = 100

    for start in range(0, len(tracked), chunk_size):
        chunk = tracked[start:start + chunk_size]

        differences = (
            chunk[:, None, :] -
            current_points[None, :, :]
        )

        distances_squared = np.sum(
            differences * differences,
            axis=2
        )

        repeated += int(
            np.sum(
                np.min(distances_squared, axis=1)
                <= radius_squared
            )
        )

    return 100.0 * repeated / len(tracked)


def format_value(value, digits=2):
    if value is None:
        return "N/A"

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "N/A"
        return f"{value:.{digits}f}"

    return str(value)


# ============================================================
# SELECT VIDEO
# ============================================================

root = tk.Tk()
root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select a video",
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
        ("All files", "*.*")
    ]
)

if not video_path:
    root.destroy()
    print("No video selected.")
    raise SystemExit

output_folder = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    messagebox.showerror(
        "Error",
        "Could not open the selected video."
    )
    cap.release()
    root.destroy()
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

if not np.isfinite(fps) or fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if width <= 0 or height <= 0:
    cap.release()
    root.destroy()
    raise SystemExit("Invalid video dimensions.")


# ============================================================
# INITIALIZE DETECTORS
# ============================================================

mser = cv2.MSER_create()
mser.setDelta(5)
mser.setMinArea(60)
mser.setMaxArea(max(61, (width * height) // 4))

sift = cv2.SIFT_create(
    nfeatures=MAX_SIFT_KEYPOINTS,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6
)

fast = cv2.FastFeatureDetector_create(
    threshold=20,
    nonmaxSuppression=True,
    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
)

orb = cv2.ORB_create(
    nfeatures=MAX_ORB_KEYPOINTS,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=20
)


# ============================================================
# OUTPUT WRITERS
# ============================================================

writers = {}

try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for algorithm in ALGORITHMS:
        path = os.path.join(
            output_folder,
            f"{algorithm.lower()}_output.mp4"
        )

        writer = cv2.VideoWriter(
            path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Could not create output video: {path}"
            )

        writers[algorithm] = writer

except Exception:
    cap.release()

    for writer in writers.values():
        writer.release()

    root.destroy()
    raise


# ============================================================
# BENCHMARK STORAGE
# ============================================================

stats = {
    name: {
        "counts": [],
        "times": [],
        "repeatability": [],
        "previous_points": np.empty(
            (0, 2), dtype=np.float32
        ),
    }
    for name in ALGORITHMS
}

previous_gray = None
frame_number = 0
peak_memory_mb = None

print("\nProcessing video...")
print("Input:", video_path)
print("Output folder:", output_folder)
print("Memory monitoring:", "Enabled" if PROCESS else "Unavailable")


# ============================================================
# PROCESS FRAMES
# ============================================================

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if BLUR_SIZE > 0:
            harris_gray = cv2.GaussianBlur(
                gray,
                (BLUR_SIZE, BLUR_SIZE),
                0
            )
        else:
            harris_gray = gray

        # --------------------------------------------
        # MSER
        # --------------------------------------------

        start = time.perf_counter()

        regions, _ = mser.detectRegions(gray)
        current_points = regions_to_points(regions)
        elapsed = (time.perf_counter() - start) * 1000

        stats["MSER"]["times"].append(elapsed)
        stats["MSER"]["counts"].append(len(regions))

        repeat = repeatability_estimate(
            previous_gray,
            gray,
            stats["MSER"]["previous_points"],
            current_points
        )

        if repeat is not None:
            stats["MSER"]["repeatability"].append(repeat)

        stats["MSER"]["previous_points"] = current_points

        mser_frame = frame.copy()

        regions_to_draw = sorted(
            regions,
            key=len,
            reverse=True
        )[:MAX_MSER_REGIONS]

        for region in regions_to_draw:
            if len(region) < 3:
                continue

            hull = cv2.convexHull(
                np.asarray(region, dtype=np.int32)
            )

            cv2.polylines(
                mser_frame,
                [hull],
                True,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        writers["MSER"].write(mser_frame)

        # --------------------------------------------
        # SIFT
        # --------------------------------------------

        start = time.perf_counter()
        sift_keypoints = sift.detect(gray, None)
        elapsed = (time.perf_counter() - start) * 1000

        stats["SIFT"]["times"].append(elapsed)
        stats["SIFT"]["counts"].append(len(sift_keypoints))

        selected = strongest_keypoints(
            sift_keypoints,
            MAX_SIFT_KEYPOINTS
        )
        current_points = keypoints_to_points(selected)

        repeat = repeatability_estimate(
            previous_gray,
            gray,
            stats["SIFT"]["previous_points"],
            current_points
        )

        if repeat is not None:
            stats["SIFT"]["repeatability"].append(repeat)

        stats["SIFT"]["previous_points"] = current_points

        sift_frame = cv2.drawKeypoints(
            frame,
            selected,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        writers["SIFT"].write(sift_frame)

        # --------------------------------------------
        # FAST
        # --------------------------------------------

        start = time.perf_counter()
        fast_keypoints = fast.detect(gray, None)
        elapsed = (time.perf_counter() - start) * 1000

        stats["FAST"]["times"].append(elapsed)
        stats["FAST"]["counts"].append(len(fast_keypoints))

        selected = strongest_keypoints(
            fast_keypoints,
            MAX_FAST_KEYPOINTS
        )
        current_points = keypoints_to_points(selected)

        repeat = repeatability_estimate(
            previous_gray,
            gray,
            stats["FAST"]["previous_points"],
            current_points
        )

        if repeat is not None:
            stats["FAST"]["repeatability"].append(repeat)

        stats["FAST"]["previous_points"] = current_points

        fast_frame = cv2.drawKeypoints(
            frame,
            selected,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        writers["FAST"].write(fast_frame)

        # --------------------------------------------
        # HARRIS
        # --------------------------------------------

        start = time.perf_counter()
        response, current_points = harris_points(harris_gray)
        elapsed = (time.perf_counter() - start) * 1000

        stats["HARRIS"]["times"].append(elapsed)
        stats["HARRIS"]["counts"].append(len(current_points))

        repeat = repeatability_estimate(
            previous_gray,
            gray,
            stats["HARRIS"]["previous_points"],
            current_points
        )

        if repeat is not None:
            stats["HARRIS"]["repeatability"].append(repeat)

        stats["HARRIS"]["previous_points"] = current_points

        harris_frame = frame.copy()

        for x, y in current_points:
            cv2.circle(
                harris_frame,
                (int(x), int(y)),
                3,
                (0, 0, 255),
                -1,
                cv2.LINE_AA
            )

        writers["HARRIS"].write(harris_frame)

        # --------------------------------------------
        # ORB
        # --------------------------------------------

        start = time.perf_counter()
        orb_keypoints = orb.detect(gray, None)
        elapsed = (time.perf_counter() - start) * 1000

        stats["ORB"]["times"].append(elapsed)
        stats["ORB"]["counts"].append(len(orb_keypoints))

        selected = strongest_keypoints(
            orb_keypoints,
            MAX_ORB_KEYPOINTS
        )
        current_points = keypoints_to_points(selected)

        repeat = repeatability_estimate(
            previous_gray,
            gray,
            stats["ORB"]["previous_points"],
            current_points
        )

        if repeat is not None:
            stats["ORB"]["repeatability"].append(repeat)

        stats["ORB"]["previous_points"] = current_points

        orb_frame = cv2.drawKeypoints(
            frame,
            selected,
            None,
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        writers["ORB"].write(orb_frame)

        # --------------------------------------------
        # MEMORY AND PROGRESS
        # --------------------------------------------

        if PROCESS:
            current_memory = PROCESS.memory_info().rss / (1024 ** 2)

            if peak_memory_mb is None:
                peak_memory_mb = current_memory
            else:
                peak_memory_mb = max(
                    peak_memory_mb,
                    current_memory
                )

        previous_gray = gray.copy()

        if frame_number % 10 == 0 or frame_number == 1:
            if frame_count > 0:
                percent = min(
                    100.0,
                    frame_number * 100 / frame_count
                )
                print(
                    f"\rProcessed {frame_number}/{frame_count} "
                    f"frames ({percent:.1f}%)",
                    end="",
                    flush=True
                )
            else:
                print(
                    f"\rProcessed {frame_number} frames",
                    end="",
                    flush=True
                )

finally:
    cap.release()

    for writer in writers.values():
        writer.release()

    root.destroy()


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

rows = []

for algorithm in ALGORITHMS:
    data = stats[algorithm]
    counts = data["counts"]
    times = data["times"]
    repeats = data["repeatability"]

    average_count = (
        float(np.mean(counts)) if counts else None
    )

    total_count = int(sum(counts)) if counts else 0

    average_time = (
        float(np.mean(times)) if times else None
    )

    median_time = (
        float(np.median(times)) if times else None
    )

    detector_fps = (
        1000.0 / average_time
        if average_time is not None and average_time > 0
        else None
    )

    count_std = (
        float(np.std(counts)) if counts else None
    )

    repeatability = (
        float(np.mean(repeats)) if repeats else None
    )

    rows.append({
        "Algorithm": algorithm,
        "Frames processed": frame_number,
        "Total detections": total_count,
        "Average detections/frame": average_count,
        "Detection count std dev": count_std,
        "Average detection time (ms)": average_time,
        "Median detection time (ms)": median_time,
        "Detector throughput (FPS)": detector_fps,
        "Repeatability estimate (%)": repeatability,
        "Peak process memory (MB)": peak_memory_mb,
        "Scale invariance (qualitative)": INVARIANCE[algorithm][0],
        "Rotation invariance (qualitative)": INVARIANCE[algorithm][1],
        "Output video": os.path.join(
            output_folder,
            f"{algorithm.lower()}_output.mp4"
        )
    })


# ============================================================
# SAVE CSV REPORT
# ============================================================

csv_path = os.path.join(
    output_folder,
    "keypoint_benchmark.csv"
)

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SAVE READABLE TEXT REPORT
# ============================================================

txt_path = os.path.join(
    output_folder,
    "keypoint_benchmark.txt"
)

headers = [
    "Algorithm",
    "Avg points/frame",
    "Avg time (ms)",
    "Median time (ms)",
    "Detector FPS",
    "Repeatability (%)",
    "Peak memory (MB)"
]

table_rows = []

for row in rows:
    table_rows.append([
        row["Algorithm"],
        format_value(row["Average detections/frame"]),
        format_value(row["Average detection time (ms)"]),
        format_value(row["Median detection time (ms)"]),
        format_value(row["Detector throughput (FPS)"]),
        format_value(row["Repeatability estimate (%)"]),
        format_value(row["Peak process memory (MB)"])
    ])

widths = [
    max(
        len(headers[i]),
        max(len(r[i]) for r in table_rows)
    )
    for i in range(len(headers))
]

def format_table_line(values):
    return " | ".join(
        str(value).ljust(widths[i])
        for i, value in enumerate(values)
    )

with open(txt_path, "w", encoding="utf-8") as file:
    file.write("VIDEO KEYPOINT ALGORITHM BENCHMARK\n")
    file.write("=" * 90 + "\n")
    file.write(f"Input video: {video_path}\n")
    file.write(f"Frames processed: {frame_number}\n")
    file.write(f"Resolution: {width} x {height}\n")
    file.write(f"Input FPS: {fps:.2f}\n\n")

    file.write(format_table_line(headers) + "\n")
    file.write("-+-".join("-" * w for w in widths) + "\n")

    for row in table_rows:
        file.write(format_table_line(row) + "\n")

    file.write("\nAdditional metrics:\n\n")

    for row in rows:
        file.write(f"\n{row['Algorithm']}\n")
        for key, value in row.items():
            if key != "Algorithm":
                file.write(f"  {key}: {value}\n")

    file.write(
        "\nNotes:\n"
        "- Repeatability is estimated across consecutive frames "
        "using optical flow and a 3-pixel matching radius.\n"
        "- Detector FPS is calculated from average detection time; "
        "it is not total video-processing FPS.\n"
        "- Peak memory is process-wide sampled resident memory, "
        "not memory attributable to one algorithm.\n"
        "- Scale and rotation properties are qualitative, not "
        "controlled invariance measurements.\n"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n\nBENCHMARK RESULTS\n")

print(format_table_line(headers))
print("-+-".join("-" * w for w in widths))

for row in table_rows:
    print(format_table_line(row))

print("\nSaved files:")
print("CSV table:", csv_path)
print("Text report:", txt_path)

for algorithm in ALGORITHMS:
    print(
        f"{algorithm} video:",
        os.path.join(
            output_folder,
            f"{algorithm.lower()}_output.mp4"
        )
    )

print("\nBenchmark completed.")
