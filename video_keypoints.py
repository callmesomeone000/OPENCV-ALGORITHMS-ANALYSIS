
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox


# ==================================================
# SETTINGS
# ==================================================

MAX_SIFT_KEYPOINTS = 800
MAX_FAST_KEYPOINTS = 1200
MAX_ORB_KEYPOINTS = 1000
MAX_MSER_REGIONS = 300
MAX_HARRIS_CORNERS = 1200

HARRIS_BLOCK_SIZE = 2
HARRIS_APERTURE_SIZE = 3
HARRIS_K = 0.04
HARRIS_THRESHOLD = 0.01

# Set to 3 for light smoothing, or 0 to disable.
BLUR_SIZE = 3

ALGORITHMS = ["MSER", "SIFT", "FAST", "HARRIS", "ORB"]


# ==================================================
# SELECT VIDEO
# ==================================================

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
    print("No video selected. Exiting.")
    raise SystemExit

output_folder = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    messagebox.showerror(
        "Video Error",
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


# ==================================================
# CREATE OUTPUT VIDEO WRITERS
# ==================================================

writers = {}

try:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for algorithm in ALGORITHMS:
        output_path = os.path.join(
            output_folder,
            f"{algorithm.lower()}_output.mp4"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Could not create output video:\n{output_path}"
            )

        writers[algorithm] = writer

except Exception as error:
    cap.release()

    for writer in writers.values():
        writer.release()

    root.destroy()
    raise SystemExit(str(error))


# ==================================================
# INITIALIZE DETECTORS
# ==================================================

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


# ==================================================
# HELPER FUNCTION
# ==================================================

def strongest_keypoints(keypoints, limit):
    """
    Return the strongest keypoints first.
    Uses sorted() to avoid modifying an OpenCV sequence.
    """
    keypoints = sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True
    )

    return keypoints[:limit]


# ==================================================
# PROCESS VIDEO
# ==================================================

frame_number = 0

print("\nProcessing video...")
print("Input video:", video_path)
print("Output folder:", output_folder)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # Convert to grayscale once per frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Light smoothing, used for Harris corner detection.
        if BLUR_SIZE > 0:
            harris_gray = cv2.GaussianBlur(
                gray,
                (BLUR_SIZE, BLUR_SIZE),
                0
            )
        else:
            harris_gray = gray

        # ------------------------------------------
        # 1. MSER: REGION DETECTION
        # ------------------------------------------

        mser_frame = frame.copy()

        regions, _ = mser.detectRegions(gray)

        # Limit regions drawn to reduce visual clutter.
        regions = sorted(
            regions,
            key=lambda region: len(region),
            reverse=True
        )[:MAX_MSER_REGIONS]

        for region in regions:
            if len(region) < 3:
                continue

            points = np.asarray(region, dtype=np.int32)
            hull = cv2.convexHull(points)

            if len(hull) >= 3:
                cv2.polylines(
                    mser_frame,
                    [hull],
                    True,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        writers["MSER"].write(mser_frame)

        # ------------------------------------------
        # 2. SIFT: SCALE-AWARE KEYPOINTS
        # ------------------------------------------

        sift_frame = frame.copy()

        sift_keypoints = sift.detect(gray, None)

        sift_keypoints = strongest_keypoints(
            sift_keypoints,
            MAX_SIFT_KEYPOINTS
        )

        cv2.drawKeypoints(
            frame,
            sift_keypoints,
            sift_frame,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        writers["SIFT"].write(sift_frame)

        # ------------------------------------------
        # 3. FAST: CORNER DETECTION
        # ------------------------------------------

        fast_frame = frame.copy()

        fast_keypoints = fast.detect(gray, None)

        fast_keypoints = strongest_keypoints(
            fast_keypoints,
            MAX_FAST_KEYPOINTS
        )

        cv2.drawKeypoints(
            frame,
            fast_keypoints,
            fast_frame,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
        )

        writers["FAST"].write(fast_frame)

        # ------------------------------------------
        # 4. HARRIS: CORNER DETECTION
        # ------------------------------------------

        harris_frame = frame.copy()

        gray_float = np.float32(harris_gray)

        response = cv2.cornerHarris(
            gray_float,
            HARRIS_BLOCK_SIZE,
            HARRIS_APERTURE_SIZE,
            HARRIS_K
        )

        max_response = float(response.max())

        if max_response > 0:
            threshold = HARRIS_THRESHOLD * max_response

            # Find local maxima to avoid marking every
            # neighboring pixel around the same corner.
            dilated = cv2.dilate(
                response,
                np.ones((3, 3), dtype=np.uint8)
            )

            corner_mask = (
                (response > threshold) &
                (response == dilated)
            )

            ys, xs = np.where(corner_mask)

            # Retain the strongest Harris responses.
            if len(xs) > MAX_HARRIS_CORNERS:
                scores = response[ys, xs]

                indices = np.argpartition(
                    scores,
                    -MAX_HARRIS_CORNERS
                )[-MAX_HARRIS_CORNERS:]

                xs = xs[indices]
                ys = ys[indices]

            for x, y in zip(xs, ys):
                cv2.circle(
                    harris_frame,
                    (int(x), int(y)),
                    3,
                    (0, 0, 255),
                    -1,
                    cv2.LINE_AA
                )

        writers["HARRIS"].write(harris_frame)

        # ------------------------------------------
        # 5. ORB: ORIENTED KEYPOINTS
        # ------------------------------------------

        orb_frame = frame.copy()

        orb_keypoints = orb.detect(gray, None)

        orb_keypoints = strongest_keypoints(
            orb_keypoints,
            MAX_ORB_KEYPOINTS
        )

        cv2.drawKeypoints(
            frame,
            orb_keypoints,
            orb_frame,
            color=(255, 0, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        writers["ORB"].write(orb_frame)

        # ------------------------------------------
        # PROGRESS
        # ------------------------------------------

        if frame_number % 10 == 0 or frame_number == 1:
            if frame_count > 0:
                percent = min(
                    100.0,
                    frame_number * 100.0 / frame_count
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


# ==================================================
# RESULTS
# ==================================================

print("\n\nProcessing finished.")
print("Frames processed:", frame_number)
print("\nOutput videos:")

for algorithm in ALGORITHMS:
    path = os.path.join(
        output_folder,
        f"{algorithm.lower()}_output.mp4"
    )

    if os.path.exists(path):
        print(f"{algorithm}: {path}")
    else:
        print(f"WARNING: Output file not found: {path}")

print("\nAll output videos are saved in the script folder.")
