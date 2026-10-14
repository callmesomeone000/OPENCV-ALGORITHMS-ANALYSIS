# OpenCV Algorithms Analysis

A simple benchmark and visualization toolkit for feature-detection algorithms (implemented in Python). It measures detection and descriptor timings, keypoint statistics and repeatability on images, and produces per-algorithm videos and reports for input videos.
 
## Stack
- Language: Python
- Notable libraries: OpenCV (cv2), NumPy, pandas, Matplotlib

## What it does
- Runs and compares 5 feature-detection algorithms: ORB, SIFT, FAST, Harris, MSER.
- For images: benchmarks detection and descriptor timings, keypoint statistics, repeatability across transformations, and saves visualizations and summary graphs.
- For video: runs detectors frame-by-frame, writes per-algorithm annotated output videos, and saves CSV / text reports with throughput and repeatability estimates.

## Files of interest
- `feature_detection.py` — image-based benchmark and visualization. Select images with the file dialog or pass image paths on the command line.
- `video_keypoints.py` — processes a selected video, produces per-algorithm output videos and tabular/text reports.

## Installation
Create a Python 3.x virtual environment and install the dependencies. For full algorithm support (SIFT) install the contrib OpenCV package.

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows use .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas matplotlib opencv-contrib-python
# optional (faster memory monitoring / Excel export):
pip install psutil openpyxl
```

Note: If you prefer a headless environment (no GUI), use `opencv-contrib-python-headless` but ensure you still have the features you need.

## Usage

Image benchmark (feature_detection.py)
- Run with a file picker:

```bash
python feature_detection.py
```

- Or pass image paths on the command line to skip the picker:

```bash
python feature_detection.py images/cat.jpg images/scene.png
```

What it produces (in `feature_benchmark_results/`):
- `complete_benchmark_results.csv` — full results table
- `complete_benchmark_results.xlsx` — Excel export (requires openpyxl)
- `algorithm_summary.csv` — aggregated per-algorithm summary
- `repeatability_by_transformation.csv` — repeatability table
- `visualizations/` — images with keypoints drawn (one per algorithm)
- `graphs/` — summary plots (PNG)

Video benchmark (video_keypoints.py)
- Run and choose a video with the file dialog:

```bash
python video_keypoints.py
```

What it produces (in the script directory):
- Per-algorithm output videos: `sift_output.mp4`, `orb_output.mp4`, etc.
- `keypoint_benchmark.csv` — per-algorithm numeric summary
- `keypoint_benchmark.txt` — human-readable report with additional metrics

## Configuration
Both scripts expose tuning constants at the top of the file (e.g., MAX_KEYPOINTS, RUNS, thresholds). Edit those constants to change limits, timing repetitions, or detection thresholds.

Key constants you may want to change:
- feature_detection.py: RUNS, MAX_KEYPOINTS, MAX_DIM, MATCH_THRESHOLD
- video_keypoints.py: MAX_*_KEYPOINTS, REPEATABILITY_RADIUS, BLUR_SIZE

## Notes & troubleshooting
- SIFT: depending on your OpenCV build, SIFT may require the `opencv-contrib-python` package. If SIFT creation fails, install the contrib package.
- GUI file pickers: both scripts use tkinter for file selection. In headless servers you must pass file paths (feature_detection.py) or provide a virtual display / modify the code to accept CLI args for video path.
- Excel export in `feature_detection.py` requires `openpyxl`. If not installed, the script will skip the Excel export and print a notice.

## Contributing
This repo currently has no CONTRIBUTING file. If you plan to extend it:
- Add tests for data-processing functions (pandas outputs, repeatability calculation).
- Consider extracting common detector builders and utilities into an importable module for reuse.

## License
No license is provided in this repository. Add a LICENSE file if you want to permit reuse.

---

Generated README based on the Python scripts in this repository (feature_detection.py and video_keypoints.py).
