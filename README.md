# 🔍 OpenCV Algorithms Analysis

**A Python-based computer vision project for detecting, visualizing, and benchmarking image features in photos and videos.**
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Excel-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white" alt="Excel"/>
</p>
This project compares five popular feature detection algorithms using OpenCV. It analyses static images and video footage, highlights detected keypoints, measures performance, and generates reports to help understand how different algorithms behave on the same input.

## ✨ Features

- **Five feature detection algorithms:** ORB, SIFT, FAST, Harris Corner, and MSER.
- **Photo analysis:** Detects and visualizes keypoints in images.
- **Video analysis:** Processes videos frame by frame and generates separate annotated output videos for each algorithm.
- **Performance benchmarking:** Measures detection time, descriptor computation time, and other available metrics.
- **Keypoint statistics:** Compares the number and distribution of detected features.
- **Repeatability analysis:** Evaluates how consistently features are detected under different image transformations.
- **Visual comparisons:** Generates annotated images, graphs, and summary reports.
- **Data export:** Saves benchmark results in CSV and, when supported, Excel format.

## 🧠 Algorithms Compared

| Algorithm | Description |
|---|---|
| **ORB** | A fast, rotation-aware feature detector and descriptor designed for efficient computer vision applications. |
| **SIFT** | Detects distinctive features that are robust to changes in scale and rotation. |
| **FAST** | A corner detector designed for speed, making it useful for real-time applications. |
| **Harris Corner** | Detects corners using changes in image intensity in different directions. |
| **MSER** | Detects maximally stable regions that remain relatively consistent across intensity thresholds. |

Each algorithm detects features differently, so the number, location, and appearance of the detected keypoints can vary significantly.

## 🛠️ Tech Stack

- **Python** — Main programming language
- **OpenCV** — Image processing and feature detection
- **NumPy** — Numerical computations
- **Pandas** — Benchmark data handling
- **Matplotlib** — Graphs and visualizations
- **Tkinter** — File selection dialogs
- **openpyxl** — Optional Excel report export
- **psutil** — Optional memory monitoring

## 📁 Project Structure

```text
OPENCV-ALGORITHMS-ANALYSIS/
│
├── feature_detection.py
├── video_keypoints.py
└── README.md
```

### Main Scripts

**`feature_detection.py`**

Used for analysing photos and comparing the five algorithms. It generates images with detected keypoints, benchmark tables, repeatability results, and summary graphs.

**`video_keypoints.py`**

Used for analysing video files. It processes the video frame by frame, applies the feature detection algorithms, and generates separate annotated videos and benchmark reports.

## 🚀 Getting Started

### 1. Clone the repository

Make sure Git and Python are installed on your computer.

Open your terminal or VS Code terminal and run:

```bash
git clone https://github.com/callmesomeone000/OPENCV-ALGORITHMS-ANALYSIS.git
```

Navigate into the project directory:

```bash
cd OPENCV-ALGORITHMS-ANALYSIS
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade pip and install the required libraries:

```bash
python -m pip install --upgrade pip
pip install numpy pandas matplotlib opencv-contrib-python
```

For optional Excel export and memory monitoring:

```bash
pip install openpyxl psutil
```

**Note:** Use `opencv-contrib-python` for the OpenCV features needed by this project, including SIFT support in compatible builds. Avoid installing multiple OpenCV wheel variants in the same environment.

## 📸 Analysing Photos

Run the image feature detection script:

```bash
python feature_detection.py
```

A file selection dialog allows you to choose an image for analysis.

You can also provide image paths directly:

```bash
python feature_detection.py images/cat.jpg images/scene.png
```

### Output

The script generates a `feature_benchmark_results/` directory containing available results such as:

- `complete_benchmark_results.csv` — Detailed benchmark results
- `complete_benchmark_results.xlsx` — Excel version of the results, when `openpyxl` is installed
- `algorithm_summary.csv` — Summary of algorithm performance
- `repeatability_by_transformation.csv` — Feature repeatability results
- `visualizations/` — Images with detected keypoints
- `graphs/` — Benchmark and comparison plots

## 🎥 Analysing Videos

Run the video analysis script:

```bash
python video_keypoints.py
```

Select the video you want to analyse using the file selection dialog.

The program processes the video frame by frame and applies the feature detection algorithms to its frames.

### Output

The script generates separate annotated videos for the algorithms, along with benchmark reports. Output filenames include:

```text
sift_output.mp4
orb_output.mp4
fast_output.mp4
harris_output.mp4
mser_output.mp4
```

It also generates:

- `keypoint_benchmark.csv` — Numerical benchmark summary
- `keypoint_benchmark.txt` — Human-readable performance report

The output videos allow you to compare how each algorithm detects and tracks the visual locations of features across successive frames.

**Important:** This project performs feature/keypoint detection, not general-purpose object recognition. Detected keypoints represent distinctive image locations or regions; they do not automatically identify objects such as people, cars, or animals.

## 📊 What Can Be Compared?

The project helps investigate several aspects of feature detection:

- **Detection speed:** How long an algorithm takes to detect features.
- **Keypoint count:** How many features each algorithm detects.
- **Visual distribution:** Where detected features appear in an image or video frame.
- **Descriptor performance:** Descriptor computation time and related statistics, where applicable.
- **Repeatability:** How consistently features are detected after image transformations.
- **Video processing performance:** How efficiently the algorithms process video frames.

The results depend on the input, image quality, algorithm parameters, hardware, and OpenCV configuration. A higher keypoint count does not necessarily mean an algorithm is better.

## ⚙️ Customization

Both Python scripts include configurable parameters near the top of the code.

Examples include:

- `RUNS` — Number of benchmark repetitions.
- `MAX_KEYPOINTS` — Limits on the number of keypoints.
- `MAX_DIM` — Image dimension limit.
- `MATCH_THRESHOLD` — Threshold used in matching-related analysis.
- `REPEATABILITY_RADIUS` — Distance tolerance for repeatability comparisons.
- `BLUR_SIZE` — Blur configuration used during video processing.

Change the relevant settings to experiment with detection limits, processing speed, and benchmark behavior.

## 🐛 Troubleshooting

**SIFT is not working**

Ensure that the correct OpenCV package is installed:

```bash
pip install opencv-contrib-python
```

**The file selection dialog does not open**

Make sure your Python installation supports Tkinter and that you are running the script in a desktop environment.

**Excel results are missing**

Install the optional Excel dependency:

```bash
pip install openpyxl
```

**The video cannot be opened**

Check that the video file is valid and supported by your OpenCV installation. Video codec availability can affect reading and writing.

**The output contains too many or too few keypoints**

Review the algorithm-specific parameters and test the same input with different settings.

## 🎯 Project Objective

The goal of this project is to explore how classical computer vision feature detection algorithms differ in their detection behavior, computational performance, and repeatability.

By applying the same algorithms to both photos and videos, the project provides a practical way to study their strengths, limitations, and potential applications in computer vision.

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [OpenCV Feature Detection and Description](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)


---

If you find this project useful, consider giving the repository a ⭐ on GitHub!
