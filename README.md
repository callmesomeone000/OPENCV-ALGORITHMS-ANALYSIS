# 🔍 OpenCV Algorithms Analysis

# 🔬 2D Feature Detector Comparison

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Excel-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white" alt="Excel"/>
</p>

<p align="center">
  <strong>A quantitative comparison of five classical computer vision feature detectors across multiple views of the same object.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Feature%20Detection-2D-blue?style=flat-square" alt="2D Feature Detection"/>
  <img src="https://img.shields.io/badge/Algorithms-5-purple?style=flat-square" alt="Five Algorithms"/>
  <img src="https://img.shields.io/badge/Analysis-Automated-success?style=flat-square" alt="Automated Analysis"/>
  <img src="https://img.shields.io/badge/Reports-CSV%20%7C%20Excel-orange?style=flat-square" alt="CSV and Excel Reports"/>
</p>

---

## 📌 Overview

**2D Feature Detector Comparison** is a Python-based computer vision benchmarking tool designed to evaluate five feature detection algorithms using a collection of images of the same object.

Instead of requiring a 3D model, the program accepts ordinary 2D images and analyses the detected keypoints, spatial distribution, computational cost, and cross-view matching performance of each algorithm.

The objective is to understand how different detectors perform under a common evaluation pipeline, using quantitative metrics, visualisations, CSV datasets, and Excel reports.

### ⚡ Algorithms Compared

| Algorithm | Full Name | Primary Characteristics |
|---|---|---|
| 🔵 **MSER** | Maximally Stable Extremal Regions | Detects stable intensity regions; represented by region centroids in this implementation. |
| 🟢 **SIFT** | Scale-Invariant Feature Transform | Detects distinctive features with scale and orientation information. |
| 🟠 **FAST** | Features from Accelerated Segment Test | Designed for fast corner detection. |
| 🟡 **Harris** | Harris Corner Detector | Identifies corners using local image-gradient information. |
| 🟣 **ORB** | Oriented FAST and Rotated BRIEF | Combines FAST-based detection with oriented binary descriptors in its native implementation. |

---

## ✨ Key Features

- **Five-algorithm comparison** — evaluates MSER, SIFT, FAST, Harris, and ORB on the same image collection.
- **Multiple-view analysis** — evaluates every unordered pair of successfully processed images.
- **Standardised keypoint selection** — ranks responses, applies spatial suppression, and enforces a common maximum keypoint budget.
- **Descriptor-based matching** — uses a shared SIFT descriptor extractor with a fixed keypoint size, upright orientation, and octave for cross-detector comparisons.
- **Geometric verification** — uses mutual ratio-test matches and fundamental-matrix RANSAC to estimate geometrically consistent matches.
- **Ground-truth repeatability** — optionally evaluates keypoint repeatability using supplied homographies.
- **Spatial distribution metrics** — measures 2D coverage, feature density, and mean nearest-neighbour distance.
- **Performance benchmarking** — separates detector execution time from Python keypoint-conversion time.
- **Failure-aware statistics** — reports pair success rates, failure-as-zero matching metrics, and success-only statistics.
- **Object-mask support** — optionally restricts keypoint selection and spatial metrics to a specified object region.
- **Automated reporting** — generates marked images, CSV files, an Excel workbook, and comparison charts.

---

## 🧠 How It Works

<p align="center">
  <img src="https://img.shields.io/badge/01-Load%20Images-2563EB?style=for-the-badge" alt="Step 1"/>
  <img src="https://img.shields.io/badge/02-Detect%20Features-7C3AED?style=for-the-badge" alt="Step 2"/>
  <img src="https://img.shields.io/badge/03-Select%20Keypoints-0891B2?style=for-the-badge" alt="Step 3"/>
  <img src="https://img.shields.io/badge/04-Evaluate-059669?style=for-the-badge" alt="Step 4"/>
  <img src="https://img.shields.io/badge/05-Generate%20Reports-EA580C?style=for-the-badge" alt="Step 5"/>
</p>

### 1. Image Loading and Preprocessing

- Loads supported image formats from the selected folder.
- Optionally resizes images so their longest side does not exceed 1,280 pixels.
- Converts images to grayscale for feature detection.
- Loads optional object masks when available.

### 2. Feature Detection

Each algorithm independently detects keypoints in every image. The program performs one warm-up detection call followed by five timed calls and records the mean execution time.

MSER regions are converted into centroid keypoints, while Harris corners are extracted from local maxima in the corner-response map.

### 3. Keypoint Selection

To reduce excessive clustering and make comparisons more consistent, the pipeline applies the following operations:

1. Rank detected keypoints by response.
2. Filter keypoints against the object mask, if available.
3. Apply greedy spatial suppression using a 3-pixel radius.
4. Retain at most 1,000 keypoints per algorithm per image.

The budget is a maximum, not a guarantee that every algorithm will produce exactly 1,000 keypoints.

### 4. Cross-View Matching

For each algorithm, the program compares every unordered pair of processed images.

- Computes SIFT descriptors at the selected keypoint locations.
- Applies a mutual nearest-neighbour ratio test.
- Requires at least 15 mutual matches before attempting geometric verification.
- Estimates a fundamental matrix using RANSAC.
- Counts geometric inliers and classifies a pair as successful when it has at least 12 inliers.

This provides a **cross-view matching proxy**, not a direct measurement of true feature repeatability.

### 5. Ground-Truth Repeatability

When a valid ground-truth homography is provided for an image pair, the program projects keypoints between the views and evaluates mutual nearest-neighbour correspondence within a 3-pixel threshold.

Homographies are converted to account for image resizing.

This evaluation is intended for planar scenes or other imaging configurations where a homography correctly describes the transformation.

### 6. Reporting and Visualisation

The program saves colour-coded keypoint images, per-image metrics, pairwise matching statistics, a summary comparison table, and charts for visual comparison.

---

## 📊 Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Detection Core Time | Time spent inside the detector's core operation. |
| Keypoint Conversion Time | Additional Python-side processing time, especially relevant to MSER and Harris. |
| Detection Total Time | Core detection time plus keypoint-conversion time. |
| Native Descriptor Time | Descriptor computation time for SIFT and ORB on selected keypoints. |
| Distinct Keypoints Available | Number remaining after spatial suppression, before the final cap. |
| Suppression Rate | Percentage of in-region raw keypoints removed by spatial suppression. |
| Feature Density | Selected keypoints per megapixel of the evaluated region. |
| 2D Coverage | Percentage of the evaluated region covered by keypoint neighbourhoods. |
| Mean NN Distance | Average distance from each selected keypoint to its nearest neighbour. |
| Normalised NN Distance | Mean nearest-neighbour distance divided by the square root of region area, expressed as a percentage. |
| Mutual Ratio-Test Matches | Matches passing the ratio test in both directions. |
| RANSAC Inliers | Matches consistent with the estimated fundamental matrix. |
| Pair Success Rate | Percentage of image pairs meeting the minimum inlier requirement. |
| Match Rate — Failures as Zero | Inlier-based match-rate proxy, with unsuccessful pairs assigned zero. |
| Success-Only Match Rate | Mean match-rate proxy among successful pairs only. |
| Homography Repeatability | Percentage of eligible keypoints that have a mutual correspondence within the ground-truth reprojection threshold. |

**Important:** These metrics measure different properties. A detector that produces more keypoints or runs faster is not automatically more repeatable or more suitable for every application.

---

## 🗂️ Project Structure

The following structure is produced by the program:

```text
project/
│
├── feature_detector_comparison_v3.py
│
├── images/
│   ├── view_01.jpg
│   ├── view_02.jpg
│   ├── view_03.jpg
│   ├── ...
│   ├── view_20.jpg
│   │
│   ├── masks/                         # Optional
│   │   ├── view_01.png
│   │   ├── view_02.png
│   │   └── ...
│   │
│   ├── ground_truth_homographies.csv  # Optional
│   │
│   └── feature_detection_analysis_2d_v3/
│       ├── marked_images/
│       ├── per_image_metrics.csv
│       ├── comparison_table.csv
│       ├── cross_view_matching.csv
│       ├── comparison_table.xlsx
│       ├── core_detection_time.png
│       ├── total_detection_time.png
│       ├── final_keypoints.png
│       ├── suppression_rate.png
│       ├── pair_success_rate.png
│       ├── match_rate_failures_zero.png
│       ├── homography_repeatability.png
│       ├── nn_distance.png
│       └── coverage.png
```

The image filenames above are illustrative. The program uses the actual filenames in the input folder.

---

## ⚙️ Installation

### Requirements

- Python 3.9 or later recommended.
- OpenCV with SIFT support.
- NumPy
- Pandas
- Matplotlib
- OpenPyXL

Install the dependencies:

```bash
pip install opencv-python numpy pandas matplotlib openpyxl
```

Clone the repository or place the Python script in your working directory.

---

## 🚀 Usage

### Option 1 — Interactive Mode

Run the script without arguments to open a folder-selection dialog.

```bash
python feature_detector_comparison_v3.py
```

Select the folder containing your images. If the folder does not contain exactly 20 images, the program asks whether you want to continue.

### Option 2 — Command-Line Mode

Specify the input folder directly:

```bash
python feature_detector_comparison_v3.py "C:\path\to\your\images"
```

This mode runs without the folder picker, image-count confirmation, or completion dialog.

The script accepts supported formats including JPG, JPEG, PNG, BMP, TIFF, and WebP.

---

## 🖼️ Output Files

### Marked Images

The `marked_images` directory contains one annotated image per input image and algorithm.

Example filenames:

```text
view_01_mser_keypoints.jpg
view_01_sift_keypoints.jpg
view_01_fast_keypoints.jpg
view_01_harris_keypoints.jpg
view_01_orb_keypoints.jpg
```

These visualisations help inspect the spatial distribution of selected keypoints.

### CSV Reports

| File | Contents |
|---|---|
| `per_image_metrics.csv` | Detection, timing, keypoint, density, coverage, and descriptor statistics for each image and algorithm. |
| `comparison_table.csv` | Aggregated comparison metrics for all five algorithms. |
| `cross_view_matching.csv` | Matching, RANSAC, pair success, and optional homography repeatability for each image pair. |

### Excel Workbook

`comparison_table.xlsx` contains four worksheets:

- **Comparison** — aggregated algorithm-level results.
- **Per Image** — individual image metrics.
- **Cross-view Pairs** — pairwise matching results.
- **Methodology** — experiment settings and evaluation rules.

The workbook includes formatted headers, filters, frozen header rows, and adjusted column widths.

### Charts

The generated charts cover detection time, final keypoint counts, suppression rate, pair success, matching rate, homography repeatability, nearest-neighbour distance, and spatial coverage.

Charts are saved as PNG files for easy insertion into reports, presentations, and research documents.

---

## 🎭 Optional Object Masks

To evaluate features only within the object, create a `masks` folder inside the input image directory.

Each mask must:

- Have the same filename stem as its corresponding image.
- Use the PNG format.
- Mark the object with non-zero pixel values.
- Mark the background with zero-valued pixels.

Example:

```text
images/
├── model_front.jpg
├── model_side.jpg
└── masks/
    ├── model_front.png
    └── model_side.png
```

Masks are resized to match the processed image dimensions using nearest-neighbour interpolation.

When a valid mask exists, it restricts keypoint selection and the spatial metrics to the object region. If no valid mask is available, the full image is used.

---

## 📐 Optional Ground-Truth Homographies

For a more direct evaluation of feature repeatability, provide a file named:

`ground_truth_homographies.csv`

It must contain these columns:

```csv
Image_A,Image_B,H00,H01,H02,H10,H11,H12,H20,H21,H22
```

Each row represents a 3 × 3 homography mapping image A coordinates to image B coordinates in the **original, unresized images**.

The matrix is interpreted as:

\[
\begin{bmatrix}
x'\\y'\\w'
\end{bmatrix}
=
H
\begin{bmatrix}
x\\y\\1
\end{bmatrix}
\]

The transformed pixel coordinates are obtained by dividing the first two components by \(w'\).

Provide accurate, invertible homographies for the relevant image pairs. The program can also use the inverse of a supplied reverse-direction homography.

**Applicability:** A single homography is appropriate for planar scenes or pure camera rotation under the usual pinhole-camera assumptions. It does not generally describe arbitrary viewpoint changes of a non-planar object.

---

## 🔬 Experimental Settings

The current default configuration includes:

| Parameter | Default |
|---|---:|
| Maximum image side | 1,280 px |
| Maximum selected keypoints | 1,000 per algorithm per image |
| Spatial suppression radius | 3 px |
| Coverage radius | 5 px |
| Timing runs | 5, after one warm-up |
| Matching descriptor | SIFT |
| Fixed descriptor keypoint size | 16 px |
| Matching orientation | Upright, 0° |
| Matching octave | 0 |
| Lowe ratio threshold | 0.75 |
| Minimum mutual matches | 15 |
| Minimum RANSAC inliers for success | 12 |
| Fundamental-matrix RANSAC threshold | 1.5 px |
| Homography correspondence threshold | 3 px |
| RANSAC seed | 12345 |

Settings can be modified at the beginning of the script.

### Descriptor Policy

The default `MATCH_DESCRIPTOR_POLICY = "fixed"` gives all detectors the same descriptor geometry at their selected locations. This helps isolate differences in detected locations from differences caused by native scale and orientation assignments.

Setting the policy to `"native"` provides a sensitivity comparison using the original keypoint metadata where supported.

The fixed policy does not make every aspect of the detectors identical; their detection mechanisms, response values, and available keypoint locations remain different.

---

## 🛠️ Technology Stack

<p align="center">
  <a href="https://www.python.org/"><img src="https://skillicons.dev/icons?i=python" alt="Python" width="55"/></a>
  &nbsp;&nbsp;
  <a href="https://opencv.org/"><img src="https://cdn.simpleicons.org/opencv/5C3EE8" alt="OpenCV" width="55"/></a>
  &nbsp;&nbsp;
  <a href="https://numpy.org/"><img src="https://cdn.simpleicons.org/numpy/013243" alt="NumPy" width="55"/></a>
  &nbsp;&nbsp;
  <a href="https://pandas.pydata.org/"><img src="https://cdn.simpleicons.org/pandas/150458" alt="Pandas" width="55"/></a>
  &nbsp;&nbsp;
  <a href="https://matplotlib.org/"><img src="https://cdn.simpleicons.org/plotly/3F4F75" alt="Plotting" width="55"/></a>
  &nbsp;&nbsp;
 <a href="https://www.microsoft.com/microsoft-365/excel"> <img src="https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoftexcel&logoColor=white" alt="Microsoft Excel"/></a>
</p>

| Technology | Purpose |
|---|---|
| Python | Experiment pipeline and automation |
| OpenCV | Feature detection, descriptors, geometric verification, and image processing |
| NumPy | Numerical operations and coordinate transformations |
| Pandas | Tabular data processing and CSV reporting |
| Matplotlib | Metric visualisation |
| OpenPyXL | Excel workbook generation and formatting |



<p align="center">
  <strong>Detect. Measure. Compare.</strong>
  <br/>
  <sub>A reproducible starting point for quantitative 2D feature detector evaluation.</sub>
</p>
