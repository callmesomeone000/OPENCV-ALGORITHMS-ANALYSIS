import cv2
import numpy as np
import pandas as pd
import time
import os




IMAGE_PATH = "closeup-cat-ground.jpg"

OUTPUT_FOLDER = "feature_detection_results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)



#LOAD IMAGE


img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

image_area = height * width



# HEADER

print("=" * 80)
print("FEATURE DETECTION ANALYSIS")
print("=" * 80)

print(f"Image: {IMAGE_PATH}")
print(f"Image size: {width} x {height}")
print(f"Total pixels: {image_area:,}")
print()


#FROM HERE WE ARE DEFINING THE FEATURE DETECTION ALGORITHM, LIKE THE ANALYSIS THAT THEY WILL DO, AND THE TIME IT TAKES TO RUN THEM. WE WILL ALSO BE STORING THE RESULTS IN A DATAFRAME FOR LATER ANALYSIS.

def harris_detector(gray):

    start = time.perf_counter()

    gray_float = np.float32(gray)

    response = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    threshold = 0.01 * response.max()

    ys, xs = np.where(
        response > threshold
    )

    keypoints = []

    for x, y in zip(xs, ys):

        keypoints.append(
            cv2.KeyPoint(
                float(x),
                float(y),
                3
            )
        )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms




def shi_tomasi_detector(gray):

    start = time.perf_counter()

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10
    )

    if corners is None:

        keypoints = []

    else:

        keypoints = []

        for corner in corners:

            x = corner[0][0]
            y = corner[0][1]

            keypoints.append(
                cv2.KeyPoint(
                    float(x),
                    float(y),
                    3
                )
            )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms




def fast_detector(gray):

    detector = cv2.FastFeatureDetector_create()

    start = time.perf_counter()

    keypoints = detector.detect(
        gray,
        None
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms




def sift_detector(gray):

    detector = cv2.SIFT_create()

    start = time.perf_counter()

    keypoints, descriptors = detector.detectAndCompute(
        gray,
        None
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, descriptors, time_ms




def orb_detector(gray):

    detector = cv2.ORB_create(
        nfeatures=1000
    )

    start = time.perf_counter()

    keypoints, descriptors = detector.detectAndCompute(
        gray,
        None
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, descriptors, time_ms




def akaze_detector(gray):

    detector = cv2.AKAZE_create()

    start = time.perf_counter()

    keypoints, descriptors = detector.detectAndCompute(
        gray,
        None
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, descriptors, time_ms




def brisk_detector(gray):

    detector = cv2.BRISK_create()

    start = time.perf_counter()

    keypoints, descriptors = detector.detectAndCompute(
        gray,
        None
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, descriptors, time_ms




def mser_detector(gray):

    detector = cv2.MSER_create()

    start = time.perf_counter()

    regions, bounding_boxes = detector.detectRegions(
        gray
    )

    keypoints = []

    for region in regions:

        x, y, w, h = cv2.boundingRect(
            region
        )

        size = max(w, h)

        keypoints.append(
            cv2.KeyPoint(
                float(x + w / 2),
                float(y + h / 2),
                float(size)
            )
        )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms




def blob_detector(gray):

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 5000

    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(
        params
    )

    start = time.perf_counter()

    keypoints = detector.detect(
        gray
    )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms



def gftt_detector(gray):

    start = time.perf_counter()

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3,
        useHarrisDetector=False
    )

    if corners is None:

        keypoints = []

    else:

        keypoints = []

        for corner in corners:

            x = corner[0][0]
            y = corner[0][1]

            keypoints.append(
                cv2.KeyPoint(
                    float(x),
                    float(y),
                    3
                )
            )

    end = time.perf_counter()

    time_ms = (end - start) * 1000

    return keypoints, None, time_ms




algorithms = {

    "Harris": harris_detector,

    "Shi-Tomasi": shi_tomasi_detector,

    "FAST": fast_detector,

    "SIFT": sift_detector,

    "ORB": orb_detector,

    "AKAZE": akaze_detector,

    "BRISK": brisk_detector,

    "MSER": mser_detector,

    "SimpleBlob": blob_detector,

    "GFTT": gftt_detector

}




results = []



# RUN ALL ALGORITHMS

for name, algorithm in algorithms.items():

    print("-" * 80)

    print(
        f"Running {name}..."
    )

    try:
        # Run algorithm

        keypoints, descriptors, processing_time = algorithm(
            gray
        )


        
        # Number of keypoints
        

        number_keypoints = len(
            keypoints
        )


        
        # Keypoint density
        # Keypoints per million pixels
        

        keypoint_density = (
            number_keypoints /
            image_area
        ) * 1_000_000


        # ----------------------------------------------------
        # Keypoint statistics
        # ----------------------------------------------------

        if number_keypoints > 0:

            responses = [
                kp.response
                for kp in keypoints
            ]

            sizes = [
                kp.size
                for kp in keypoints
            ]

            x_positions = [
                kp.pt[0]
                for kp in keypoints
            ]

            y_positions = [
                kp.pt[1]
                for kp in keypoints
            ]


            average_response = np.mean(
                responses
            )

            average_size = np.mean(
                sizes
            )


            # ------------------------------------------------
            # Spatial spread
            # ------------------------------------------------

            x_std = np.std(
                x_positions
            )

            y_std = np.std(
                y_positions
            )

            spatial_spread = np.sqrt(
                x_std ** 2 +
                y_std ** 2
            )

        else:

            average_response = 0

            average_size = 0

            spatial_spread = 0


        

        if descriptors is not None:

            descriptor_count = len(
                descriptors
            )

            descriptor_dimensions = (
                descriptors.shape[1]
            )

            descriptor_type = str(
                descriptors.dtype
            )

            descriptor_memory = (
                descriptors.nbytes
            )

        else:

            descriptor_count = "N/A"

            descriptor_dimensions = "N/A"

            descriptor_type = "N/A"

            descriptor_memory = "N/A"


       
        # Draw keypoints
        

        output_image = cv2.drawKeypoints(
            img,
            keypoints,
            None,
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
        )


        
        # Safe filename
        

        safe_name = (
            name
            .lower()
            .replace("-", "_")
        )


        
        # Save visualization
        

        output_path = os.path.join(
            OUTPUT_FOLDER,
            safe_name + "_keypoints.jpg"
        )

        cv2.imwrite(
            output_path,
            output_image
        )


        
        # Store results
        

        results.append({

            "Algorithm":
                name,

            "Keypoints":
                number_keypoints,

            "Keypoint Density":
                round(
                    keypoint_density,
                    2
                ),

            "Processing Time (ms)":
                round(
                    processing_time,
                    3
                ),

            "Average Keypoint Size":
                round(
                    float(average_size),
                    3
                ),

            "Average Response":
                round(
                    float(average_response),
                    5
                ),

            "Spatial Spread":
                round(
                    float(spatial_spread),
                    2
                ),

            "Descriptor Count":
                descriptor_count,

            "Descriptor Dimensions":
                descriptor_dimensions,

            "Descriptor Type":
                descriptor_type,

            "Descriptor Memory (bytes)":
                descriptor_memory

        })


        
        # Console output
        

        print(
            f"Keypoints: {number_keypoints}"
        )

        print(
            f"Processing time: "
            f"{processing_time:.3f} ms"
        )

        print(
            f"Output saved: {output_path}"
        )


    except Exception as error:

        
        # Keep failures clean in published results
        

        print(
            f"{name}: Results not available "
            f"in this environment."
        )

        


        
        # Store N/A instead of ERROR
        

        results.append({

            "Algorithm":
                name,

            "Keypoints":
                "N/A",

            "Keypoint Density":
                "N/A",

            "Processing Time (ms)":
                "N/A",

            "Average Keypoint Size":
                "N/A",

            "Average Response":
                "N/A",

            "Spatial Spread":
                "N/A",

            "Descriptor Count":
                "N/A",

            "Descriptor Dimensions":
                "N/A",

            "Descriptor Type":
                "N/A",

            "Descriptor Memory (bytes)":
                "N/A"

        })



# FROM HERE WE CREATE DATAFRAME AND FRAME THE RESULTS INTO A TABLE USING PANDAS


df = pd.DataFrame(
    results
)


# PRINT FINAL TABLE


print("\n")

print("=" * 160)

print(
    "FINAL FEATURE DETECTION COMPARISON"
)

print("=" * 160)

print(
    df.to_string(
        index=False
    )
)



# SAVE CSV


csv_path = os.path.join(
    OUTPUT_FOLDER,
    "feature_detection_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)



# SAVE EXCEL FILE

excel_path = os.path.join(
    OUTPUT_FOLDER,
    "feature_detection_comparison.xlsx"
)

df.to_excel(
    excel_path,
    index=False
)



print("\n")

print("=" * 80)

print("ANALYSIS COMPLETE!")

print("=" * 80)

print()

print(
    "Comparison CSV:"
)

print(
    csv_path
)

print()

print(
    "Comparison Excel:"
)

print(
    excel_path
)

print()

print(
    "Visualizations saved in:"
)

print(
    OUTPUT_FOLDER
)

print()

print(
    "You now have results for all 10 algorithms."
)

print()

print(
    "Note: Algorithms that could not be executed "
    "in the current environment are reported as N/A."
)