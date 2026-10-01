import cv2
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "images"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "fire_smoke_result.jpg"
)


# ============================================================
# FIND INPUT IMAGE
# ============================================================

def find_input_image():

    possible_files = [
        "fire_smoke_image.jpg",
        "fire_smoke_image.jpg.jpg",
        "fire_smoke_image.jpeg",
        "fire_smoke_image.png",
        "fire_smoke_image.webp"
    ]

    for filename in possible_files:

        path = os.path.join(
            IMAGE_DIR,
            filename
        )

        if os.path.isfile(path):
            return path

    return None


# ============================================================
# FIRE DETECTION
# ============================================================

def detect_fire(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_fire = np.array([
        0,
        100,
        100
    ])

    upper_fire = np.array([
        40,
        255,
        255
    ])

    fire_mask = cv2.inRange(
        hsv,
        lower_fire,
        upper_fire
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    fire_mask = cv2.morphologyEx(
        fire_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fire_mask = cv2.morphologyEx(
        fire_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return fire_mask


# ============================================================
# SMOKE DETECTION
# ============================================================

def detect_smoke(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_smoke = np.array([
        0,
        0,
        60
    ])

    upper_smoke = np.array([
        180,
        80,
        220
    ])

    smoke_mask = cv2.inRange(
        hsv,
        lower_smoke,
        upper_smoke
    )

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    smoke_mask = cv2.morphologyEx(
        smoke_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    smoke_mask = cv2.morphologyEx(
        smoke_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return smoke_mask


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    fire_mask,
    smoke_mask
):

    result = image.copy()

    # --------------------------------------------------------
    # FIRE
    # --------------------------------------------------------

    fire_contours, _ = cv2.findContours(
        fire_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fire_area = 0

    for contour in fire_contours:

        area = cv2.contourArea(contour)

        if area < 300:
            continue

        fire_area += area

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3
        )

        cv2.putText(
            result,
            "FIRE",
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # SMOKE
    # --------------------------------------------------------

    smoke_contours, _ = cv2.findContours(
        smoke_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    smoke_area = 0

    for contour in smoke_contours:

        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        smoke_area += area

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        cv2.putText(
            result,
            "SMOKE",
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if fire_area > 1000 and smoke_area > 1000:

        status = "FIRE + SMOKE DETECTED"
        status_color = (0, 0, 255)

    elif fire_area > 1000:

        status = "FIRE DETECTED"
        status_color = (0, 0, 255)

    elif smoke_area > 1000:

        status = "SMOKE DETECTED"
        status_color = (255, 0, 0)

    else:

        status = "NO FIRE/SMOKE DETECTED"
        status_color = (0, 255, 0)

    cv2.putText(
        result,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3
    )

    return (
        result,
        status,
        fire_area,
        smoke_area
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("REAL-TIME FIRE AND SMOKE DETECTION")
    print("=" * 60)

    # Find image automatically
    image_path = find_input_image()

    if image_path is None:

        print("\nERROR: Fire/smoke image not found!")

        print("\nExpected folder:")
        print(IMAGE_DIR)

        print("\nFiles currently available:")

        if os.path.exists(IMAGE_DIR):

            for filename in os.listdir(IMAGE_DIR):
                print(" -", filename)

        else:

            print("Image folder does not exist.")

        return

    print("\nInput image:")
    print(image_path)

    # Read image
    image = cv2.imread(
        image_path
    )

    if image is None:

        print("\nERROR: OpenCV could not read the image.")
        print("Check that the file is a valid image.")
        return

    print("Image loaded successfully.")

    # Detect fire
    fire_mask = detect_fire(
        image
    )

    # Detect smoke
    smoke_mask = detect_smoke(
        image
    )

    # Draw results
    (
        result,
        status,
        fire_area,
        smoke_area
    ) = draw_detections(
        image,
        fire_mask,
        smoke_mask
    )

    # Create results folder
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Save result
    cv2.imwrite(
        OUTPUT_PATH,
        result
    )

    # Terminal results
    print("\nDetection Results")
    print("-" * 40)

    print(
        f"Status     : {status}"
    )

    print(
        f"Fire area  : {fire_area:.0f} pixels"
    )

    print(
        f"Smoke area : {smoke_area:.0f} pixels"
    )

    print(
        "\nResult saved:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 60)

    # Display
    cv2.imshow(
        "Fire and Smoke Detection",
        result
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()