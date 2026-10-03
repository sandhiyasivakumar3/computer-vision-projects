from ultralytics import YOLO
import cv2
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "images",
    "waste_image.jpg"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "yolo11n.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "waste_detection_result.jpg"
)


# ============================================================
# COCO CLASS IDs
# ============================================================

# COCO dataset:
# 39 = bottle
# 41 = cup
# 45 = bowl

WASTE_CLASSES = {
    39: "PLASTIC BOTTLE",
    41: "CUP",
    45: "CONTAINER"
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VISION-BASED WASTE DETECTION AND CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    print("\nChecking input image...")

    if not os.path.isfile(IMAGE_PATH):

        print("\nERROR: Image not found!")
        print(IMAGE_PATH)

        return

    print("Image found:")
    print(IMAGE_PATH)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        print("\nERROR: OpenCV could not read the image.")

        return

    height, width = image.shape[:2]

    print(
        f"\nImage size: {width} x {height}"
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print("\nLoading YOLO11n model...")

    if not os.path.isfile(MODEL_PATH):

        print("\nERROR: YOLO model not found!")
        print(MODEL_PATH)

        return

    model = YOLO(
        MODEL_PATH
    )

    print("YOLO model loaded successfully.")

    # --------------------------------------------------------
    # Run detection
    # --------------------------------------------------------

    print("\nRunning waste detection...")
    print("Confidence : 0.01")
    print("Image size : 1920")
    print("Classes    : bottle, cup, bowl")

    results = model.predict(
        source=IMAGE_PATH,
        conf=0.01,
        imgsz=1920,
        max_det=300,
        augment=True,
        classes=[39, 41, 45],
        verbose=True
    )

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    detected_objects = 0

    bottle_count = 0
    cup_count = 0
    container_count = 0

    print("\n")
    print("=" * 70)
    print("DETECTIONS")
    print("=" * 70)

    for result in results:

        if result.boxes is None:

            print("No bounding boxes returned.")

            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            if class_id not in WASTE_CLASSES:

                continue

            class_name = WASTE_CLASSES[
                class_id
            ]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            print(
                f"{class_name:20s} "
                f"confidence = {confidence:.3f}"
            )

            # ------------------------------------------------
            # Count objects
            # ------------------------------------------------

            detected_objects += 1

            if class_id == 39:

                bottle_count += 1

            elif class_id == 41:

                cup_count += 1

            elif class_id == 45:

                container_count += 1

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                image,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        30
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # ========================================================
    # RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Total waste objects : {detected_objects}"
    )

    print(
        f"Plastic bottles     : {bottle_count}"
    )

    print(
        f"Cups                : {cup_count}"
    )

    print(
        f"Containers          : {container_count}"
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if detected_objects > 0:

        status = (
            f"WASTE DETECTED: "
            f"{detected_objects} OBJECTS"
        )

        print(
            "\nStatus: WASTE DETECTED"
        )

    else:

        status = (
            "NO WASTE OBJECTS DETECTED"
        )

        print(
            "\nStatus: NO WASTE OBJECTS DETECTED"
        )

    # --------------------------------------------------------
    # Draw status
    # --------------------------------------------------------

    cv2.putText(
        image,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        3
    )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    success = cv2.imwrite(
        OUTPUT_PATH,
        image
    )

    if success:

        print("\nResult saved successfully:")
        print(OUTPUT_PATH)

    else:

        print("\nERROR: Could not save result.")

    print("=" * 70)

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Waste Detection and Classification",
        image
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()