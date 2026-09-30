import cv2
import numpy as np
import os
import sys


def analyze_fruit(image_path, output_dir="results"):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.resize(image, (800, 600))
    original = image.copy()

    # =========================================================
    # 1. GRABCUT SEGMENTATION
    # =========================================================

    mask = np.zeros(image.shape[:2], np.uint8)

    h, w = image.shape[:2]

    # Assume fruit is located near the centre
    rect = (
        int(w * 0.10),
        int(h * 0.05),
        int(w * 0.80),
        int(h * 0.90)
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    fruit_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    # Clean mask
    kernel = np.ones((7, 7), np.uint8)

    fruit_mask = cv2.morphologyEx(
        fruit_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    fruit_mask = cv2.morphologyEx(
        fruit_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # =========================================================
    # 2. KEEP ONLY LARGEST OBJECT
    # =========================================================

    contours, _ = cv2.findContours(
        fruit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("Fruit not detected.")
        return

    fruit_contour = max(
        contours,
        key=cv2.contourArea
    )

    fruit_area = cv2.contourArea(fruit_contour)

    if fruit_area < 5000:
        print("Fruit region too small.")
        return

    clean_mask = np.zeros_like(fruit_mask)

    cv2.drawContours(
        clean_mask,
        [fruit_contour],
        -1,
        255,
        -1
    )

    fruit_mask = clean_mask

    # =========================================================
    # 3. EXTRACT FRUIT
    # =========================================================

    fruit = cv2.bitwise_and(
        image,
        image,
        mask=fruit_mask
    )

    hsv = cv2.cvtColor(
        fruit,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # =========================================================
    # 4. DEFECT DETECTION
    # =========================================================

    # Dark/brown areas inside fruit
    dark = (
        (V < 80) &
        (S > 50) &
        (fruit_mask > 0)
    )

    brown = (
        (H >= 5) &
        (H <= 25) &
        (S > 80) &
        (V < 170) &
        (fruit_mask > 0)
    )

    defect_mask = (
        dark | brown
    ).astype(np.uint8) * 255

    # Remove small noise
    defect_kernel = np.ones((5, 5), np.uint8)

    defect_mask = cv2.morphologyEx(
        defect_mask,
        cv2.MORPH_OPEN,
        defect_kernel,
        iterations=2
    )

    defect_mask = cv2.morphologyEx(
        defect_mask,
        cv2.MORPH_CLOSE,
        defect_kernel,
        iterations=1
    )

    # =========================================================
    # 5. REMOVE SMALL DEFECTS
    # =========================================================

    defect_contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_defects = np.zeros_like(
        defect_mask
    )

    for contour in defect_contours:

        area = cv2.contourArea(contour)

        if area > 150:
            cv2.drawContours(
                clean_defects,
                [contour],
                -1,
                255,
                -1
            )

    defect_mask = clean_defects

    # =========================================================
    # 6. CALCULATE DEFECT PERCENTAGE
    # =========================================================

    fruit_pixels = cv2.countNonZero(
        fruit_mask
    )

    defect_pixels = cv2.countNonZero(
        defect_mask
    )

    defect_percentage = (
        defect_pixels / fruit_pixels * 100
        if fruit_pixels > 0
        else 0
    )

    # =========================================================
    # 7. QUALITY CLASSIFICATION
    # =========================================================

    if defect_percentage < 3:
        quality = "GOOD"

    elif defect_percentage < 10:
        quality = "MODERATE"

    else:
        quality = "DEFECTIVE"

    # =========================================================
    # 8. RESULT IMAGE
    # =========================================================

    result = original.copy()

    # Fruit boundary
    cv2.drawContours(
        result,
        [fruit_contour],
        -1,
        (0, 255, 0),
        3
    )

    # Defect boundaries
    defect_contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in defect_contours:

        if cv2.contourArea(contour) > 150:

            cv2.drawContours(
                result,
                [contour],
                -1,
                (0, 0, 255),
                3
            )

    # =========================================================
    # 9. TEXT
    # =========================================================

    if quality == "GOOD":
        color = (0, 255, 0)
    elif quality == "MODERATE":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(
        result,
        f"Quality: {quality}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.putText(
        result,
        f"Defect Area: {defect_percentage:.2f}%",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Fruit Area: {fruit_area:.0f} px",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =========================================================
    # 10. SAVE RESULTS
    # =========================================================

    cv2.imwrite(
        os.path.join(output_dir, "original.jpg"),
        original
    )

    cv2.imwrite(
        os.path.join(output_dir, "fruit_mask.jpg"),
        fruit_mask
    )

    cv2.imwrite(
        os.path.join(output_dir, "defects.jpg"),
        defect_mask
    )

    cv2.imwrite(
        os.path.join(output_dir, "fruit_quality_result.jpg"),
        result
    )

    # =========================================================
    # 11. TERMINAL OUTPUT
    # =========================================================

    print("\n===== FRUIT QUALITY ANALYSIS =====")
    print(f"Fruit area    : {fruit_area:.2f} pixels")
    print(f"Defect pixels : {defect_pixels}")
    print(f"Defect area   : {defect_percentage:.2f}%")
    print(f"Quality       : {quality}")
    print("==================================")
    print("\nResults saved in: results")


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python src\\fruit_quality.py <image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    analyze_fruit(image_path)