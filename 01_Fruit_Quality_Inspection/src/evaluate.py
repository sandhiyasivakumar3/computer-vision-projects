import os
import cv2
import numpy as np

DATASET_DIR = "dataset"

EXPECTED = {
    "apple_good_01.jpg.jpeg": "GOOD",
    "apple_good_02.jpg.jpg": "GOOD",
    "apple_good_03.jpg.jpg": "GOOD",
    "apple_defect_01.jpg.jpg": "DEFECTIVE",
    "apple_defect_02.jpg.jpg": "DEFECTIVE",
}


def predict_quality(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return "ERROR"

    image = cv2.resize(image, (800, 600))

    # Fruit segmentation using GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)

    h, w = image.shape[:2]

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

    contours, _ = cv2.findContours(
        fruit_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "ERROR"

    fruit_contour = max(
        contours,
        key=cv2.contourArea
    )

    clean_mask = np.zeros_like(fruit_mask)

    cv2.drawContours(
        clean_mask,
        [fruit_contour],
        -1,
        255,
        -1
    )

    fruit_mask = clean_mask

    # Extract fruit
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

    # Dark defects
    dark = (
        (V < 80) &
        (S > 50) &
        (fruit_mask > 0)
    )

    # Brown defects
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

    # Remove tiny defects
    defect_contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_defects = np.zeros_like(defect_mask)

    for contour in defect_contours:

        if cv2.contourArea(contour) > 150:

            cv2.drawContours(
                clean_defects,
                [contour],
                -1,
                255,
                -1
            )

    defect_mask = clean_defects

    # Calculate defect percentage
    fruit_pixels = cv2.countNonZero(fruit_mask)
    defect_pixels = cv2.countNonZero(defect_mask)

    defect_percentage = (
        defect_pixels / fruit_pixels * 100
        if fruit_pixels > 0
        else 0
    )

    # Classification
    if defect_percentage < 3:
        return "GOOD"

    elif defect_percentage < 8:
        return "MODERATE"

    else:
        return "DEFECTIVE"


def main():

    correct = 0
    total = 0

    results = []

    print("\n========================================")
    print("       FRUIT QUALITY EVALUATION")
    print("========================================")

    for filename, expected in EXPECTED.items():

        image_path = os.path.join(
            DATASET_DIR,
            filename
        )

        predicted = predict_quality(image_path)

        if predicted == expected:
            correct += 1

        total += 1

        results.append(
            (filename, expected, predicted)
        )

        print(
            f"{filename:<30} "
            f"Expected: {expected:<10} "
            f"Predicted: {predicted}"
        )

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print("\n========================================")
    print(f"Correct predictions : {correct}/{total}")
    print(f"Accuracy            : {accuracy:.2f}%")
    print("========================================")

    os.makedirs("evaluation_results", exist_ok=True)

    with open(
        "evaluation_results/evaluation.txt",
        "w"
    ) as file:

        file.write("FRUIT QUALITY EVALUATION\n")
        file.write("========================\n\n")

        for filename, expected, predicted in results:

            file.write(
                f"{filename} | "
                f"Expected: {expected} | "
                f"Predicted: {predicted}\n"
            )

        file.write(
            f"\nAccuracy: {accuracy:.2f}%\n"
        )


if __name__ == "__main__":
    main()