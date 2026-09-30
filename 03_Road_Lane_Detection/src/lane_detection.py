import cv2
import numpy as np
import os
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "images",
    "road_image.jpg.avif"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "lane_detection_result.jpg"
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    if not os.path.exists(path):
        print("ERROR: Image not found!")
        print(path)
        return None

    image = cv2.imread(path)

    if image is not None:
        return image

    try:
        pil_image = Image.open(path).convert("RGB")
        image = np.array(pil_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    except Exception as e:
        print("ERROR: Could not read image.")
        print(e)
        return None


# ============================================================
# REGION OF INTEREST
# ============================================================

def region_of_interest(image):

    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.40), int(height * 0.58)),
        (int(width * 0.60), int(height * 0.58)),
        (int(width * 0.95), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


# ============================================================
# AVERAGE LANE LINE
# ============================================================

def average_lane_line(lines):

    if len(lines) == 0:
        return None

    x1_values = []
    y1_values = []
    x2_values = []
    y2_values = []

    for x1, y1, x2, y2 in lines:
        x1_values.append(x1)
        y1_values.append(y1)
        x2_values.append(x2)
        y2_values.append(y2)

    return (
        int(np.mean(x1_values)),
        int(np.mean(y1_values)),
        int(np.mean(x2_values)),
        int(np.mean(y2_values))
    )


# ============================================================
# LANE DETECTION
# ============================================================

def detect_lanes(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # Region of interest
    cropped_edges = region_of_interest(edges)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=100
    )

    result = image.copy()

    left_lines = []
    right_lines = []

    if lines is not None:

        # Make every detected line a simple [x1,y1,x2,y2]
        lines = np.asarray(lines).reshape(-1, 4)

        for line in lines:

            x1, y1, x2, y2 = map(int, line)

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Ignore horizontal lines
            if abs(slope) < 0.4:
                continue

            # Left lane
            if slope < 0:
                left_lines.append(
                    (x1, y1, x2, y2)
                )

            # Right lane
            else:
                right_lines.append(
                    (x1, y1, x2, y2)
                )

    left_lane = average_lane_line(left_lines)
    right_lane = average_lane_line(right_lines)

    # Draw left lane
    if left_lane is not None:

        x1, y1, x2, y2 = left_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

    # Draw right lane
    if right_lane is not None:

        x1, y1, x2, y2 = right_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            8
        )

    return result, left_lane, right_lane


# ============================================================
# LANE DEPARTURE WARNING
# ============================================================

def calculate_departure_warning(
    image,
    left_lane,
    right_lane
):

    height, width = image.shape[:2]

    image_center = width // 2

    # If lanes are missing
    if left_lane is None or right_lane is None:

        cv2.putText(
            image,
            "LANE NOT CLEAR",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            3
        )

        return image

    # Bottom x positions
    left_x = left_lane[2]
    right_x = right_lane[2]

    # Lane center
    lane_center = (left_x + right_x) // 2

    # Vehicle/image center
    deviation = lane_center - image_center

    # Draw vehicle center
    cv2.line(
        image,
        (image_center, height),
        (image_center, int(height * 0.65)),
        (255, 0, 0),
        3
    )

    # Draw lane center
    cv2.line(
        image,
        (lane_center, height),
        (lane_center, int(height * 0.65)),
        (0, 255, 255),
        3
    )

    threshold = int(width * 0.10)

    if abs(deviation) > threshold:

        warning = "LANE DEPARTURE WARNING!"
        warning_color = (0, 0, 255)

    elif deviation > threshold * 0.5:

        warning = "MOVE LEFT"
        warning_color = (0, 165, 255)

    elif deviation < -threshold * 0.5:

        warning = "MOVE RIGHT"
        warning_color = (0, 165, 255)

    else:

        warning = "LANE CENTERED"
        warning_color = (0, 255, 0)

    # Warning text
    cv2.putText(
        image,
        warning,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        warning_color,
        3
    )

    # Deviation
    cv2.putText(
        image,
        f"Lane deviation: {deviation}px",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ROAD LANE DETECTION AND DEPARTURE WARNING")
    print("=" * 60)

    print("\nLoading image:")
    print(IMAGE_PATH)

    image = load_image(IMAGE_PATH)

    if image is None:
        return

    print("Image loaded successfully.")

    # Detect lanes
    result, left_lane, right_lane = detect_lanes(image)

    print("\nDetection results:")

    if left_lane is not None:
        print("Left lane  : DETECTED")
    else:
        print("Left lane  : NOT DETECTED")

    if right_lane is not None:
        print("Right lane : DETECTED")
    else:
        print("Right lane : NOT DETECTED")

    # Departure warning
    result = calculate_departure_warning(
        result,
        left_lane,
        right_lane
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

    print("\nResult saved:")
    print(OUTPUT_PATH)

    print("=" * 60)

    # Display
    cv2.imshow(
        "Road Lane Detection",
        result
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()