import cv2
import numpy as np
import os
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "videos",
    "road_video.mp4.mp4"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "lane_detection_output.mp4"
)


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
# AVERAGE LANE
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
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Canny
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # ROI
    cropped_edges = region_of_interest(edges)

    # Hough Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=100
    )

    result = frame.copy()

    left_lines = []
    right_lines = []

    if lines is not None:

        lines = np.asarray(lines).reshape(-1, 4)

        for line in lines:

            x1, y1, x2, y2 = map(int, line)

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.4:
                continue

            if slope < 0:
                left_lines.append(
                    (x1, y1, x2, y2)
                )
            else:
                right_lines.append(
                    (x1, y1, x2, y2)
                )

    left_lane = average_lane_line(left_lines)
    right_lane = average_lane_line(right_lines)

    # --------------------------------------------------------
    # Draw lanes
    # --------------------------------------------------------

    if left_lane is not None:

        x1, y1, x2, y2 = left_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            6
        )

    if right_lane is not None:

        x1, y1, x2, y2 = right_lane

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            6
        )

    # --------------------------------------------------------
    # Lane status
    # --------------------------------------------------------

    height, width = frame.shape[:2]

    image_center = width // 2

    if left_lane is not None and right_lane is not None:

        left_x = left_lane[2]
        right_x = right_lane[2]

        lane_center = (left_x + right_x) // 2

        deviation = lane_center - image_center

        threshold = int(width * 0.10)

        if abs(deviation) > threshold:

            status = "LANE DEPARTURE WARNING!"
            status_color = (0, 0, 255)

        elif deviation > threshold * 0.5:

            status = "MOVE LEFT"
            status_color = (0, 165, 255)

        elif deviation < -threshold * 0.5:

            status = "MOVE RIGHT"
            status_color = (0, 165, 255)

        else:

            status = "LANE CENTERED"
            status_color = (0, 255, 0)

        # Vehicle center
        cv2.line(
            result,
            (image_center, height),
            (image_center, int(height * 0.65)),
            (255, 0, 0),
            3
        )

        # Lane center
        cv2.line(
            result,
            (lane_center, height),
            (lane_center, int(height * 0.65)),
            (0, 255, 255),
            3
        )

        cv2.putText(
            result,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3
        )

        cv2.putText(
            result,
            f"Lane deviation: {deviation}px",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            result,
            "LANE NOT CLEAR",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            3
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("REAL-TIME ROAD LANE DETECTION")
    print("=" * 60)

    if not os.path.exists(VIDEO_PATH):

        print("\nERROR: Video not found!")
        print(VIDEO_PATH)
        return

    print("\nOpening video:")
    print(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"\nResolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps if fps > 0 else 30,
        (width, height)
    )

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = process_frame(frame)

        writer.write(result)

        frame_count += 1

        cv2.imshow(
            "Road Lane Detection",
            result
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed : {frame_count}")
    print(f"Output video     : {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()