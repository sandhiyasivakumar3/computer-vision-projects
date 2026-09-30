import cv2
import os


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

OUTPUT_VIDEO = os.path.join(
    BASE_DIR,
    "results",
    "lane_detection_output.mp4"
)

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation_results"
)

EVALUATION_FILE = os.path.join(
    EVALUATION_DIR,
    "evaluation.txt"
)


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("ROAD LANE DETECTION - EVALUATION")
    print("=" * 60)

    # Check input video
    if not os.path.exists(VIDEO_PATH):
        print("ERROR: Input video not found.")
        print(VIDEO_PATH)
        return

    # Check output video
    if not os.path.exists(OUTPUT_VIDEO):
        print("ERROR: Output video not found.")
        print(OUTPUT_VIDEO)
        return

    # Read video information
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open input video.")
        return

    input_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    # Read output video
    output_cap = cv2.VideoCapture(
        OUTPUT_VIDEO
    )

    if not output_cap.isOpened():
        print("ERROR: Could not open output video.")
        return

    output_frames = int(
        output_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_fps = output_cap.get(
        cv2.CAP_PROP_FPS
    )

    output_cap.release()

    # Create evaluation folder
    os.makedirs(
        EVALUATION_DIR,
        exist_ok=True
    )

    # Frame processing percentage
    if input_frames > 0:
        processing_percentage = (
            output_frames / input_frames
        ) * 100
    else:
        processing_percentage = 0

    # Save evaluation
    with open(
        EVALUATION_FILE,
        "w"
    ) as file:

        file.write(
            "ROAD LANE DETECTION AND DEPARTURE WARNING SYSTEM\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Input video frames      : {input_frames}\n"
        )

        file.write(
            f"Output video frames     : {output_frames}\n"
        )

        file.write(
            f"Input FPS               : {fps:.2f}\n"
        )

        file.write(
            f"Output FPS              : {output_fps:.2f}\n"
        )

        file.write(
            f"Resolution              : {width} x {height}\n"
        )

        file.write(
            f"Frames processed        : {output_frames}\n"
        )

        file.write(
            f"Processing completeness : {processing_percentage:.2f}%\n"
        )

        file.write(
            "\nComputer Vision methods:\n"
        )

        file.write(
            "- Gaussian Blur\n"
        )

        file.write(
            "- Canny Edge Detection\n"
        )

        file.write(
            "- Region of Interest (ROI)\n"
        )

        file.write(
            "- Hough Line Transform\n"
        )

        file.write(
            "- Lane Center Estimation\n"
        )

        file.write(
            "- Lane Departure Warning\n"
        )

    print("\nEvaluation completed.")

    print(
        f"Input frames  : {input_frames}"
    )

    print(
        f"Output frames : {output_frames}"
    )

    print(
        f"FPS           : {fps:.2f}"
    )

    print(
        f"Completeness  : {processing_percentage:.2f}%"
    )

    print(
        f"\nEvaluation saved:"
    )

    print(EVALUATION_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()