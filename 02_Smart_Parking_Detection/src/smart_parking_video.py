import cv2
import glob
from ultralytics import YOLO

# -----------------------------
# FIND INPUT VIDEO
# -----------------------------
video_files = glob.glob(
    "02_Smart_Parking_Detection/dataset/videos/*"
)

if not video_files:
    print("ERROR: No video found!")
    exit()

VIDEO_PATH = video_files[0]

OUTPUT_PATH = (
    "02_Smart_Parking_Detection/results/"
    "smart_parking_output.mp4"
)

# -----------------------------
# LOAD YOLO
# -----------------------------
model = YOLO("yolo11n.pt")

# Vehicle classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# PROCESS VIDEO
# -----------------------------
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(
        frame,
        conf=0.35,
        verbose=False
    )

    vehicle_count = 0

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            vehicle_count += 1

            label = VEHICLE_CLASSES[cls]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # Information panel
    cv2.rectangle(
        frame,
        (10, 10),
        (330, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # Save frame
    out.write(frame)

    # Display
    cv2.imshow(
        "Smart Parking Video Detection",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("=" * 50)
print("SMART PARKING VIDEO DETECTION")
print("=" * 50)
print(f"Video processed: {VIDEO_PATH}")
print(f"Frames processed: {frame_count}")
print(f"Output saved to: {OUTPUT_PATH}")
print("=" * 50)