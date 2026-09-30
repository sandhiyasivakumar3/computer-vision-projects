import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Vehicle classes from COCO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Input image
IMAGE_PATH = "02_Smart_Parking_Detection/dataset/images/parking_03.jpg.jpg"

# Output image
OUTPUT_PATH = "02_Smart_Parking_Detection/results/parking_result.jpg"

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found!")
    print(IMAGE_PATH)
    exit()

# Detect vehicles
results = model(image, conf=0.35)

vehicle_count = 0

# Draw detections
for result in results:
    boxes = result.boxes

    for box in boxes:
        cls = int(box.cls[0])
        confidence = float(box.conf[0])

        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = VEHICLE_CLASSES[cls]

        vehicle_count += 1

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Display vehicle count
cv2.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)

cv2.putText(
    image,
    f"Vehicles detected: {vehicle_count}",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

# Save result
cv2.imwrite(OUTPUT_PATH, image)

print("=" * 45)
print("SMART PARKING DETECTION")
print("=" * 45)
print(f"Vehicles detected: {vehicle_count}")
print(f"Result saved to: {OUTPUT_PATH}")
print("=" * 45)

# Show result
cv2.imshow("Smart Parking Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()