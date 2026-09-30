import cv2
from ultralytics import YOLO

IMAGE_PATH = "02_Smart_Parking_Detection/dataset/images/parking_03.jpg.jpg"
OUTPUT_PATH = "02_Smart_Parking_Detection/results/parking_occupancy.jpg"

model = YOLO("yolo11n.pt")

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image not found!")
    exit()

results = model(image, conf=0.30, verbose=False)

# Manually defined parking spaces for this image.
# Format: (x1, y1, x2, y2)
PARKING_SPACES = [
    (285, 285, 350, 365),
    (350, 280, 420, 365),
    (420, 275, 490, 365),
    (490, 275, 560, 365),
    (560, 270, 630, 365),
    (630, 270, 700, 365),
    (700, 270, 770, 365),
    (770, 270, 840, 365),
]

occupied = 0
free = 0

vehicle_boxes = []

for result in results:
    for box in result.boxes:

        cls = int(box.cls[0])

        if cls not in [2, 3, 5, 7]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        vehicle_boxes.append((x1, y1, x2, y2))


def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


for i, space in enumerate(PARKING_SPACES, start=1):

    is_occupied = False

    for vehicle in vehicle_boxes:

        iou = calculate_iou(space, vehicle)

        if iou > 0.10:
            is_occupied = True
            break

    x1, y1, x2, y2 = space

    if is_occupied:
        color = (0, 0, 255)
        status = "OCCUPIED"
        occupied += 1
    else:
        color = (0, 255, 0)
        status = "FREE"
        free += 1

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.putText(
        image,
        f"{i}: {status}",
        (x1, max(y1 - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


# Summary
cv2.rectangle(
    image,
    (10, 10),
    (330, 75),
    (0, 0, 0),
    -1
)

cv2.putText(
    image,
    f"FREE: {free}  OCCUPIED: {occupied}",
    (20, 52),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2
)

cv2.imwrite(OUTPUT_PATH, image)

print("=" * 50)
print("SMART PARKING OCCUPANCY DETECTION")
print("=" * 50)
print(f"Total spaces : {len(PARKING_SPACES)}")
print(f"Free spaces  : {free}")
print(f"Occupied     : {occupied}")
print(f"Result saved : {OUTPUT_PATH}")
print("=" * 50)

cv2.imshow("Parking Occupancy", image)
cv2.waitKey(0)
cv2.destroyAllWindows()