# Vision-Based Waste Detection and Classification

## Problem Statement

Improper waste disposal creates environmental and public health problems. Computer vision can help identify waste objects automatically from images.

## Objective

To develop a vision-based system that detects and classifies visible waste objects using a YOLO deep learning model.

## Dataset

The project uses a waste image containing plastic bottles and other waste materials.

Input image:

`dataset/images/waste_image.jpg`

## Methodology

1. Load the input waste image.
2. Load the YOLO11n object detection model.
3. Detect waste-related objects using YOLO.
4. Filter detections for bottle, cup, and container classes.
5. Draw bounding boxes around detected objects.
6. Display the detected waste category and confidence score.
7. Count the detected waste objects.
8. Save the processed result.

## Tools and Libraries

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- PyTorch
- YOLO11n

## Results

The system successfully detected visible waste objects in the test image.

| Parameter | Result |
|---|---:|
| Detection Model | YOLO11n |
| Confidence Threshold | 0.01 |
| Image Size | 1920 |
| Waste Objects Detected | 6 |
| Detected Category | Plastic Bottle |

The output image is saved at:

`results/waste_detection_result.jpg`

## Evaluation

The evaluation details are available in:

`evaluation_results/evaluation.txt`

The system successfully generated bounding boxes for detected waste objects.

## Screenshots

The processed detection result is available at:

`screenshots/waste_detection_result.jpg`

## Limitations

The system uses a pretrained YOLO11n model. Its performance depends on the objects and visual conditions represented in the pretrained dataset. For reliable waste-specific classification, a dedicated waste dataset and fine-tuning would improve performance.

## Conclusion

The developed system demonstrates vision-based waste detection using YOLO11n. Plastic bottles were detected and highlighted with bounding boxes and confidence scores. The project demonstrates how deep learning-based computer vision can be applied to automated waste monitoring.