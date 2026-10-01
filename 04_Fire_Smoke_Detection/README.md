# Real-Time Fire and Smoke Detection

## Problem Statement

Fire and smoke can cause serious damage if they are not detected early. A computer vision system can analyse images and identify visible fire and smoke regions automatically.

## Objective

To develop a computer vision system that detects fire and smoke in an image using image processing techniques.

## Dataset

The dataset contains a fire and smoke image stored in:

`dataset/images/fire_smoke_image.jpg.jpg`

## Methodology

The system uses the following image processing steps:

1. Read the input image using OpenCV.
2. Convert the image from BGR to HSV colour space.
3. Detect fire using HSV colour thresholding.
4. Detect smoke using HSV-based intensity and saturation thresholding.
5. Apply morphological operations to remove noise.
6. Detect regions using contours.
7. Draw bounding boxes around detected fire and smoke.
8. Calculate the detected fire and smoke pixel areas.
9. Display the final detection status.

## Tools and Libraries

- Python
- OpenCV
- NumPy

## Results

The system successfully detected both fire and smoke.

| Parameter | Result |
|---|---:|
| Detection Status | FIRE + SMOKE DETECTED |
| Fire Area | 18,686 pixels |
| Smoke Area | 104,024 pixels |

The processed result is saved as:

`results/fire_smoke_result.jpg`

## Evaluation

The evaluation result is stored in:

`evaluation_results/evaluation.txt`

The system successfully detected the visible fire and smoke regions in the test image.

## Screenshots

The output screenshot is available at:

`screenshots/fire_smoke_result.jpg`

## Conclusion

The developed computer vision system successfully detects visible fire and smoke regions using HSV colour thresholding, morphological processing, and contour detection. The approach demonstrates how traditional image processing techniques can be used for basic fire and smoke detection.
