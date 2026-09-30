# Fruit Quality Inspection System

## 1. Problem Statement

Manual fruit inspection is time-consuming and can be inconsistent. This project uses Computer Vision to inspect apple images, detect visible defects, and classify fruit quality.

## 2. Objective

- Detect and isolate the apple from the background.
- Preprocess the input image.
- Detect visible defect regions.
- Calculate defective area.
- Classify the apple as GOOD, MODERATE, or DEFECTIVE.
- Evaluate the system using test images.

## 3. Dataset

A custom dataset containing 5 apple images was used.

| Image | Expected Class |
|---|---|
| apple_good_01.jpg.jpeg | GOOD |
| apple_good_02.jpg.jpg | GOOD |
| apple_good_03.jpg.jpg | GOOD |
| apple_defect_01.jpg.jpg | DEFECTIVE |
| apple_defect_02.jpg.jpg | DEFECTIVE |

## 4. Methodology

```text
Input Image
    ↓
Image Resizing
    ↓
GrabCut Segmentation
    ↓
Morphological Processing
    ↓
Fruit Contour Detection
    ↓
HSV Colour Conversion
    ↓
Defect Detection
    ↓
Defect Area Calculation
    ↓
Quality Classification
    ↓
Final Result