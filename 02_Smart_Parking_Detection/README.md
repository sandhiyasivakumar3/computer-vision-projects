# Smart Parking Space Detection and Occupancy Monitoring System

## 1. Problem Statement

Finding available parking spaces manually can be difficult in crowded parking areas. This project uses Computer Vision to detect vehicles and determine the occupancy status of predefined parking spaces.

The system identifies parking spaces as either:

- FREE
- OCCUPIED

It also processes a parking video and counts detected vehicles.

---

## 2. Objective

The main objectives are:

- Detect vehicles in parking-area images.
- Identify the occupancy status of parking spaces.
- Count free and occupied parking spaces.
- Process parking videos frame-by-frame.
- Generate visual output showing parking occupancy.
- Demonstrate a practical Computer Vision application.

---

## 3. Dataset

The project uses a small parking dataset containing:

### Images

Three parking-area images were used for testing.

The images contain multiple parking spaces with vehicles and empty spaces.

### Video

One parking video was used for real-time video processing.

Video frames processed:

**844 frames**

### Dataset Type

The dataset used for this implementation consists of parking images and a parking video collected/selected for project demonstration.

---

## 4. Methodology

The system follows this pipeline:

```text
Parking Image / Video
        ↓
Image Preprocessing
        ↓
YOLO Vehicle Detection
        ↓
Vehicle Bounding Boxes
        ↓
Parking Space Analysis
        ↓
FREE / OCCUPIED Classification
        ↓
Parking Space Count
        ↓
Output Image / Video