# Road Lane Detection and Departure Warning System

## 1. Problem Statement

Lane detection is an important Computer Vision application for Advanced Driver Assistance Systems (ADAS). The system must identify road lane boundaries from images and video and provide a warning when the vehicle moves away from the detected lane.

## 2. Objective

The objective of this project is to develop a Computer Vision-based road lane detection system that:

- Detects road lane lines.
- Identifies the left and right lane boundaries.
- Estimates the center of the detected lane.
- Calculates lane deviation.
- Provides a lane departure warning.
- Processes road-driving video in real time.

## 3. Dataset

The project uses:

- One road image for image-based lane detection.
- One road-driving video for video-based lane detection.

### Input Files

```text
dataset/
├── images/
│   └── road_image.jpg.avif
└── videos/
    └── road_video.mp4.mp4