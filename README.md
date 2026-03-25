# 🕵️‍♂️ Face Anonymizer Suite (Image, Video & Live Webcam)

A complete suite of highly optimized, local computer vision scripts built with Python, OpenCV, and Google's MediaPipe Tasks API. These tools automatically detect human faces, extract their exact bounding box coordinates, and apply a Gaussian blur to anonymize the subjects.

This repository includes three specific implementations:
1. **Static Image Anonymizer** (`ImageFaceAnonymizer.py`)
2. **Video File Anonymizer** (`VideoFaceAnonymizer.py`)
3. **Live Webcam Anonymizer** (`WebcamFaceAnonymizer.py`)

## 🌟 Features
* **Lightning Fast:** Uses the highly optimized MediaPipe `face_detector.tflite` model, allowing even the live webcam feed to run seamlessly on standard CPUs without lag.
* **100% Local & Private:** All processing happens directly on your machine. No image data is ever sent to a cloud server.
* **Precision Blurring:** Extracts exact pixel coordinates for tight, accurate region-of-interest (ROI) blurring.

## 🛠️ Prerequisites
Before running any of the scripts, ensure you have Python installed along with the required libraries. 

Install the dependencies via pip:
```bash
pip install opencv-python mediapipe
