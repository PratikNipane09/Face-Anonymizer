# Face-Anonymizer
# 🕵️‍♂️ Image Face Anonymizer

A lightweight computer vision script built with Python, OpenCV, and Google's MediaPipe Tasks API. This tool automatically detects human faces in an image, isolates the region of interest (ROI), applies a blur effect to anonymize the faces, and saves the output.

## 🌟 Features
* **Fast & Local:** Uses the highly optimized MediaPipe `face_detector.tflite` model to run entirely locally without sending data to the cloud.
* **Accurate Bounding Boxes:** Extracts exact pixel coordinates for precise blurring.
* **Easy Setup:** Works on standard CPU hardware; no heavy GPUs required.

## 🛠️ Prerequisites
Before running this project, make sure you have Python installed along with the following libraries:
* `opencv-python` (`cv2`)
* `mediapipe` (Version 0.10.31 or newer)

You can install these via pip:
```bash
pip install opencv-python mediapipe
