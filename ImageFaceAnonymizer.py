import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""Creating a Directory"""
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


"""Read Image"""
img = cv2.imread(os.path.join('.', 'data', 'HumanFace.jpg'))

H, W, _ = img.shape

"""Detect Faces"""
# Downloading the face_detector.task model into my directory   
base_options = python.BaseOptions(model_asset_path='Face_Detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)

# Creating a detector 
with vision.FaceDetector.create_from_options(options) as detector:
     
     # Converting an OpenCV image directly into a Mediapipe Image object
     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

     # Process the image
     detection_result = detector.detect(mp_image)
     

     if detection_result is not None:
        for detection in detection_result.detections:
            
            bbox = detection.bounding_box

            x1 = bbox.origin_x
            y1 = bbox.origin_y
            w = bbox.width
            h = bbox.height

            """ Blur Faces"""
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (10,10))

    
"""Saving the Image"""

cv2.imwrite(os.path.join(output_dir, 'Output.jpg'), img)

