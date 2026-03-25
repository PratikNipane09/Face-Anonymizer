import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse



def process_img(img, face_detection):
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
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (20,20))

     return img

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default='None')

args = args.parse_args()


"""Creating a Directory"""
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



"""Detect Faces"""
# Downloading the face_detector.task model into my directory   
base_options = python.BaseOptions(model_asset_path='Face_Detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)

# Creating a detector 
with vision.FaceDetector.create_from_options(options) as detector:

    if args.mode in ["image"]:
        """Read Image"""
        img = cv2.imread(args.filePath)
        img = process_img(img, detector)
        # Saving an Image
        cv2.imwrite(os.path.join(output_dir, 'Output.jpg'), img)


    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)

        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       25,
                                       (frame.shape[1], frame.shape[0]))

        
        
        while ret:
            frame = process_img(frame, detector)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(1)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, detector)
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            ret, frame = cap.read(0)

            

        cap.release()
