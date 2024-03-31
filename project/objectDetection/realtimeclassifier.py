import cv2
import sys
import time
import numpy as np
import os
from tflite_support.task import vision
from tflite_support.task import processor
from tflite_support.task import core


def classify_pothole_position(predictions, image_width):
    positions = {'left': 0, 'right': 0, 'middle': 0}
    
    for prediction in predictions:
        bbox = prediction.bounding_box
        bbox_center = (bbox.origin_x + bbox.width / 2) / image_width
        if bbox_center < 0.33:
            positions['left'] += 1
        elif bbox_center > 0.67:
            positions['right'] += 1
        else:
            positions['middle'] += 1
    
    if positions['left'] > 0 and positions['right'] > 0:
        return 'both sides'
    elif positions['middle'] > 0:
        return 'middle'
    elif positions['left'] > 0:
        return 'left'
    elif positions['right'] > 0:
        return 'right'
    else:
        return 'no potholes detected'
    

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:

    counter, fps = 0, 0
    start_time = time.time()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)
    
    detectStartTime = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = cv2.flip(image, 0)  
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detection_result = detector.detect(input_tensor)
        positions = classify_pothole_position(detection_result.detections, image.shape[1])
        print("Detected pothole positions:", positions) 

     
        image_with_detections = image  

        cv2.imshow('object_detector', image_with_detections)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()