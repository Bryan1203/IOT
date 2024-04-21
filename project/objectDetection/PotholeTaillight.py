####### Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/2/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a
# video. It draws boxes and scores around the objects of interest in each frame
# from the video.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import threading
import time
import numpy as np
import sys
import importlib.util
from gtts import gTTS
from playsound import playsound
import taillight
import multiprocessing
import playSound
import queue
import pygame

def play_alert(side):
    print(f"Play alert called for {side}")
    audio_file = f"{side.lower()}.mp3"
    
    if not os.path.exists(audio_file):
        text = f'Pothole {side}'
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        print(f"Generated audio file: {audio_file}")
    else:
        print(f"Audio file already exists: {audio_file}")
    
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Failed to play sound: {e}")
    

# Create a separate process to handle the alerts
def alert_process_queue(input):
    while True:
        alert = input.get()
        if alert is None:
            break
        play_alert(alert)



def main():

    last_alert_time = 0  # Update last alert time
    alert_cooldown = 3  # Cooldown in seconds (adjust as needed)

    # Create a queue to store the alerts
    alert_queue = multiprocessing.Queue()
    # Start the alert process
    alert_process = multiprocessing.Process(target=alert_process_queue, args=(alert_queue,))
    alert_process.start()

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--video', help='Name of the video file',
                        default='test.mp4')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    VIDEO_NAME = args.video
    min_conf_threshold = float(args.threshold)
    use_TPU = args.edgetpu
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()




    # delay_time = 2 
    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'   

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to video file
    VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Open video file
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)


    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)

    while(video.isOpened()):
        t1 = cv2.getTickCount()
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        mid_x = imW // 2
        closest_pothole = None
        max_area = 0  # Initialize the maximum area found

        # Iterate through all detections to find the one with the largest bounding box area
        
        
        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                ymin, xmin, ymax, xmax = [int(max(1, boxes[i][j] * imH)) if j % 2 == 0 else int(max(1, boxes[i][j] * imW)) for j in range(4)]
                area = (xmax - xmin) * (ymax - ymin)  # Calculate area of the bounding box
                bbox_mid_x = (xmin + xmax) // 2
                side = 'Left' if bbox_mid_x < mid_x else 'Right'
                
                # Construct label text with side information
                label = f'{side} Pothole: {int(scores[i]*100)}%'  # Updated label to include "Left" or "Right"

                # Always draw bounding box for detected potholes
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Determine if this pothole is the closest
                if area > max_area:
                    max_area = area
                    closest_pothole = (side, ymin, xmin, ymax, xmax)  # Update closest pothole info

        # Check if the closest pothole needs an alert and if cooldown is over
        alert_cooldown = 3
        if closest_pothole and (time.time() - last_alert_time >= alert_cooldown):
            last_alert_time = time.time()  # Update last alert time
            side, ymin, xmin, ymax, xmax = closest_pothole
            # Trigger the alert
            if side == "Right":
                print("Detection on right detected")
                alert_queue.put("right")
                taillight.send_message("rd")
            elif side == "Left":
                print("Detection on left detected")
                alert_queue.put("left")
                taillight.send_message("ld")

            


        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()