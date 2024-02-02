import threading
import object_detect
import mapping

#for object detect
import argparse
import sys
import time

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import picar_4wd as fc
import threading
import os

#for mapping
#import time
#import picar_4wd as fc
import numpy as np
import math
from queue import Queue
from queue import PriorityQueue
from multiprocessing import Process, Event

#mapping global variable

speed = 30
map_size = 50
orientation = 0
curr_x = 25
curr_y = 0


# This is your flag to signal the thread to pause
#pause_event = threading.Event()

def stopSignWait(waitTime):
    event.set()
    time.sleep(waitTime)
    #pause_event.set()
    event.clear()



def padPointMap(arr):
    def pad_ones(arr):
        # Create a copy of the array to avoid modifying the original array
        arr_copy = arr.copy()

        # Loop through each row in the array
        for i, row in enumerate(arr):
            # Find the indices of the 1s
            ones_idx = np.where(row == 1)[0]

            # Find the groups of consecutive 1s
            groups = np.split(ones_idx, np.where(np.diff(ones_idx) != 1)[0] + 1)

            # Loop through each group
            for group in groups:
                # If the group is longer than 3
                if len(group) >= 3:
                    # Pad two 1s before and after the group
                    if group[0] > 1:
                        arr_copy[i, group[0] - 1] = 3
                        #arr_copy[i, group[0] - 2] = 3
                    if group[-1] < len(row) - 1:
                        arr_copy[i, group[-1] + 1] = 3
                        #arr_copy[i, group[-1] + 2] = 3

        return arr_copy

    # Pad 1s around groups of consecutive 1s in rows
    after_arr = pad_ones(arr)

    # Pad 1s around groups of consecutive 1s in columns
    after_arr = pad_ones(after_arr.T).T

    return after_arr

def goRight(): 
    global orientation
    fc.turn_right(1)
    time.sleep(1.15)
    fc.stop()
    #update the orientation
    if orientation== 0:
        orientation = -90
    elif orientation == -90:
        orientation = 180
    elif orientation == 180:
        orientation = 90
    elif orientation == 90:
        orientation = 0

def goForward(): 
    fc.forward(1)
    time.sleep(0.4)
    fc.stop()

def goBackward(): 
    fc.backward(1)
    time.sleep(0.3)
    fc.stop()

def goLeft(): 
    global orientation
    fc.turn_left(1)
    time.sleep(1.25)
    fc.stop()
    #update the orientation
    if orientation== 0:
        orientation = 90
    elif orientation == -90:
        orientation = -0
    elif orientation == -180:
        orientation = -90
    elif orientation == 90:
        orientation = 180

def move_car(next_x, next_y):
    global orientation
    global curr_x
    global curr_y
    if next_x == curr_x + 1:  # Move right
        if orientation == 0:
            goRight()
            #time.sleep(0.1)
            #goForward()

        elif orientation == -90: 
            goForward()
            #update our curr x and y 
            curr_x = next_x
            curr_y = next_y

        elif orientation == 180:
            goLeft()
            #time.sleep(0.1)
            #goForward()
        elif orientation == 90:
            goRight()
            #time.sleep(0.1)
            #goRight()
            #time.sleep(0.1)
            #goForward()
        print("going Right")

    elif next_x == curr_x - 1:  # Move left
        if orientation == 0:
            goLeft()
            #time.sleep(0.1)
            #goForward()
        elif orientation == 90:
            goForward()
            #update x and y
            curr_x = next_x
            curr_y = next_y
        elif orientation == 180:
            goRight()
            #time.sleep(0.1)
            #goForward()
        elif orientation == -90:
            goLeft()

        print("going Left")
    elif next_y == curr_y + 1:  # Move forward
        if orientation == 0:
            goForward()
            #update x and y
            curr_x = next_x
            curr_y = next_y
            print("(", curr_x, " ", curr_y, " )", "(", next_x, " ", next_y, " )")

        elif orientation == -90:
            goLeft()
            #goForward()
        elif orientation == 90:
            goRight()
            #time.sleep(0.1)
            #goForward()
        elif orientation == 180:
           goBackward()
           #update x and y
           curr_x = next_x
           curr_y = next_y
        print("going forward")
    elif next_y == curr_y - 1:  # Move backward
        if orientation == 180:
            goForward()
            #update x and y
            curr_x = next_x
            curr_y = next_y
        elif orientation == -90:
            goRight()
            #time.sleep(0.1)
            #goForward()
        elif orientation == 90:
            goLeft()
            #time.sleep(0.1)
            #goForward()
        elif orientation == 0:
            goBackward()
            #update x and y
            curr_x = next_x
            curr_y = next_y
            
            
        print("going backward")

    #time.sleep(0.1)  # Adjust as needed for the movement duration
    #fc.stop()


def interpolate(point_map, curr_x, curr_y, obs_x, obs_y):
    # Interpolate between (curr_x, curr_y) and (obs_x, obs_y)
    for x in range(min(5,abs(obs_x - curr_x))):
        if obs_x > curr_x:
            x_val = curr_x + x
        else:
            x_val = curr_x - x

        y_val = int((obs_y - curr_y) / (obs_x - curr_x) * (x_val - curr_x) + curr_y)

        # Update point_map
        point_map[min(x_val, 49), min(y_val, 49)] = 1


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star(point_map, start, goal):
    pq = PriorityQueue()
    pq.put((0, [start]))
    visited = set()
    g_cost = {start: 0}

    while not pq.empty():
        current_cost, path = pq.get()
        current_node = path[-1]

        if current_node == goal:
            return path

        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = current_node[0] + dx, current_node[1] + dy
            new_cost = g_cost[current_node] + 1
            if 0 <= nx < len(point_map) and 0 <= ny < len(point_map[0]) and point_map[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                g_cost[(nx, ny)] = new_cost
                f_cost = new_cost + heuristic((nx, ny), goal)
                pq.put((f_cost, path + [(nx, ny)]))

    return None


def slam(event):
    #curr_x, curr_y = 25, 0
    goal_x, goal_y = 25, 15 
    point_map = np.zeros((map_size, map_size))
    #point_map[goal_x, goal_y] = 9
    obs_y = 0
    obs_x = 0
    global orientation
    global pause_event

    while (curr_x, curr_y) != (goal_x, goal_y):
        for i in range(-90, 90, 5):
            
            # Convert degrees to radians
            angle_radians = np.radians(i + 90 + orientation) 
            
            # Calculate sine and cosine
            

            dist = fc.get_distance_at(i)
            if dist >= 50:
                dist = -3
            else:
                dist/=7.5
            
            print("Distance at ",i + 90,"degree is ",dist)
            # this checks for if the car is in bounds and  ignores the obstacles outside of the boundaries
            # add offset for each search direction (right,left,top,bottom)
            #obs_y = min(max(0,int(dist*np.sin(angle_radians))),49)
            #obs_x = min(max(0,int(dist*np.cos(angle_radians))),49)
            if dist >= 0:
                obs_x = int(((dist*np.cos(angle_radians)) + curr_x))
                obs_y = int(((dist*np.sin(angle_radians)) + curr_y))

            print("Obstacle at (",obs_x," ,",obs_y,")","is ",dist, "cm away from the car")
            if dist >= 0 and obs_x < map_size and obs_y < map_size and obs_x >=0 and obs_y>=0 and point_map[obs_x, obs_y] != 2:  
                point_map[obs_x, obs_y] = 1
            time.sleep(0.09)
            point_map = padPointMap(point_map)
                #print(point_map)
                # interpolation
                #if counter > 5:
                #interpolate(point_map, curr_x, curr_y, obs_x, obs_y)
            np.savetxt('my_array.txt', np.rot90(point_map), fmt='%d', delimiter=', ')

        #path = bfs(point_map, (curr_x, curr_y), (goal_x, goal_y))
        path = a_star(point_map, (curr_x, curr_y), (goal_x, goal_y))
        
        if not path:
            print("No path found")
            break
        #elif ~pause_event.wait():
        elif  not event.is_set():
        
            next_x, next_y = path[1]
            # Code to move the car to (next_x, next_y)
            move_car(next_x, next_y)
            #curr_x, curr_y = next_x, next_y
            print("Curr: (", curr_x,",",curr_y,")")
            #print("Path1: (", next_x,",",next_y,")")

            print("Full Path: (", path,")")
            point_map[curr_x, curr_y] = 2
            time.sleep(0.1)


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool,event) -> None:
         
  #global pause_event
  #pause_event.set() 
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  
  detectStartTime = 0
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 0)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    
    
    for detection in detection_result.detections:
      for category in detection.categories:

        if (category.category_name=="stop sign" and time.time()>(detectStartTime+30)):
          print("stop sign detected!")
          #pause_event.clear()
          #calls the stop sign wait func
          p3.start()
          
    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def object_detect_func(event):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU),event)
  

# t1 = threading.Thread(target=slam)
# t2 = threading.Thread(target=object_detect_func)

# t1.start()
# t2.start()

# t1.join()
# t2.join()
  
stop_event = Event()

p1 = Process(target=slam, args=(stop_event,))
p2 = Process(target=object_detect_func, args=(stop_event,))
p3 = Process(target=stopSignWait, args=(10,))

p1.start()
p2.start()

p1.join()
p2.join()
