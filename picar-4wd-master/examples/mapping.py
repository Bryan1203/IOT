import time
import picar_4wd as fc
import numpy as np
import math
from queue import Queue

speed = 30
map_size = 50
orientation = 0

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
    time.sleep(0.8)
    fc.stop()

def goBackward(): 
    fc.backward(1)
    time.sleep(0.1)
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

def move_car(curr_x, curr_y, next_x, next_y):
    global orientation
    if next_x == curr_x + 1:  # Move right
        if orientation == 0:
            goRight()
            goForward()
        elif orientation == -90: 
            goForward()
        elif orientation == 180:
            goLeft()
            goForward()
        elif orientation == 90:
            goRight()
            goRight()
            goForward()
        print("going Right")

    elif next_x == curr_x - 1:  # Move left
        if orientation == 0:
            goLeft()
            goForward()
        elif orientation == 90:
            goForward()
        elif orientation == 180:
            goRight()
            goForward()
        print("going Left")
    elif next_y == curr_y + 1:  # Move forward
        if orientation == 0:
            goForward()
        elif orientation == -90:
            goLeft()
            goForward()
        elif orientation == 90:
            goRight()
            goForward()
        elif orientation == 180:
           goBackward()
        print("going forward")
    elif next_y == curr_y - 1:  # Move backward
        if orientation == 180:
            goForward()
        elif orientation == -90:
            goRight()
            goForward()
        elif orientation == 90:
            goLeft()
            goForward()
        elif orientation == 0:
            goBackward()
            
            
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


def bfs(point_map, start, goal):
    q = Queue()
    q.put([start])
    visited = set()

    while not q.empty():
        path = q.get()
        x, y = path[-1]

        if(x, y) == goal:
            return path
        
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < map_size and 0 <= ny < map_size and point_map[nx, ny] == 0 and (nx, ny) not in visited:
                q.put(path + [(nx, ny)])
                visited.add((nx, ny))
    return None

def main():
    curr_x, curr_y = 25, 0
    goal_x, goal_y = 25, 8 
    point_map = np.zeros((map_size, map_size))
    obs_y = 0
    obs_x = 0
    global orientation

    while (curr_x, curr_y) != (goal_x, goal_y):
       

        for i in range(-90, 90, 5):
            
            # Convert degrees to radians
            angle_radians = np.radians(i + 90 + orientation) 
            
            # Calculate sine and cosine
            
            dist = fc.get_distance_at(i)/1.5
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
                #print(point_map)
                # interpolation
                #if counter > 5:
                #interpolate(point_map, curr_x, curr_y, obs_x, obs_y)
            np.savetxt('my_array.txt', np.rot90(point_map), fmt='%d', delimiter=', ')

        path = bfs(point_map, (curr_x, curr_y), (goal_x, goal_y))
        
        if not path:
            print("No path found")
            break
        else:
        
            next_x, next_y = path[1]
            # Code to move the car to (next_x, next_y)
            move_car(curr_x, curr_y, next_x, next_y)
            curr_x, curr_y = next_x, next_y
            print("Curr: (", curr_x,",",curr_y,")")
            #print("Path1: (", next_x,",",next_y,")")

            print("Full Path: (", path,")")
            point_map[curr_x, curr_y] = 2
            time.sleep(0.1)
if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()
