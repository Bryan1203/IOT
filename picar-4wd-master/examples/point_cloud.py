import time
import picar_4wd as fc
import numpy as np
import math

speed = 30
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

def main():
    curr_x = 25
    curr_y = 0
    point_map = np.zeros((50, 50))
    while True:
        counter = 0
        for i in range(-90, 90, 18):
            
            # Convert degrees to radians
            angle_radians = np.radians(i + 90)

            # Calculate sine and cosine
            obs_y = 0
            obs_x = 0
            prev_y = obs_y
            prev_x = obs_x
            dist = fc.get_distance_at(i)
            print("Distance at ",i + 90,"is ",dist)
            obs_y = min(max(0,int(dist*np.sin(angle_radians))+curr_y),49)
            obs_x = min(max(0,int(dist*np.cos(angle_radians))+curr_x),49)
            
            print("Distance at (",obs_x," ,",obs_y,")",i + 90,"is ",fc.get_distance_at(i))
            print("Speed: " , fc.speed_val())
            if dist != -1:  
                point_map[obs_x,obs_y] = 1
            time.sleep(0.09)
            #print(point_map)
            # interpolation
            #if counter > 5:
                #interpolate(point_map, curr_x, curr_y, obs_x, obs_y)
            
            np.savetxt('my_array.txt', np.rot90(point_map), fmt='%d', delimiter=', ')
            counter = counter +1
if __name__ == "__main__":
    try: 
        main()
    finally: 
        fc.stop()