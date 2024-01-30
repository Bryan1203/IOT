import numpy as np
import time
import picar_4wd as fc
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


def polar_to_cartesian(angle, distance):
    angle_rad = np.radians(angle)
    x = distance * np.cos(angle_rad)
    y = distance * np.sin(angle_rad)
    return int(x), int(y)

def create_map(ultrasonic_sensor, map_size=(100, 100), car_position=(50, 0)):
    map_array = np.zeros(map_size)

    for angle in range(-ultrasonic_sensor.max_angle, ultrasonic_sensor.max_angle, ultrasonic_sensor.STEP):
        ultrasonic_sensor.servo.set_angle(angle)  
        time.sleep(0.04) 
        distance = ultrasonic_sensor.get_distance() 

        if distance > 0:  
            x, y = polar_to_cartesian(angle, distance) 
            map_x = car_position[0] + x  
            map_y = car_position[1] + y

            if 0 <= map_x < map_size[0] and 0 <= map_y < map_size[1]:
                map_array[map_y, map_x] = 1  

    return map_array

def calculate_delay(distance, speed):
    if speed <= 0:
        return 0 
    time_delay = distance / speed
    return time_delay

def follow_path(path, motor):
    for i in range(len(path) - 1):
        current_point = path[i]
        next_point = path[i + 1]

        turn_angle, distance = calculate_turn_and_distance(current_point, next_point)

        if turn_angle > 0:
            fc.turn_right(turn_angle) 
        elif turn_angle < 0:
            fc.turn_left(-turn_angle)  
        
        fc.forward(distance)  

        time.sleep(calculate_delay(distance, 10))

def main():
    curr_x = 25
    curr_y = 0
    start = (curr_x, curr_y)
    goal = (75, 75)

    point_map = np.zeros((50, 50))   
    
    while True:
        for angle in range(-90, 90, 5):
            # Rest of your loop with modifications
            # Convert degrees to radians
            angle_radians = np.radians(i + 90)
            dist = fc.get_distance_at(angle)

            if dist != -1:
                obs_x, obs_y = polar_to_cartesian(angle, dist)
                curr_x = curr_x + obs_x
                curr_y = curr_y + obs_y
                if 0 <= map_x < 100 and 0 <= map_y < 100:
                    point_map[map_y, map_x] = 1  # Update the map
            
            path = navigate(point_map, start, goal)
            if path:
                follow_path(path, motor)  # Move the robot along the path
            else:
                print("No path found")
                break  # Break the loop if no path is found

            # Check if the goal is reached
            if (curr_x, curr_y) == goal:
                print("Goal reached")
                break
            
            np.savetxt('my_array.txt', np.rot90(point_map), fmt='%d', delimiter=', ')
            
            time.sleep(0.1)

      

if __name__ == '__main__':
    main()
