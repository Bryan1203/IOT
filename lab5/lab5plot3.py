import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from JSON file
filename = input("Enter the path to the JSON file: ")
with open(filename, 'r') as file:
    data = json.load(file)

# Extract data for each animal
animal_ids = [key for key in data.keys() if key.startswith('Lion') or key.startswith('Elephant')]

fig_cdf, ax_cdf = plt.subplots()
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection='3d')

for animal_id in animal_ids:
    animal_coords = data[animal_id]['gps coordinates']
    
    # Calculate distances between consecutive points
    distances = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for (x1,y1), (x2,y2) in zip(animal_coords[:-1], animal_coords[1:])]
    
    # Calculate speeds assuming 1 second between each data point
    speeds = distances
    
    # Calculate CDF of speeds
    sorted_speeds = np.sort(speeds)
    p = 1. * np.arange(len(speeds)) / (len(speeds) - 1)
    
    # Plot CDF
    ax_cdf.plot(sorted_speeds, p, label=animal_id)
    
    # Extract x,y coordinates and calculate cumulative time for z
    x = [coord[0] for coord in animal_coords]
    y = [coord[1] for coord in animal_coords]
    z = np.arange(len(x))
    
    # Plot 3D movement
    ax_3d.plot(x, y, z, label=animal_id)

ax_cdf.set_xlabel('Speed')
ax_cdf.set_ylabel('CDF')
ax_cdf.set_title("CDF of Animals' Movement Speed")
ax_cdf.legend()

ax_3d.set_xlabel('Longitude')
ax_3d.set_ylabel('Latitude')
ax_3d.set_zlabel('Time')
ax_3d.set_title("Animals' Movement")
ax_3d.legend()

fig_hr_loc = plt.figure()
ax_hr_loc = fig_hr_loc.add_subplot(111, projection='3d')

fig_hr_temp, ax_hr_temp = plt.subplots()

fig_hr_o2, ax_hr_o2 = plt.subplots()

for animal_id in animal_ids:
    animal_coords = data[animal_id]['gps coordinates']
    heart_rates = data[animal_id]['heart_rate']
    temperatures = data[animal_id]['temperature']
    oxygen_saturations = data[animal_id]['oxygen_saturation']
    
    # Extract x,y coordinates
    x = [coord[0] for coord in animal_coords]
    y = [coord[1] for coord in animal_coords]
    
    # Plot heart rate vs location
    ax_hr_loc.scatter(x, y, heart_rates, label=animal_id)
    
    # Plot heart rate vs temperature
    ax_hr_temp.scatter(temperatures, heart_rates, label=animal_id)
    
    # Plot heart rate vs oxygen saturation
    ax_hr_o2.scatter(heart_rates, oxygen_saturations, label=animal_id)

ax_hr_loc.set_xlabel('Longitude')
ax_hr_loc.set_ylabel('Latitude')
ax_hr_loc.set_zlabel('Heart Rate')
ax_hr_loc.set_title("Heart Rate vs Location")
ax_hr_loc.legend()

ax_hr_temp.set_xlabel('Temperature')
ax_hr_temp.set_ylabel('Heart Rate')
ax_hr_temp.set_title("Heart Rate vs Temperature")
ax_hr_temp.legend()

ax_hr_o2.set_xlabel('Heart Rate')
ax_hr_o2.set_ylabel('Oxygen Saturation')
ax_hr_o2.set_title("Heart Rate vs Oxygen Saturation")
ax_hr_o2.legend()

plt.show()