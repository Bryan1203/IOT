import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import re

# Get a list of all text files
file_list = glob.glob('./point_map_txt_good_two/*.txt')  # replace with your path

# Create a color map for the different values in the array
cmap = mcolors.ListedColormap(['white', 'black', 'red', 'blue'])

# Create a normalized boundary map from values to colors
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Process each file
for file_path in file_list:
    # Load the text file into a numpy array
    arr = np.loadtxt(file_path,delimiter=',')

    # Extract the sequence number from the file name
    sequence_number = re.search(r'\d+', file_path).group()

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap=cmap, norm=norm)

    # Save the plot
    plt.savefig(f'output_{sequence_number}.png')

    # Close the plot
    plt.close()
