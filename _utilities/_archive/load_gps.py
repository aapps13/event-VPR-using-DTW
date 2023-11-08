import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def remap_vector(vector, old_min, old_max, new_min, new_max):
    # Calculate the range of the old vector
    old_range = old_max - old_min
    
    # Calculate the range of the new vector
    new_range = new_max - new_min
    
    # Initialize an empty list to store the remapped values
    remapped_vector = []
    
    # Loop through each element in the vector and remap it to the new range
    for value in vector:
        # Calculate the proportion of the current value in the old range
        proportion = (value - old_min) / old_range
        
        # Calculate the value in the new range using the proportion
        new_value = new_min + (proportion * new_range)
        
        # Add the remapped value to the new vector
        remapped_vector.append(new_value)
    
    return remapped_vector






filepath = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/daytime_gps.csv'
map_img_path = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/map.png'

#---- Load Data ----#
data = np.loadtxt(filepath, delimiter=',')
data[:,0] = -data[:,0]
img = plt.imread(map_img_path)


max_lon = -27.49
min_lon = -27.52
max_lat = 152.95
min_lat = 152.90

new_width = img.shape[0]
new_height = img.shape[1]

new_lat = remap_vector(data[:,1], min_lat, max_lat, 0, new_width)
new_lon = remap_vector(data[:,0], min_lon, max_lon, 0, new_height)

plt.figure()
plt.imshow(img)
plt.plot(new_lon, new_lat)
# plt.xlim(new_width)
# plt.ylim(new_height)

plt.show()
