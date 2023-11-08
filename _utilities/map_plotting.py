###########################################################################
####                           Import Modules                         #####
###########################################################################
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

#---- Import Modules ----#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import folium

data_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/original_analysis/sunset1_sunset2_0.5(1)/'
filename = 'sunset1_sunset2_0.5_data_filtered_2.csv'

gps_interp_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_full_data_interp/'
gps_file_dict = {'sunset1' : 'sunset1_gps.csv',
                 'sunset2' : 'sunset2_gps.csv'}

query_name = 'sunset2'
reference_name = 'sunset1'

#---- Plot Params ----#
suptitle_weight = 'bold'
suptitle_size = 16
title_weight = 'bold'
title_size = 14
label_weight = 'normal'
label_size = 14

plot_linewidth = 2
plot_boxwidth = 1.5

#---- Load Data ----#
# load DTW batch values 
df = pd.read_csv(data_dir + filename)

query_positions = [df['query_position_0'],df['query_position_1']]
query_positions = np.asarray(query_positions).T
estimated_positions = [df['estimated_position_0'],df['estimated_position_1']]
estimated_positions = np.asarray(estimated_positions).T

# load the reference map
reference_start_time = 0
reference_end_time = 120
reference_map = np.genfromtxt(gps_interp_dir + gps_file_dict[reference_name], delimiter=',')
reference_path = reference_map[:,1:3]


#---- Mapping ----#
# Define map parameters
map_lat = reference_path[:50,0]
map_lon = reference_path[:50,1]

min_lat = min(map_lat)
max_lat = max(map_lat)
median_lat = np.median(map_lat)
min_lon = min(map_lon)
max_lon = max(map_lon)
median_lon = np.median(map_lon)

# Create map
zoom = 18
m = folium.Map(min_zoom=zoom, max_bounds=True, location=[median_lat, median_lat], zoom_start=zoom, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)

# Create path
reference_map = folium.PolyLine(locations=reference_path, color='black', weight=2, alpha=0.5)

# Add features to map
reference_map.add_to(m)

for i in range(query_positions.shape[0]):
    query_marker = folium.CircleMarker(location=[query_positions[i,0], query_positions[i,1]], radius=4, color='red', fill=True, fill_color='red',fill_opacity=1)
    query_marker_label = folium.Popup('Query', parse_html=True)
    query_marker.add_to(m)
    query_marker.add_child(query_marker_label)

    estimate_marker = folium.CircleMarker(location=[estimated_positions[i,0], estimated_positions[i,1]], radius=4, color='blue', fill=True, fill_color='blue',fill_opacity=1)
    estimate_marker_label = folium.Popup('Query', parse_html=True)
    estimate_marker.add_to(m)
    estimate_marker.add_child(estimate_marker_label)

    points = np.array([[query_positions[i,0], query_positions[i,1]],
                       [estimated_positions[i,0], estimated_positions[i,1]]])
    linking_map = folium.PolyLine(locations=points, color='orange', weight=3)#, dash_array=[5,5])
    linking_map.add_to(m)


# Convert the map to an image
print('Converting the map to an image')
img_data = m._to_png(1)
img = Image.open(io.BytesIO(img_data))


#---- Plotting ----#
fig, ax = plt.subplots(figsize=(12,6))
fig.subplots_adjust(top=0.942,bottom=0.038,left=0.01,right=0.99,hspace=0.19,wspace=0.2)
fig.suptitle("SubDTW using Spatio-Temporal Pooling: Every 10th Pixel", fontweight=suptitle_weight, fontsize=suptitle_size)
ax.set_title("Sunset1-Sunset2: Matching using a 0.5s Query in a 30s Reference", fontweight=title_weight, fontsize=title_size)
fig.tight_layout()
ax.imshow(img)
ax.scatter(0, 0, s=200, label='Query Position', alpha=1, color='red')
ax.scatter(0, 0, s=200, label='Estimated Position', alpha=1, color='blue')

ax.set_ylim((100,500))
ax.set_xlim((200,1000))

ax.legend(fontsize=label_size,loc='lower left')
ax.invert_yaxis()
ax.axis('off')
plt.show()