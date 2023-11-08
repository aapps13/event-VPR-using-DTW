import sys 
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy.io import loadmat
import os
from _functions.determine_ground_truth import calc_ground_truth
from _functions.visualisation import event_visualisation, play_video_frame
import argparse

# file_dir = 'sunset1_morning_(1)/'
file_dir = 'resolution_[5-5]_threshold_100/'

reference_name = 'sunset1'
query_name = 'sunset2'

parser = argparse.ArgumentParser(description='Visualise the outputs of the Subsequence DTW localisation algorithm')
parser.add_argument('-f', '--file_directory', type=str, nargs='?', required=False, help='The spacing used for pixel filtering')
args = parser.parse_args()

if args.file_directory:
    file_dir = args.file_directory + '/'


#--- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'


## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##

filename_dict = {'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                 'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                 'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                 'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                 'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                 'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

video_offset = {'sunset1' : 5,
                'sunset2' : -5.5,
                'daytime' : 0,
                'sunrise' : 4.5,
                'morning' : 0,
                'night'    : 2}

# reference_name = file_dir.split('_')[0]
# query_name = file_dir.split('_')[1][:]

file_dir_split = file_dir.split('_')
resolution = file_dir_split[1]
threshold = file_dir_split[3][:-1]

# data_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/localisation/' + file_dir

data_dir = 'F:/Data/Output_Data/_localisation/sunset1_sunset2/' + file_dir
mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'


#---- Load Data ----#
covariance_array = np.load(data_dir + 'covariance_array.npy')
localisation_data = np.load(data_dir + 'localisation_data.npy', allow_pickle=True)
reference_map = np.load(data_dir + 'map.npy')

#---- Process Data ----#
# Unpack the covariance matrix
x_cov = np.zeros((covariance_array.shape[0],1))
for i in range(covariance_array.shape[0]):
    x_cov[i] = covariance_array[i][0,0]

# # Unpack the localisation data
# query_time = localisation_data[:,0]
# esimated_time = localisation_data[:,1]
# query_position = localisation_data[:,2:4]
# estimated_position = localisation_data[:,4:6]
# distance = localisation_data[:,6]
# query_window = localisation_data[:,7:9]
# reference_window = localisation_data[:,9:11]

# Unpack the localisation data
'''
query_start, query_end, reference_start, reference_end, query_lat, query_lon, reference_lat, reference_lon, distance, accumulated_cost, query_length
'''

query_time = localisation_data[:,1]
estimated_time = localisation_data[:,3]
query_position = localisation_data[:,4:6]
estimated_position = localisation_data[:,6:8]
distance = localisation_data[:,8]
accumulated_cost = localisation_data[:,9]
query_length = localisation_data[:,10]


#---- Create Figures ----#
# Covariance
fig, ax = plt.subplots()#figsize=(10,8))
fig.subplots_adjust(top=0.9)
fig.suptitle('Evolution of Position Covariance', fontweight='bold', fontsize=16)
ax.set_title(f'Query: {query_name}    Reference: {reference_name}    Resolution: {resolution}    Threshold: {threshold}')
ax.plot(x_cov, marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
ax.set_xlabel('Iteration', fontweight='bold', fontsize=12)
ax.set_ylabel('Covariance', fontweight='bold', fontsize=12)
ax.grid(which='both')
plt.show()

# Position and Accumulated Error
fig, ax = plt.subplots(1,2)#figsize=(10,8))
fig.canvas.setWindowTitle('position_error_and_accumulated_cost')
fig.suptitle('Comparison of Estimated Position Error and Accumulated Cost', fontweight='bold', fontsize=16)
fig.subplots_adjust(top=0.85)
fig.text(0.5, 0.92, f'Query: {query_name}    Reference: {reference_name}    Resolution: {resolution}    Threshold: {threshold}', ha='center', fontsize=12, fontweight='normal')
ax[0].plot(distance, marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
ax[0].set_title('Distance between Ground Truth and Estimate', fontweight='bold')
ax[0].set_xlabel('Iteration', fontweight='bold', fontsize=12)
ax[0].set_ylabel('Distance(m)', fontweight='bold', fontsize=12)
ax[0].grid(which='both')

ax[1].plot(accumulated_cost/query_length, marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
ax[1].set_title('Normalised Accumulated Cost', fontweight='bold')
ax[1].set_xlabel('Iteration', fontweight='bold', fontsize=12)
ax[1].set_ylabel('Cost (.arb)', fontweight='bold', fontsize=12)
ax[1].grid(which='both')
plt.show()

# Localisation
colors = np.linspace(0, 1, len(query_position))
fig, ax = plt.subplots(1,2,figsize=(12,6))
fig.canvas.setWindowTitle('localisation')
fig.suptitle('Subsequence DTW: Localisation Algorithm', fontweight='bold', fontsize=16)
fig.text(0.5, 0.92, f'Query: {query_name}    Reference: {reference_name}    Resolution: {resolution}    Threshold: {threshold}', ha='center', fontsize=12, fontweight='normal')
fig.subplots_adjust(wspace=0.4, bottom=0.2)
ax[0].scatter(query_position[:,1], query_position[:,0],  c=colors, cmap='viridis', marker='*', label='Query Position', zorder=2 )
ax[0].scatter(estimated_position[:,1], estimated_position[:,0],  c=colors, cmap='viridis', marker='^', label='Estimated Position', zorder=1)
ax[0].plot(reference_map[:,1], reference_map[:,0], color='k', alpha=0.5, zorder=0)
ax[0].grid(which='both')
ax[0].set_xlabel('Latitude (deg)', fontweight='bold')
ax[0].set_ylabel('Longitude (deg)', fontweight='bold')
ax[0].set_xticklabels(ax[0].get_xticks(), rotation=90)
ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))

# plot points with matching colours
ax[1].scatter(query_position[:,1], query_position[:,0], c=colors, cmap='viridis', marker='*', s=50, zorder=2)
ax[1].scatter(estimated_position[:,1], estimated_position[:,0],c=colors, cmap='viridis', marker='^', s=50, zorder=2)

# plot a line connecting corresponding points
for i in range(len(query_position)):
    ax[1].plot([query_position[i,1], estimated_position[i,1]], [query_position[i,0], estimated_position[i,0]], color='k', linestyle='--', alpha=0.5, zorder=1)

ax[1].plot(reference_map[:,1], reference_map[:,0], color='k', alpha=0.5, zorder=0)
x_concat = np.concatenate((query_position[:,1],estimated_position[:,1]))
y_concat = np.concatenate((query_position[:,0],estimated_position[:,0]))
ax[1].set_xlim([min(x_concat)-1e-4, max(x_concat)+1e-4])
ax[1].set_ylim([min(y_concat)-1e-4,max(y_concat)+1e-4])
ax[1].set_xlabel('Latitude (deg)', fontweight='bold')
ax[1].set_ylabel('Longitude (deg)', fontweight='bold')
ax[1].set_xticklabels(ax[1].get_xticks(), rotation=90)
ax[1].xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
ax[1].grid(which='both')
ax[1].legend(['True Postion', 'Estimate'])
plt.show()


