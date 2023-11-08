import sys 
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import os
from _functions.subsequence_dtw_functions import filter_data

## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##

data_dir = "C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/original_analysis/sunset1_sunset2_0.5(1)/"
mat_dir = "C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/"

#---- Load Subsequence DTW Results ----#
# This section loads the P_mat files from the batch output directory in a dictionary.
# The variable name is sliced from the .npy file so that it can be associated to the
# correct data

subDTW_data = {}
files = os.listdir(data_dir)
files = [file for file in files if file.endswith('.npy')]

for f in files:
    test_time_str = f[f.find('_', 2)+2:-5] # slice out the number section of title
    start = (test_time_str[:test_time_str.find('-')])
    # start = start[:start.find('.')]
    end = (test_time_str[test_time_str.find('-')+1:])
    # end = end[:end.find('.')]
    var_name = f"{start} to {end}"
    subDTW_data[var_name] = np.load(os.path.join(data_dir, f), 'r')


#---- Unpack Parameters ----#
filter_gap = 10

# camera parameters
rows = 260
cols = 346


#--- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'


#---- Load Data ----#
# Load sunset data
reference_full = loadmat(f'{mat_dir}dvs_vpr_2020-04-21-17-03-03')['data']
query_full = loadmat(f'{mat_dir}dvs_vpr_2020-04-29-06-20-23')['data']

# Create a copy to filter.
reference = reference_full
query = query_full

data_dict = {'reference':reference, 
             'query':query}


#---- Filter the data ----#
reference = filter_data(reference_full, filter_gap)
query = filter_data(query_full, filter_gap)

#---- Plotting ----#
xlim_upper = np.where(reference[:,0] >= 30)[0][0]

fig, ax = plt.subplots(1,1,figsize=(5,5))
fig.suptitle("Sliding Query with Subsquence DTW Analysis", fontweight='bold', fontsize=16)
fig.subplots_adjust(top=0.95)
fig.text(0.5, 0.01, f'Query: Sunset2     Reference: Sunset1', ha='center', fontsize=12)

# Combined
for i in range(len(subDTW_data)):
    lab = list(subDTW_data)[i]
    vals = list(subDTW_data.values())[i]
    ax.plot(vals[:,1], vals[:,0], label = lab)
    ax.vlines(vals[-1,1], 0, vals[-1,0], colors='k', linestyles='--', alpha=0.5)

ax.set_xlabel('Reference (Index)')
ax.set_ylabel('Query (Index)')
ax.set_xlim([0, xlim_upper])
ax.grid(True)
legend = ax.legend()
legend.set_title('Query Time (s)')

plt.show()