import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import os
import numpy as np
from scipy.io import loadmat, savemat
from _functions.subsequence_dtw_functions import region_filter
import argparse

parser = argparse.ArgumentParser(description='Apply data compression')
parser.add_argument('-r', '--resolution', type=int, nargs=2, required=True, help='Data compression resolution')
args = parser.parse_args()
resolution_list = [args.resolution]


#---- Parameters ----#
threshold_array = np.arange(20, 101, 10) 

dt = 0.1

# resolution_list = [[5,5], [10,10], [15,15], [20,20], [25,25], [30,30], [35,35], [40,40], [45,45], [50,50], [55,55], [60,60]]
# resolution_list = [[30,30], [40,40], [50,50]]


filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}
 
# camera parameters
rows = 260
cols = 346

#---- Load Data ----#
curr_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/'
mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
comp_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/compressed_data/'

#---- Set up the saving directory ----#
# save_dir = comp_dir + f'/compressed_data_{threshold}_[{resolution[0]}-{resolution[1]}]/'

for resolution in resolution_list:
    print(f'Resolution: {resolution}')
    save_dir = comp_dir + f'/resolution_[{resolution[0]}-{resolution[1]}]/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # ---- Loop through each file  ----#
    for threshold in threshold_array:

        sub_save_dir = save_dir + f'threshold_{threshold}/'
        if not os.path.exists(sub_save_dir):
            os.mkdir(sub_save_dir)

        for key, value in filename_dict.items():
            print(f'Threshold: {threshold} \t Key: {key} \t Value: {value}')
            data_dir = mat_dir + value
            data = loadmat(data_dir)['data']
            filtered_data = region_filter(data, threshold, resolution, dt, cols=346, rows=260)
            data_dict = {'data' : filtered_data}
            savemat(sub_save_dir + value, data_dict)

            del data, filtered_data 




