import sys
function_dir = '/home/aapps/Documents/other/Honours/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import os
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat, savemat
import psutil
import gc
from tqdm import tqdm
import _functions.subsequence_dtw_functions as subsequence_dtw_functions
import _functions.determine_ground_truth as determine_ground_truth
import _utilities.organise_files as organise_files
import argparse


parser = argparse.ArgumentParser(description='Perform batch subsequence DTW on the compressed data.')
parser.add_argument('-b', '--batch', type=str, nargs='?', required=True, help='Batch directory name')
args = parser.parse_args()
batch = args.batch


#---- Parameters ----#

query_filename = 'sunset2'
reference_filename = 'sunset1'
data_selection = 3

query_time_array = np.arange(15, 600, 10)
query_length = 1

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat'}
 
# camera parameters
rows = 260
cols = 346


#---- Load Data ----#
comp_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/batch/'
output_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/batch/output_ds1_3s/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# else:
#     action = input('Output Directory Exists: (0) to exit, (1) to proceed \n')
#     if action == '0':
#         print('Exiting \n')
#         exit()


#---- Setup of Storage Arrays ----#
# files = os.listdir(comp_dir)
# files.remove('output') # remove the output folder from the list
# files = sorted(files, key=organise_ficd _bles.get_resolution_and_threshold) # sort by resolution and threshold


# # can remove files from list if they are already done
# files_to_discard = 'resolution_[50-50]'
# files = [item for item in files if not item.startswith(files_to_discard)]
 
 
#---- Perform SubDTW on all the files ----#
# for batch in files:
    # create save directory for the batch file
comp_dir_batch = comp_dir + batch
output_dir_batch = output_dir + batch
if not os.path.exists(output_dir_batch):
    os.mkdir(output_dir_batch)

# get the threshold and resolution from the directory name    
threshold = int(batch.split('_')[-1])
resolution_temp = batch.split('_')[1]
resolution = int(resolution_temp.split('-')[-1][:-1])

# load the full datasets for the batch
query_full = loadmat(comp_dir_batch + '/' + filename_dict[query_filename])['data']
reference_full = loadmat(comp_dir_batch + '/' + filename_dict[reference_filename])['data']

print(f'Batch: {batch} \t Reference: {reference_full.shape} \t Query: {query_full.shape}')

# Set up saving
accumulated_cost_filename = output_dir_batch + '/' + 'accumulated_cost.mat' 
df_filename = output_dir_batch + '/' + 'batch_data.csv' 


# batch storage arrays
accumulated_cost = {}
# create pandas data frame
headers = ['Resolution', 'Threshold', 'Reference', 'Query', 'Data Selection', 'Query Start (s)', 'Query End (s)', 'Query Points', 'Reference Points' ,'Estimated Start (s)', 'Estimated End (s)', 'Distance (m)','Runtime (s)','Memory (B)', 'DTW Successful']
df_batch = pd.DataFrame(columns=headers)
loc_index = 0

for query_start in tqdm(query_time_array, 'Performing each time:'):
    query_end = query_start + query_length   

    #---- Select the Query ----#        
    query = subsequence_dtw_functions.select_data_sequence(query_full, query_start, query_end)
    # Select the variables used for DTW matching
    #   0 - x, y 
    #   1 - x, y, sig
    #   2 - x_norm, y_norm, sig 
    #   3 - dt, x, y, sig         print(df)
    #   4 - dt, x_norm, y_norm, sig
    #   5 - dt, sig
    query_dtw, reference_dtw = subsequence_dtw_functions.select_data(query, reference_full, data_selection) # dt, x, y, sig

    #---- Perform DTW and Analyse Output ----#
    # check resource requirements
    available_memory = psutil.virtual_memory().available # bytes
    required_memory = (query_dtw.shape[0])*(reference_dtw.shape[0])*8 # memory required for D array (in bytes)
    # print(f'Memory Available: {available_memory / 1e9} \t Memory Required: {required_memory /1e9}')

    if available_memory - required_memory > 1e9: # perform DTW if there is enough RAM
        start_time = time.time()
        _, D, P = subsequence_dtw_functions.subsequence_dtw(query_dtw, reference_dtw, print_en=0)
        elapsed_time = time.time() - start_time
        a_ast = P[0, 1]
        b_ast = P[-1, 1]
        batch_cost = D[-1,:]

        # release variables
        del D, P

        reference_match_start = reference_full[a_ast, :]
        reference_match_end = reference_full[b_ast, :]

        # groundtruth_path, query_position, estimated_postion, distance = determine_ground_truth.calc_ground_truth(query_filename, query_end, reference_filename, reference_match_end[0])

        groundtruth_result =  determine_ground_truth.calc_ground_truth(query_filename, query_end, reference_filename, reference_match_end[0])

        if groundtruth_result != -1:
            groundtruth_path, query_position, estimated_postion, distance = groundtruth_result
            accumulated_cost_dict_label = f'{query_start}-{query_end}'
            accumulated_cost[accumulated_cost_dict_label] = batch_cost

            #---- Add data to Pandas Dataframe ----#
            # headers: ['Resolution', 'Threshold', 'Reference', 'Query', 'Data Selection', 'Query Start', 'Query End', 'Query Points', 'Reference Points' ,'Estimated Start', 'Estimated End', 'Distance', 'Runtime', 'Memory', 'DTW Successful']
            df_data = [resolution, threshold, reference_filename, query_filename, data_selection, query_start, query_end, query.shape[0], reference_full.shape[0], reference_match_start[0], reference_match_end[0], distance, elapsed_time, required_memory, 1]
        else:
            df_data = [resolution, threshold, reference_filename, query_filename, data_selection, query_start, query_end, query.shape[0], reference_full.shape[0], reference_match_start[0], reference_match_end[0], -1, elapsed_time, required_memory, 0]


    else:
        df_data = [resolution, threshold, reference_filename, query_filename, data_selection, query_start, query_end, query.shape[0], reference_full.shape[0], -1, -1, -1, -1, required_memory, 0]

    df_batch.loc[0] = df_data
    # loc_index += 1 # increment df.loc index

    if os.path.exists(df_filename):
        df_batch.to_csv(df_filename, index=False, header=False ,mode='a') # append if file exists
    else:
        df_batch.to_csv(df_filename, index=False, mode='w') # write if the file doesn't exit



#---- Save the Accumulated Cost ----#
data = accumulated_cost
accumulated_cost_filename = output_dir_batch + '/' + 'accumulated_cost.mat' 
savemat(accumulated_cost_filename, data, do_compression=False)
print("Accumulated cost saved")

# #---- Save the Pandas Data to csv ----#
# df_filename = output_dir_batch + '/' + 'batch_data.csv' 
# df_batch.to_csv(df_filename, index=False)
# print("Data csv saved")

# del reference_full, query_full
