#---- Append function directory to the file path ----#
linux = 1
if linux:
    function_dir = '/home/aapps/Documents/other/Honours/ENGN4350_Honours/subsequence_dtw/'
else:
    function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'

import sys
sys.path.append(function_dir)

#---- Import Modules ----#
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import os
import _functions.subsequence_dtw_functions as subsequence_dtw_functions
import _functions.determine_ground_truth as determine_ground_truth

# mat_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32_2/'
# output_data_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32_2/processed/'

# mat_dir = '/media/aapps/Elements/Data/Brisbane_Event_Data/mat_files/'
# output_data_dir ='/media/aapps/Elements/Data/Output_Data/31-10-2023/'

if linux: 
    mat_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32_2/'
    output_data_dir = '/media/aapps/Elements/Data/Output_Data/31-10-2023/'
else:
    mat_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32_2/'
    output_data_dir ='F:/Data/Output_Data/31-10-2023/'

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

#--- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'

#---- Load Data ----#
reference_end_time = 90
reference_name = 'sunset1'
reference_full_dataset = loadmat(mat_dir + filename_dict[reference_name])['data']

reference_full_dataset[:,0] -= reference_full_dataset[0,0]
reference_full_dataset = subsequence_dtw_functions.select_data_sequence(reference_full_dataset, -1, reference_end_time)


query_name_list = ['night', 'sunset2', 'daytime', 'morning']

for query_name in query_name_list:

    query_full_dataset = loadmat(mat_dir + filename_dict[query_name])['data']
    # Convert time
    query_full_dataset[:,0] -= query_full_dataset[0,0]
    print(f"Query: {query_name} \nReference Duration: {reference_full_dataset[-1,0]} \t Query Duration: {query_full_dataset[-1,0]}")


    #---- Batch Parameters ----#
    data_selection = 3 # dt, x, y, pol

    query_length = 0.5
    query_time_array = np.linspace(15, reference_full_dataset[-1,0]-15, 20)

    # Storage Arrays
    headers = ['query_start', 'query_end', 'query_points', 'reference_points' ,'estimated_start', 'estimated_end', 'query_position_0', 'query_position_1', 'estimated_position_0', 'estimated_position_1', 'groundtruth_index', 'distance','runtime']
    df = pd.DataFrame(columns=headers)
    df_filename = output_data_dir + f'{reference_name}_{query_name}_batch_data.csv'

    df_batch = pd.DataFrame()
    loc_index = 0


    #---- Perform Batch ----#
    for query_start_time in query_time_array:
        query_end_time = query_start_time + query_length

        # Select data (dt,x,y,pol)
        query_batch_sequence = subsequence_dtw_functions.select_data_sequence(query_full_dataset, query_start_time, query_end_time)
        query_data, reference_data = subsequence_dtw_functions.select_data(query_batch_sequence, reference_full_dataset, data_selection)

        #---- Perform DTW ----#
        start_time = time.time()
        _, _, P = subsequence_dtw_functions.subsequence_dtw(query_data, reference_data, print_en=0)
        dtw_time = time.time() - start_time

        a_ast = P[0, 1]
        b_ast = P[-1, 1]

        estimated_start_time = reference_full_dataset[a_ast, 0]
        estimated_end_time = reference_full_dataset[b_ast, 0]

        # get some metrics
        reference_path, query_position, estimated_postion, distance, groundtruth_index = determine_ground_truth.calc_ground_truth_interp(query_name, query_end_time, reference_name, estimated_end_time, linux=linux)

        pandas_data = [query_start_time, query_end_time, query_data.shape[0], reference_data.shape[0] ,estimated_start_time, estimated_end_time, query_position[0], query_position[1], estimated_postion[0], estimated_postion[1], groundtruth_index, distance, dtw_time]
        df.loc[0] = pandas_data

        if os.path.exists(df_filename):
            df.to_csv(df_filename, index=False, header=False ,mode='a') # append if file exists
        else:
            df.to_csv(df_filename, index=False, mode='w') # write if the file doesn't exit



    

