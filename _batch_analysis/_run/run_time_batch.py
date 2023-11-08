#---- Append function directory to the file path ----#
linux = 0
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
    output_data_dir = '/media/aapps/Elements/Data/Output_Data/02-11-2023/'
else:
    mat_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32_2/'
    output_data_dir ='F:/Data/Output_Data/02-11-2023/'

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

if not os.path.exists(output_data_dir):
    user_input = input("Save directory does not exists, would you like to create it? Yes[1]/No[0]: \t")
    if user_input == "1":
        os.makedirs(output_data_dir)
        print("Save directory has been created")
    else:
        exit()
else:
    user_input = input("Save directory already exists, would you like to continue? Yes[1]/No[0]: \t")
    if user_input == '0':
        exit()

#---- Load Data ----#
reference_end_time = 90
reference_name = 'sunset1'
query_name = 'sunset2'

reference_full_dataset = loadmat(mat_dir + filename_dict[reference_name])['data']
query_full_dataset = loadmat(mat_dir + filename_dict[query_name])['data']

query_full_dataset[:,0] -= query_full_dataset[0,0]
reference_full_dataset[:,0] -= reference_full_dataset[0,0]
reference_full_dataset = subsequence_dtw_functions.select_data_sequence(reference_full_dataset, -1, reference_end_time)


#---- Batch Parameters ----#
data_selection = 3 # dt, x, y, pol


# Storage Arrays
headers = ['query_points', 'reference_points','runtime']
df = pd.DataFrame(columns=headers)
df_filename = output_data_dir + f'{reference_name}_{query_name}_batch_data.csv'

#---- Generate Points ----#
query_index_array = np.random.randint(0, high=query_full_dataset.shape[0]*0.8, size=100, dtype=int)
query_length_array = np.random.uniform(1000,2000,150).astype(int)


#---- Perform Batch ----#
for i in range(len(query_index_array)):
           
    query_start_index = query_index_array[i]
    query_length = query_length_array[i]
    query_end_index = query_start_index + query_length

    query_batch_sequence = query_full_dataset[query_start_index:query_end_index,:]


    # Select data (dt,x,y,pol)
    query_data, reference_data = subsequence_dtw_functions.select_data(query_batch_sequence, reference_full_dataset, data_selection)

    print(f'Start: {query_start_index} \t End: {query_end_index} \t Length: {query_length} \t Data: {query_data.shape}')

    #---- Perform DTW ----#
    start_time = time.time()
    _, _, _ = subsequence_dtw_functions.subsequence_dtw(query_data, reference_data, print_en=0)
    dtw_time = time.time() - start_time

    pandas_data = [query_data.shape[0], reference_data.shape[0], dtw_time]
    df.loc[0] = pandas_data

    if os.path.exists(df_filename):
        df.to_csv(df_filename, index=False, header=False ,mode='a') # append if file exists
    else:
        df.to_csv(df_filename, index=False, mode='w') # write if the file doesn't exit



    

