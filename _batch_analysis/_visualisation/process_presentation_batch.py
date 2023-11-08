#---- Append function directory to the file path ----#
linux = 0
if linux:
    function_dir = '/home/aapps/Documents/other/Honours/ENGN4350_Honours/subsequence_dtw/'
    batch_data_dir = ''
else:
    function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
    batch_data_dir = 'F:/Data/Output_Data/31-10-2023/'

import sys
sys.path.append(function_dir)

#---- Import Modules ----#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import _functions.subsequence_dtw_functions as subsequence_dtw_functions
import _functions.determine_ground_truth as determine_ground_truth


batch_data_filename_dict = {'sunset2' : 'sunset1_sunset2_batch_data.csv',
                            'night'   : 'sunset1_night_batch_data.csv',
                            'daytime' : 'sunset1_daytime_batch_data.csv',
                            'morning' : 'sunset1_morning_batch_data.csv'}

reference_name = 'sunset1'
file_to_process = 'sunset2'

df = pd.read_csv(batch_data_dir + batch_data_filename_dict[file_to_process])

df_query_start = df['query_start']
df_query_end = df['query_end']
df_estimated_start = df['estimated_start']
df_estimated_end = df['estimated_end']
df_distance = df['distance']

for i in range(len(df_query_end)):
    # print(f'Query: {df_query_end[i]:.2f} \t Estimated: {df_estimated_end[i]:.2f} \t Distance: {df_distance[i]:.2f}')
    query_time = df_query_end[i]
    estimated_time = df_estimated_end[i]

    reference_path, query_position, estimated_position, distance, closet_reference_time = determine_ground_truth.calc_ground_truth_interp(file_to_process, query_time, reference_name, estimated_time)


    # fig, ax = plt.subplots()
    # ax.plot(reference_path[:,1], reference_path[:,0])
    # ax.scatter(estimated_position[1], estimated_position[0], color='red')
    # ax.scatter(query_position[1], query_position[0], color='blue')
    # ax.scatter(query_position_interp[1], query_position_interp[0], color='green')
    # ax.scatter(estimated_position_interp[1], estimated_position_interp[0], color='orange')

    # plt.show()

    print(f'Query: {query_time:.2f} \t Estimated \t {estimated_time:.2f} \t Original Distance: {df_distance[i]:.2f} \t New Distance: {distance:.2f}')




