import subprocess
import numpy as np
import os
import shutil

'''
Valid reference and query names
'sunset1'  'sunset2'  'night'  'daytime'  'morning'  'sunrise'
'''

#----- Parameters -----#
file = 'python .\subsequence_dtw_with_mask.py'
reference = 'sunset1'
query = 'night'

script_directory = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/_main_scripts/' 
os.chdir(script_directory)


#---- Setup Batch Values ----#

filter = 10
threshold = 5
query_length = 1
save = 1


#---- Create time arrays ----#
query_start_array = np.arange(15, 30, 1)
query_end_array = query_start_array + query_length
ref_time_array =  np.arange(2, 15, 1)

iteration = 0

for query_start, query_end in zip(query_start_array, query_end_array):
    ref_start_array = query_start - ref_time_array
    ref_end_array = query_end + ref_time_array

    #----- Setup saving -----#
    save_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/fixed_query/master_batch'
    new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}')
    i = 1
    if iteration == 0:
        while os.path.exists(new_dir):
            new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}({i})')
            i += 1
    
    new_dir = os.path.join(new_dir, f'{iteration}_[{query_start}-{query_end}]/')
    iteration += 1
    os.makedirs(new_dir)

    #----- Run the batch analysis -----#
    for ref_start, ref_end in zip(ref_start_array, ref_end_array):
        command = f'{file} -q {query_start} {query_end} -r {ref_start} {ref_end} -f {filter} -t {threshold} -s {save} -sd "{new_dir}" -b 1 -qd {query} -rd {reference}'
        subprocess.run(command, check=True)

    #---- Modify files ----#
    M_name = [file for file in os.listdir(new_dir) if file.startswith('M')][0]
    P_name = [file for file in os.listdir(new_dir) if file.startswith('P')][0]

    os.rename(new_dir+M_name, new_dir+'M.npy')
    os.rename(new_dir+P_name, new_dir+'P.npy')

    #----- Save file setup -----#
    test_setup = np.zeros((len(ref_time_array)+1, 2))
    test_setup[0,:] = [query_start, query_end]
    test_setup[1:,:] = np.transpose([ref_start_array, ref_end_array])
    np.save(new_dir + '/test_setup.npy', test_setup)
