import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import subprocess
import numpy as np
import os
import shutil
from _utilities.organise_files import combine_files


'''
Valid reference and query names
'sunset1'  'sunset2'  'night'  'daytime'  'morning'  'sunrise'
'''

resolution = 'resolution_[1-1]'
master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/compressed_data/'+resolution+'/'
threshold_list = sorted(os.listdir(master_mat_dir), key=lambda s: int(s.split('_')[-1]))

print(f'Thresholds: {threshold_list}')


#----- Parameters -----#
file = 'python .\subsequence_dtw_run.py'
reference = 'sunset1'
query = 'sunset2'

# setup the correct threshold dirs

data_selection = 3
filter = -1
query_length = 3
save = 1
query_start_array = np.arange(20,25,1) 
query_end_array = query_start_array + query_length
ref_start = 15
ref_end = 30


# data_selection = 0
# filter = -1
# query_length =0.1
# save = 1
# query_start_array = np.arange(10,13.1,1) 
# query_end_array = query_start_array + query_length
# ref_start = 9
# ref_end = 12


#---- Change to script dir to run DTW file ----#
script_directory = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/_main_scripts/' 
os.chdir(script_directory)

#----- Setup saving -----#
master_save_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/threshold_test/'
save_dir = os.path.join(master_save_dir, resolution)

i = 1

if os.path.exists(save_dir):
    action = input('Save directory already exists, would you like to create a new one (0), save in existing (1), or exit program (2)? \n')
    match action:
        case '0': # create new dir
            while os.path.exists(save_dir):
                save_dir = os.path.join(master_save_dir, f'{resolution}_({i})')
                i += 1
            os.makedirs(save_dir)
        case '1': 
            save_dir = save_dir
        case '2':
            print('Program aborted')
            exit()
else:
    os.makedirs(save_dir)

#----- Run the batch analysis -----#
for threshold in threshold_list:
    print(f'Threshold: {threshold}')
    mat_dir = master_mat_dir + threshold    
    print(mat_dir)
    sub_save_dir = save_dir + '/' + threshold
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)

    for query_start, query_end in zip(query_start_array, query_end_array):
        command = f'{file} -q {query_start} {query_end} -r {ref_start} {ref_end} -f {filter} -s {save} -b 1 -d {data_selection} -qd {query} -rd {reference} -m "{mat_dir}" -sd "{sub_save_dir}"'
        subprocess.run(command, check=True)
    combine_files(sub_save_dir, 'cost', 'accumulated_cost.npy')



# #----- Move files -----#
# items = os.listdir(save_dir)
# files = [item for item in items if os.path.isfile(os.path.join(save_dir, item))]

# # Print the list of files
# for file in files:
#     shutil.move(os.path.join(save_dir, file), new_dir)
