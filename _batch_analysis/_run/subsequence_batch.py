import subprocess
import numpy as np
import os
import shutil

'''
Valid reference and query names
'sunset1'  'sunset2'  'night'  'daytime'  'morning'  'sunrise'
'''

#----- Parameters -----#
file = 'python .\subsequence_dtw_run.py'
reference = 'sunset1'
query = 'daytime'

script_directory = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/_main_scripts/' 
os.chdir(script_directory)

filter = 10
query_length = 1
save = 1
query_start_array = np.arange(5,25,1) 
query_end_array = query_start_array + query_length
ref_start = 0
ref_end = 50


#----- Run the batch analysis -----#
for query_start, query_end in zip(query_start_array, query_end_array):
    command = f"{file} -q {query_start} {query_end} -r {ref_start} {ref_end} -f {filter} -s {save} -b 1 -qd {query} -rd {reference}"
    subprocess.run(command, check=True)

#----- Setup saving -----#
save_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/'
new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}')
i = 1
while os.path.exists(new_dir):
    new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}({i})')
    i += 1
os.makedirs(new_dir)

#----- Move files -----#
items = os.listdir(save_dir)
files = [item for item in items if os.path.isfile(os.path.join(save_dir, item))]

# Print the list of files
for file in files:
    shutil.move(os.path.join(save_dir, file), new_dir)
