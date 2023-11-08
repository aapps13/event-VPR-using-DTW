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

script_directory = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/' 
os.chdir(script_directory)

filter = 10
threshold = 5
query_length = 2
save = 1
query_start_array = np.arange(20,25,1) 
query_end_array = query_start_array + query_length
ref_start = 10
ref_end = 30


#----- Run the batch analysis -----#
for query_start, query_end in zip(query_start_array, query_end_array):
    command = f"{file} -q {query_start} {query_end} -r {ref_start} {ref_end} -f {filter} -t {threshold} -s {save} -b 1 -qd {query} -rd {reference}"
    subprocess.run(command, check=True)


#----- Setup saving -----#
save_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/fixed_query/'
new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}')
i = 1
while os.path.exists(new_dir):
    new_dir = os.path.join(save_dir, f'{reference}_{query}_{query_length}({i})')
    i += 1

os.makedirs(new_dir)

#----- Move files -----#
# combine the masks into a single array
files_starting_with_M = [file for file in os.listdir(save_dir) if file.startswith('M')]
start_time = []

# slice of the start time from the mask array
for file in files_starting_with_M:
    index_1 = file.find('[')
    index_2 = file.find('-')
    start_time.append(float(file[index_1+1:index_2]))

# sort the start time and mask filenames in ascending order
sorted_indices = [i for i, v in sorted(enumerate(start_time), key=lambda x: x[1])]
sorted_files = [files_starting_with_M[i] for i in sorted_indices]

# stacked the masks into a single array
M_stacked = np.zeros((np.shape(sorted_files)[0], 260,346))

for i, file in enumerate(sorted_files):
    M = np.load(save_dir+file)
    M_stacked[i] = M
np.save(new_dir+'/M.npy', M_stacked)

# delete individual mask files
for file in sorted_files:  
    os.remove(save_dir+file)

# move the files to the appropriate directories
items = os.listdir(save_dir)
files = [item for item in items if os.path.isfile(os.path.join(save_dir, item))]

# Print the list of files
for file in files:
    shutil.move(os.path.join(save_dir, file), new_dir)
