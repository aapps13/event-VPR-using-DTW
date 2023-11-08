import sys
function_dir = '/home/aapps/Documents/other/Honours/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import subprocess
import numpy as np
import os
import _utilities.organise_files as organise_files
'''
Valid reference and query names
'sunset1'  'sunset2'  'night'  'daytime'  'morning'  'sunrise'
'''

#----- Parameters -----#
file = 'python ./subsequence_compression_run.py'

reference = 'sunset1'
query = 'sunset2'

script_directory = '/home/aapps/Documents/other/Honours/ENGN4350_Honours/subsequence_dtw/_main_scripts/'

os.chdir(script_directory)

#---- Set up batch names ----#
comp_dir = '/media/aapps/Elements/Data/Output_Data/full_dataset/batch/'

#---- Setup of Storage Arrays ----#
comp_files = os.listdir(comp_dir)
comp_files = [x for x in comp_files if x.startswith('resolution')]
comp_files = sorted(comp_files, key=organise_files.get_resolution_and_threshold) # sort by resolution and threshold


# remove ones that have already been done
files_to_discard = ['resolution_[50-50]_threshold_100','resolution_[100-100]_threshold_100'] 
comp_files = [item for item in comp_files if not item.startswith(files_to_discard[0])]
comp_files = [item for item in comp_files if not item.startswith(files_to_discard[1])]


#----- Run the batch analysis -----#
for batch_file in comp_files:
    command = [sys.executable, "./subsequence_compression_run.py", "-b", f"{batch_file}"]
    subprocess.run(command, check=True)
