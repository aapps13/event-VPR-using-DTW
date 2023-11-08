import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse
import _functions.subsequence_dtw_functions as subsequence_dtw_functions # load the function file
import _functions.visualisation as visualisation


#---- Setup Data Locations ----#
mat_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_camera_bias/'
filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

if __name__ == '__main__':
    # Create parser arguments
    parser = argparse.ArgumentParser(description='Calculate the camera bias over dataset.')
    parser.add_argument('-f', '--filename', type=str, nargs='?', required=True, help='Name of the dataset')
    args = parser.parse_args()
    
#---- Load the dataset ----#
data = loadmat(mat_directory + filename_dict[args.filename])['data']

#---- Get the framesize values ----#
cols = 346
rows = 260
t_max = np.max(data[:,0])

print(f"Rows: {rows} \t Cols: {cols} \t t_max: {t_max}")

#---- Generate Accumulator Frame ----#
M = visualisation.event_visualisation(data, 0, t_max, cols=cols, rows=rows, count_threshold=-1)

#---- Save M ----#
save_filename = filename_dict[args.filename][:-4]
save_directory = save_directory + save_filename + '_M_accumulation.npy'
# np.save(save_directory, M)


#---- Load M accumulator ----#
M_accumulator = np.load(save_directory)

print(f'Mean: {np.mean(M_accumulator)} \t Max: {np.max(M_accumulator)} \t Min: {np.min(M_accumulator)}')
plt.figure()
im = plt.imshow(M_accumulator)
plt.colorbar(im)
plt.show()