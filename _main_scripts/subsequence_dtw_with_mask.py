import os
import sys 
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import datetime
import argparse
from tabulate import tabulate
import _functions.subsequence_dtw_functions as subsequence_dtw_functions # load the function file
from _functions.subsequence_dtw_functions import filter_data
from _functions.visualisation import event_visualisation

default_reference = 'sunset1'
default_query = 'sunset2'

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}
 
# Create parser arguments
parser = argparse.ArgumentParser(description='Perform subsequence DTW on the event stream data.')
parser.add_argument('-q', '--query', type=float, nargs=2, required=True, help='Query start and end values (seconds)')
parser.add_argument('-r', '--reference', type=float, nargs=2, required=True, help='Reference start and end values (seconds)')
parser.add_argument('-f', '--filter_gap', type=int, nargs='?', default=10, required=False, help='The spacing used for pixel filtering')
parser.add_argument('-t', '--mask_threshold', type=int, nargs='?', default=10, required=False, help='The threshold used for producing the pixel mask')
parser.add_argument('-s', '--save', type=int, nargs='?', default=0, required=False, help='Determine if the varibles are saved or not')
parser.add_argument('-b', '--in_batch', type=int, nargs='?', default=0, required=False, help='Determine if the script is being in a batch analysis')
parser.add_argument('-qd', '--query_dataset', type=str, nargs='?', default=default_query, required=False, help='File name of the query event sequence')
parser.add_argument('-rd', '--reference_dataset', type=str, nargs='?', default=default_reference, required=False, help='File name of the reference event sequence')
parser.add_argument('-sd', '--save_directory', type=str, nargs='?', required=False, help='Location to save the output')

args = parser.parse_args()

# Check that inputs are valid
assert all(v >= 0 for v in args.query + args.reference), 'All values for q and r must be positive'
assert args.query[0] < args.query[1], 'Invalid Query Values: Query Start < Query End'
assert args.reference[0] < args.reference[1], 'Invalid Reference Values: Reference Start < Reference End'
assert args.save in (0, 1), 'Invalid Save Value: Must be 0 or 1'
assert args.query_dataset in filename_dict.keys(), "The query_dataset must be a dictionary key."
assert args.reference_dataset in filename_dict.keys(), "The reference_dataset must be a dictionary key."

## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##
print('')
today = datetime.date.today()
current_date = datetime.date.today().strftime('%d-%m-%y')
current_time = datetime.datetime.now().strftime('%H%M')

#---- Parameters ----#
filter_gap = args.filter_gap
threshold = args.mask_threshold

save_data = args.save
in_batch = args.in_batch

# Slice the query 
query_start_time = args.query[0]
query_end_time = args.query[1]
# Take a subset of the reference
reference_start_time = args.reference[0]
reference_end_time = args.reference[1]
# camera parameters
rows = 260
cols = 346

#--- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'

#---- Load Data ----#
curr_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/'
mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'

# Load sunset data
reference_dataset = loadmat(f'{mat_dir}{filename_dict[args.reference_dataset]}')['data']
query_dataset = loadmat(f'{mat_dir}{filename_dict[args.query_dataset]}')['data']

#---- Determine the saving location ----#
if args.save_directory:
    save_dir = args.save_directory
else:
    save_dir = os.path.join(curr_dir, f'output_{current_date}', f'{current_time}_[{query_start_time}-{query_end_time}]/')

# Create directory for this run
if save_data == 1:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


#---- Slice Reference and Query ----#
# Query 
query_idx = np.where((query_dataset[:,0] > query_start_time) & (query_dataset[:,0] < query_end_time))
query = query_dataset[query_idx[0],:]

# Reference
reference_idx = np.where((reference_dataset[:,0] > reference_start_time) & (reference_dataset[:,0] < reference_end_time))
reference = reference_dataset[reference_idx[0],:]

#---- Apply Mask ----#
# reference, query, M = subsequence_dtw_functions.key_event_mask(query, reference, threshold, filter_gap)
query, M = subsequence_dtw_functions.remove_active_pixels(query, threshold=5)

#---- Filter reference ----#
reference = filter_data(reference, filter_gap)
query = filter_data(query, filter_gap)

reference_length = np.shape(reference)[0]
query_length = np.shape(query)[0]

setup_string = f'''----------- Data Setup -----------
Query: {args.query_dataset}
Reference: {args.reference_dataset}
Query Time (s): {query_start_time} - {query_end_time}
Query Length: {query_length} 
Query Shape: {np.shape(query[:,1:3])} 
Reference Time (s): {reference_start_time} - {reference_end_time}
Reference Length: {reference_length} 
Reference Shape: {np.shape(reference[:,1:3])}

Filter Gap: {filter_gap}
Pixel Threshold: {threshold}
'''
print(setup_string)

#---- Compute Subsequence DTW ----#
C, D, P = subsequence_dtw_functions.subsequence_dtw(query, reference)

# Analyse the outputs
a_ast = P[0, 1]
b_ast = P[-1, 1]

subsequence_string = f'''----------- Subsequence Output -----------
Cost matrix C = {C.shape}
Accumulated cost matrix D = {D.shape}
Optimal warp matrix P = {P.shape}
a* = {a_ast}
\t --> Time in Reference = {reference[a_ast, 0]:.2f}
b* = {b_ast}
\t --> Time in Reference = {reference[b_ast, 0]:.2f} \n
'''
print(subsequence_string)

results =  [['', 'Query', 'Reference'],
            ['Start', query_start_time, f'{reference[a_ast, 0]:.2f}'],
            ['End', query_end_time, f'{reference[b_ast, 0]:.2f}']]

results_table = tabulate(results, headers='firstrow', tablefmt='grid')
print(results_table)

#---- Save Data ----#
if save_data == 1:
    if in_batch == 1:
        f = open(f'{save_dir}/batch_description.txt', 'a')
        f.write(setup_string)
        f.write(subsequence_string)
        f.write(results_table)
        f.write('\n\n############################################ \n')
        f.close()

        f = open(f'{save_dir}/{args.reference_dataset}_{args.query_dataset}.txt', 'a')
        f.write(f"{query_start_time}, {query_end_time}, {reference[a_ast, 0]:.4f}, {reference[b_ast, 0]:.4f}, {D[-1, b_ast]}, {query_length} \n")
        f.close()

    else:
    #Save the text file
        f = open(f'{save_dir}/{current_time}_description.txt', 'w')
        f.write(setup_string)
        f.write(subsequence_string)
        f.write(results_table)
        f.close()

    # Save DTW cost matrices
    # np.save(save_dir+'C_mat', C)
    # np.save(save_dir+'D_mat', D)
    np.save(save_dir+f'P_mat_[{query_start_time}-{query_end_time}]', P)
    np.save(save_dir+f'M_[{query_start_time}-{query_end_time}]', M)

del C, D, P, M

