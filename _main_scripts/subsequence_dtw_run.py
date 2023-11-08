
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import os
import numpy as np
import libfmp.b
import libfmp.c3
from scipy.io import loadmat
import datetime
import argparse
from tabulate import tabulate
import _functions.subsequence_dtw_functions as subsequence_dtw_functions # load the function file
import _functions.determine_ground_truth as determine_groundtruth

default_reference = 'sunset1'
default_query = 'sunset2'
default_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files'

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
parser.add_argument('-f', '--filter_gap', type=int, nargs='?', default=-1, required=False, help='The spacing used for pixel filtering')
parser.add_argument('-s', '--save', type=int, nargs='?', default=0, required=False, help='Determine if the varibles are saved or not')
parser.add_argument('-d', '--data_select', type=int, nargs='?', default=0, required=False, help='Determine which data is used for DTW')
parser.add_argument('-b', '--in_batch', type=int, nargs='?', default=0, required=False, help='Determine if the script is being in a batch analysis')
parser.add_argument('-m', '--mat_dir', type=str, nargs='?', default=default_mat_dir, required=False, help='Directory for the mat file datasets')
parser.add_argument('-qd', '--query_dataset', type=str, nargs='?', default=default_query, required=False, help='File name of the query event sequence')
parser.add_argument('-rd', '--reference_dataset', type=str, nargs='?', default=default_reference, required=False, help='File name of the reference event sequence')
parser.add_argument('-sd', '--save_dir', type=str, nargs='?', required=False, help='Directory to save data')
args = parser.parse_args()

# Check that inputs are valid
assert all(v >= 0 for v in args.query + args.reference), 'All values for q and r must be positive'
assert args.query[0] < args.query[1], 'Invalid Query Values: Query Start < Query End'
assert args.reference[0] < args.reference[1], 'Invalid Reference Values: Reference Start < Reference End'
assert args.save in (0, 1), 'Invalid Save Value: Must be 0 or 1'
assert args.query_dataset in filename_dict.keys(), "The query_dataset must be a dictionary key."
assert args.reference_dataset in filename_dict.keys(), "The reference_dataset must be a dictionary key."
assert args.data_select in (0, 1, 2, 3, 4, 5), 'Invalid Data Select mode: Must be 0, 1, 2, 3, 4, 5'
    # 0 - x, y      1 - x, y, sig       2 - x_norm, y_norm, sig     3 - dt, x, y, sig     4 - dt, x_norm, y_norm, sig
assert os.path.exists(args.mat_dir), 'The provided mat file directory does not exist'
if args.save_dir:
    assert os.path.exists(args.save_dir)

## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##
print('')
today = datetime.date.today()
current_date = datetime.date.today().strftime('%d-%m-%y')
current_time = datetime.datetime.now().strftime('%H%M')

#---- Parameters ----#
filter_gap = args.filter_gap
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
mat_dir = args.mat_dir +'/'

# Load sunset data
reference_dataset = loadmat(f'{mat_dir}{filename_dict[args.reference_dataset]}')['data']
query_dataset = loadmat(f'{mat_dir}{filename_dict[args.query_dataset]}')['data']

# check if the time starts at zero, and correct if necessary (t = t-t_0)
if query_dataset[0,0] != 0:
    query_dataset[:,0] = query_dataset[:,0] - query_dataset[0,0]
if reference_dataset[0,0] != 0:
    reference_dataset[:,0] = reference_dataset[:,0] - reference_dataset[0,0]

# Create directory for this run
if save_data == 1:
    i = 1
    if in_batch == 1:
        save_dir = args.save_dir       
    else:
        save_dir = os.path.join(curr_dir, f'output_{current_date}', f'{current_time}_[{query_start_time}-{query_end_time}]/')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


#---- Filter the data ----#
if filter_gap != -1:
    print('Filtering data')
    reference_dataset = subsequence_dtw_functions.filter_data(reference_dataset, filter_gap)
    query_dataset = subsequence_dtw_functions.filter_data(query_dataset, filter_gap)


#---- Slice Reference and Query ----#
# Query 
query = subsequence_dtw_functions.select_data_sequence(query_dataset, query_start_time, query_end_time)
query_length = np.shape(query)[0]

# Reference
reference = subsequence_dtw_functions.select_data_sequence(reference_dataset, reference_start_time, reference_end_time)
reference_length = np.shape(reference)[0]

#---- Select the data for DTW ----#
# 0 - x, y      1 - x, y, sig       2 - x_norm, y_norm, sig     3 - dt, x, y, sig     4 - dt, x_norm, y_norm, sig
# x, y normalised to [0, 1]


match args.data_select:
    case 0: #(x,y)
        query_data = query[:,1:3]
        reference_data = reference[:,1:3]
    case 1: # (x, y, sig)
        query_data = query[:,1:4]
        reference_data = reference[:,1:4]
    case 2: #(x_norm, y_norm, sig)
        query_xy_norm = query[:,1:3]/np.max(query[:,1:3], axis=0)
        reference_xy_norm = reference[:,1:3]/np.max(reference[:,1:3], axis=0)   
        query_data = np.zeros((query.shape[0], 3))
        query_data[:,0:2] = query_xy_norm
        query_data[:,2] = query[:,3]
        reference_data = np.zeros((reference.shape[0], 3))
        reference_data[:,0:2] = reference_xy_norm
        reference_data[:,2] = reference[:,3]
    case 3: #(dt, x, y, sig)
        query_data = np.zeros((query.shape[0]-1, 4))
        query_data[:,0] = np.diff(query[:,0])
        query_data[:,1:4] = query[1:,1:4]
        reference_data = np.zeros((reference.shape[0]-1, 4))
        reference_data[:,0] = np.diff(reference[:,0])
        reference_data[:,1:4] = reference[1:,1:4]
    case 4: #(dt, x_norm, y_norm, sig)
        query_xy_norm = query[:,1:3]/np.max(query[:,1:3], axis=0)
        reference_xy_norm = reference[:,1:3]/np.max(reference[:,1:3], axis=0)
        query_data = np.zeros((query.shape[0]-1, 4))
        query_data[:,0] = np.diff(query[:,0])
        query_data[:,1:3] = query_xy_norm[1:,:]
        query_data[:,3] = query[1:,3]
        reference_data = np.zeros((reference.shape[0]-1, 4))
        reference_data[:,0] = np.diff(reference[:,0])
        reference_data[:,1:3] = reference_xy_norm[1:,:]
        reference_data[:,3] = reference[1:,3]
    case 5: #dt, sig
        query_data = np.zeros((query.shape[0]-1, 2))
        reference_data = np.zeros((reference.shape[0]-1, 2))
        query_data[:,0] = np.diff(query[:,0])
        query_data[:,1] = query[1:, 3]
        reference_data[:,0] = np.diff(reference[:,0])
        reference_data[:,1] = reference[1:, 3]
    case other:
        # assertion at the beginning should avoid any case of this
        ValueError('Unknown Data Select Input')


setup_string = f'''----------- Data Setup -----------
Query: {args.query_dataset}
Reference: {args.reference_dataset}
Filter gap: {filter_gap}
Query Time (s): {query_start_time} - {query_end_time}
Query Length: {query_length} 
Query Shape: {query_data.shape} 
Reference Time (s): {reference_start_time} - {reference_end_time}
Reference Length: {reference_length} 
Reference Shape: {reference_data.shape}
Data Select: {args.data_select}
'''
print(setup_string)



#---- Compute Subsequence DTW ----#
# Compute cost matrix using Euclidean distance
print('Computing the cost matrix C:')
# C =  libfmp.c3.compute_cost_matrix(query[:,1:3].T, reference[:,1:3].T, metric='euclidean')
C =  libfmp.c3.compute_cost_matrix(query_data.T, reference_data.T, metric='euclidean')
print('Cost matrix complete \n')

# Compute the accumulated cost
print('Computing the accumulated cost matrix D:')
D =  subsequence_dtw_functions.compute_accumulated_cost_matrix_subsequence_dtw(C)
print('Accumulated cost matrix complete \n')

# Compute optimal warping path
print('Computing the optimal warping path:')
P = subsequence_dtw_functions.compute_optimal_warping_path_subsequence_dtw(D)
print('Optimal path computed \n')

# Analyse the outputs
a_ast = P[0, 1]
b_ast = P[-1, 1]

#---- Get performance metrics ----#
reference_path, query_position, estimated_postion, distance, closet_reference_time = determine_groundtruth.calc_ground_truth(args.query_dataset, query_end_time, args.reference_dataset, reference_end_time)

subsequence_string = f'''----------- Subsequence Output -----------
Cost matrix C = {C.shape}
Accumulated cost matrix D = {D.shape}
Optimal warp matrix P = {P.shape}
a* = {a_ast}
\t --> Time in Reference = {reference[a_ast, 0]:.2f}
b* = {b_ast}
\t --> Time in Reference = {reference[b_ast, 0]:.2f} \n
Match Distance Error: {distance}
'''
print(subsequence_string)

results =  [['', 'Query', 'Reference', 'Matching Error (m)'],
            ['Start Time (s)', query_start_time, f'{reference[a_ast, 0]:.2f}',''],
            ['End Time (s)', query_end_time, f'{reference[b_ast, 0]:.2f}', distance]]

results_table = tabulate(results, headers='firstrow', tablefmt='grid')
print(results_table)

#---- 

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
        f.write(f"{query_start_time}, {query_end_time}, {reference[a_ast, 0]:.4f}, {reference[b_ast, 0]:.4f}, {D[-1, b_ast]} \n")
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
    np.save(save_dir+f'/cost_[{query_start_time}-{query_end_time}]', D[-1,:])
    np.save(save_dir+f'/P_mat_[{query_start_time}-{query_end_time}]', P)

del C, D, P

