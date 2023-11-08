
import os
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
import _functions.subsequence_dtw_functions as subsequence_dtw_functions# load the function file
import _functions.determine_ground_truth as determine_ground_truth# load the function file

#---- Parameters ----#
delta = 5 # time between measurements
starting_pos = 22
delt_Q = 2.5
delt_R = 15

#---- File Dictionary ----#
default_reference = 'sunset1'
default_query = 'sunset2'
mat_dir_options = ['C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/',
                   'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/full_datasets/resolution_[2-2]_threshold_750/']

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}


#---- Parser Arguments ----#
parser = argparse.ArgumentParser(description='Run the localisation algorithm using Subsequence DTW')
parser.add_argument('-q', '--query_dataset', type=str, nargs='?', default=default_query, required=False, help='File name of the query event sequence')
parser.add_argument('-r', '--reference_dataset', type=str, nargs='?', default=default_reference, required=False, help='File name of the reference event sequence')
parser.add_argument('-f', '--filter_gap', type=int, nargs='?', default=-1, required=False, help='The spacing used for pixel filtering')
parser.add_argument('-m,', '--mat_dir', type=int, nargs='?', default=0, required=False, help='Option for directory of .mat files')
args = parser.parse_args()

# Check that inputs are valid
assert args.query_dataset in filename_dict.keys(), "The provided query name is invalid, it must be a dictionary key."
assert args.reference_dataset in filename_dict.keys(), "The provided reference name is invalid, it must be a dictionary key."
assert (args.filter_gap>0 or args.filter_gap==-1) and isinstance(args.filter_gap, int), "The filter gap must be a postive integer"
assert args.mat_dir >=0 and args.mat_dir < len(mat_dir_options), "mat_dir option is not available"


#---- Directories ----#
# mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
mat_dir = mat_dir_options[args.mat_dir]
save_dir = f'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/localisation/{args.reference_dataset}_{args.query_dataset}/'

# save_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/localisation/'
output_dir = f'{args.reference_dataset}_{args.query_dataset}/'

i = 1
while os.path.exists(save_dir+output_dir):
    output_dir = f'{args.reference_dataset}_{args.query_dataset}_({i})/'
    i += 1
save_dir = save_dir + output_dir    
os.makedirs(save_dir)


#---- Setup DataFrame ----#
col_names = ['query_start', 'query_end', 'estimated_start', 'estimated_end', 'x_t_hat', 'x_t', 'cov_position','query_postion_x', 'query_position_y', 'estimated_postion_x', 'estimated_position_y', 'distance', 'accumulated_cost', 'query_length']

df = pd.DataFrame(columns=col_names)

#---- Load Data ----#
reference_full = loadmat(f'{mat_dir}{filename_dict[args.reference_dataset]}')['data']
query_full = loadmat(f'{mat_dir}{filename_dict[args.query_dataset]}')['data']
# filter data sets
if args.filter_gap != -1:
    reference_dataset = subsequence_dtw_functions.filter_data(reference_full, args.filter_gap)
    query_dataset = subsequence_dtw_functions.filter_data(query_full, args.filter_gap)
else:
    reference_dataset = reference_full
    query_dataset = query_full
#---- Initialise Variables ----#
# state space
A = np.array([[1,delta], [0, 1]])
C = np.array([[1,0],[0,1]])
x_t = np.zeros((2,1)) # state (x_t|t)
x_t_hat = np.zeros((2,1)) # state (x_t_1|t)
z_t = np.zeros((2,1)) # measurement

# kalman filter variables (arb from now)
Q = np.array([[1e5,0],[0,1e5]])
R = np.array([[1e5,0],[0,1e5]])
K = np.array([[1,0],[0,1]])
P_t = np.array([[1e6,0],[0,1e6]])
# P_t = np.array([[10,0],[0,10]])
P_t_hat = np.array([[0,0],[0,0]])
I = np.eye(2)

#---- Setup Storage Arrays ----#
output_array = []
cov_mat = []
pos_mat = []

#---- Looping for Kalman Filter ----# 
iteration = 1
i = 0

while True: ## determine exit condition
    if iteration == 1:
        iteration += 1
        # select the data signals
        q_start = starting_pos
        q_end = q_start + delt_Q
        query = subsequence_dtw_functions.select_data_sequence(query_dataset, q_start, q_end)
        reference = reference_dataset
        query_length = query.shape[0]

        # measurement
        _, D, P = subsequence_dtw_functions.subsequence_dtw(query, reference, print_en=0)
        a_ast = P[0, 1]
        b_ast = P[-1, 1]
        
        accumulated_cost = D[-1, b_ast]
        normalised_cost = accumulated_cost/query_length
        r_start = reference[a_ast, 0]
        r_end = reference[b_ast, 0]

        # time in full reference
        r_start_idx = np.where(reference_full[:,0]==r_start)[0][0]
        r_end_idx = np.where(reference_full[:,0]==r_end)[0][0]
        r_start_time = reference_full[r_start_idx,0]
        r_end_time = reference_full[r_end_idx,0]

        # calculate time
        dt_R = r_end_time - r_start_time
        num_events_R = b_ast - a_ast
        V = num_events_R/dt_R # reference velocity (events/sec) 
        
        # store position and velocity in the state variable
        x_t[0] = r_end_idx
        x_t[1] = V   
        
        # assess performance and add to storage arrays
        groundtruth_path, query_position, estimated_postion, distance, _ = determine_ground_truth.calc_ground_truth(args.query_dataset, q_end, args.reference_dataset, reference_full[int(x_t[0][0]),0])

    else:
        #---- Prediction Step ----#
        x_t_hat = A*x_t 
        P_t_hat =  A*P_t*A.T + Q

        #---- Measurement (perform SubDTW) ----#
        q_start = q_end + delta # end of previous query + the time step
        q_end = q_start + delt_Q

        # print(f'Query: {q_start:.2f}-{q_end:.2f} \t Reference: {reference_full[r_end_idx,0] - delt_R/2:.2f}-{reference_full[r_end_idx,0] + delt_R/2:.2f}')

        # check the query should still be in the reference
        if (q_end+5) > reference_full[-1,0]:
            print('Query has exceeded the reference map')
            break
        if (q_end+5) > query_full[-1,0]:
            print('Query has exceeded the existing query dataset')
            break

        # select appropriate signals
        query = subsequence_dtw_functions.select_data_sequence(query_dataset, q_start, q_end)
        # reference = subsequence_dtw_functions.select_data_sequence(reference_dataset, reference_full[r_end_idx,0] - delt_R, reference_full[r_end_idx,0] + delt_R)
        reference = reference_dataset
        query_length = query.shape[0]

        # compute SubDTW
        _, D, P = subsequence_dtw_functions.subsequence_dtw(query, reference, print_en=0)
        a_ast = P[0, 1]
        b_ast = P[-1, 1]
        accumulated_cost = D[-1, b_ast]
        normalised_cost = accumulated_cost/query_length

        # find the start and end times of the day 
        r_start = reference[a_ast, 0]
        r_end = reference[b_ast, 0]

        # time in full reference
        r_start_idx = np.where(reference_full[:,0]==r_start)[0][0]
        r_end_idx = np.where(reference_full[:,0]==r_end)[0][0]
        r_start_time = reference_full[r_start_idx,0]
        r_end_time = reference_full[r_end_idx,0]

        # calculate time
        dt_R = r_end_time - r_start_time
        num_events_R = b_ast - a_ast
        V = num_events_R/dt_R # reference velocity (events/sec) 

        z_t[0] = r_end_idx
        z_t[1] = V

        #---- Update Step ----#
        K = P_t_hat * C.T * np.linalg.inv(C*P_t_hat*C.T + R)
        x_t = x_t_hat + K * (z_t - C*x_t)
        P_t = (I - K*C) * P_t_hat * (I - K*C).T + (K*R*K.T)
        
        _, query_position, estimated_postion, distance, _ = determine_ground_truth.calc_ground_truth(args.query_dataset, q_end, args.reference_dataset, reference_full[int(x_t[0][0]),0])

    #---- Add data to storage array ----#
    # ['query_start', 'query_end', 'estimated_start', 'estimated_end', 'x_t_hat', 'x_t', 'cov_position','query_postion_x', 'query_position_y', 
    #  'estimated_postion_x', 'estimated_position_y', 'distance', 'accumulated_cost', 'query_length']"

    data = [q_start, q_end, r_start_time, r_end_time, x_t_hat[0], x_t[0], P_t[0,0], query_position[1], query_position[0], estimated_postion[1], estimated_postion[0], distance, accumulated_cost, query_length]
    # data = [q_start, q_end, reference_full[r_start_idx,0], reference_full[r_end_idx,0], query_position[0], query_position[1], estimated_postion[0], estimated_postion[1], distance, accumulated_cost, query_length]

    df.loc[i] = data
    cov_mat.append(P_t) 
    pos_mat.append(x_t)
    i += 1

    #---- Print diagnostics ----#
    print(f'Query: {q_start:.2f}-{q_end:.2f} \t Reference: {r_start_time:.2f}-{r_end_time:.2f} \t Pos: {reference_full[int(x_t[0][0]),0]:.3f} \t Cov: {P_t[0,0]:.3f} \t Dist: {distance:.2f}\t Normalised Cost: {normalised_cost} ')

# Save the output data
df.to_csv(save_dir+'localisation_data.csv')
# output_array = np.asarray(output_array)
# np.save(save_dir+'localisation_data.npy', output_array)

# Save covariance data
cov_mat = np.asarray(cov_mat)
np.save(save_dir+'covariance_array.npy', cov_mat)

# save ground truth path (map)
np.save(save_dir+'map.npy', groundtruth_path)

# save test description
test_description = f'''----------- Data Setup ----------- 
Query: {args.query_dataset}
Reference: {args.reference_dataset}
Filter Gap: {args.filter_gap}
delta: {delta}
delt_Q: {delt_Q}
'''

file_path = save_dir + 'test_description.txt'
with open(file_path, "w") as file:
    file.write(test_description)
    file.close()