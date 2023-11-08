#---- Append function directory to the file path ----#
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cv2
import os
from _functions.visualisation import event_visualisation

## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

# folder = "1619_[1.0-2.0]/"
# data_dir = "C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/output_18-07-23/"

folder = "output_18-07-23/"
data_dir = "C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/"
mat_dir = "C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/"

# read description text file
time = folder[0:folder.find("_")]
desc_file = f"{time}_description.txt"

f = open(os.path.join(data_dir, folder, desc_file), 'r')
desc = f.readlines()


#---- Unpack Parameters ----#
query_file_name_str = desc[1][7:-1]
query_file_name_str_2 = query_file_name_str + '.mat'

reference_file_name_str = desc[2][11:-1]
reference_file_name_str_2 = reference_file_name_str + '.mat'

if query_file_name_str in filename_dict:
    query_file_name = filename_dict[query_file_name_str]
    reference_file_name = filename_dict[reference_file_name_str]

elif query_file_name_str in filename_dict.values():
    query_file_name = query_file_name_str
    reference_file_name = reference_file_name_str

elif query_file_name_str_2 in filename_dict:
    query_file_name = filename_dict[query_file_name_str_2]
    reference_file_name = filename_dict[reference_file_name_str_2]

elif query_file_name_str_2 in filename_dict.values():
    query_file_name = query_file_name_str_2
    reference_file_name = reference_file_name_str_2


# filter gap
filter_gap = int(desc[3][desc[3].find(":")+2:-1])
# query time
q_time = desc[4][desc[4].find(":")+2:-1]
query_start_time = float(q_time[0:q_time.find("-")-1])
query_end_time = float(q_time[q_time.find("-")+2:])
# reference time
r_time = desc[7][desc[7].find(":")+2:-1]
reference_start_time = float(r_time[0:r_time.find("-")-1])
reference_end_time = float(r_time[r_time.find("-")+2:])

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
# Load sunset data
reference_full = loadmat(os.path.join(mat_dir, reference_file_name))['data']
query_full = loadmat(os.path.join(mat_dir, query_file_name))['data']

reference = reference_full
query = query_full

data_dict = {'reference':reference, 
             'query':query}


#---- Filter the data ----#
x_filter = np.arange(0, cols+1, filter_gap)
y_filter = np.arange(0, rows+1, filter_gap)

for key in data_dict:
    data = data_dict[key]
    x_idx = np.isin(data[:,1], x_filter)
    y_idx = np.isin(data[:,2], y_filter)
    idx = np.bitwise_and(x_idx, y_idx)
    data_dict[key] = data[idx,:]

reference = data_dict['reference']
query = data_dict['query']


#---- Slice Reference and Query ----#
# Query 
query_idx = np.where((query[:,0] > query_start_time) & (query[:,0] < query_end_time))
query = query[query_idx[0],:]
query_length = np.shape(query_idx)[1]

# Reference
reference_idx = np.where((reference[:,0] > reference_start_time) & (reference[:,0] < reference_end_time))
reference = reference[reference_idx[0],:]
reference_length = np.shape(reference_idx)[1]

print(f"\n----------- Data Setup -----------")
print(f"Query Time: {query_start_time}s to {query_end_time}s of query")
print(f"Query Length: {query_length}")
print(f"Query Shape: {np.shape(query[:,1:3])} \n")
print(f"Reference Time: {reference_start_time}s to {reference_end_time}s of reference")
print(f"Reference Length: {reference_length}")
print(f"Reference Shape: {np.shape(reference[:,1:3])} \n")


#---- Load Subsequence DTW Results ----#
P_mat_exists = os.path.exists(os.path.join(data_dir, folder, f"P_mat_[{query_start_time}-{query_end_time}].npy"))
C_mat_exists = os.path.exists(os.path.join(data_dir, folder, f"C_mat_[{query_start_time}-{query_end_time}].npy"))
D_mat_exists = os.path.exists(os.path.join(data_dir, folder, f"D_mat_[{query_start_time}-{query_end_time}].npy"))

if P_mat_exists:
    P = np.load(os.path.join(data_dir, folder, f"P_mat_[{query_start_time}-{query_end_time}].npy"))
    a_ast = P[0, 1]
    b_ast = P[-1, 1]
    print(f"\n----------- Subsequence DTW Output -----------")
    print(f'a* = {a_ast}  --->  Time in reference: {reference[a_ast,0]:.3f}')
    print(f'b* = {b_ast}  --->  Time in reference: {reference[b_ast,0]:.3f} \n')
else:
    print('Invalid P_mat file, check that the file exists and the naming is correct')

if C_mat_exists:
    C = np.load(os.path.join(data_dir, folder, f"C_mat_[{query_start_time}-{query_end_time}].npy"))
else:
    print('\t No C_mat file')

if D_mat_exists:
    D = np.load(os.path.join(data_dir, folder, f"D_mat_[{query_start_time}-{query_end_time}].npy"))
    D_scaled = cv2.resize(D, [D.shape[1]//10, D.shape[0]//10])
    D_scaled = D_scaled/D_scaled.max()
    print(f'Accumulated cost D[N, b*] = {D[-1, b_ast]:.2f} \n')
else:
    print('\t No D_mat file')



# #---- Plotting ----#
if (P_mat_exists and C_mat_exists and D_mat_exists):
    fig, ax = plt.subplots(3, 1, figsize=(8,12))
    fig.suptitle("Subsequence DTW Analysis", fontweight='bold', fontsize=16)
    fig.subplots_adjust(top=0.95)
    # Accumulated cost matrix
    im = ax[0].imshow(D_scaled)
    ax[0].set_title("Accumulated Cost Matrix", fontweight='bold')
    ax[0].set_xlabel("Reference (index)")
    ax[0].set_ylabel("Query (index)")
    ax[0].invert_yaxis()
    fig.colorbar(im, ax=ax[0], location='right', shrink=0.75)

    # Optimal Warp Path 
    ax[1].set_title("SubsequenceDTW: Optimal Warping Path", fontweight='bold')
    ax[1].plot(P[:,1], P[:,0], color='r')
    ax[1].set_xlim([0, D.shape[1]])
    ax[1].set_ylim([0, D.shape[0]])
    ax[1].set_xlabel("Reference (index)")
    ax[1].set_ylabel("Query (index)")
    ax[1].grid(True)

    # Combined
    ax[2].imshow(D_scaled)
    ax[2].plot(P[:,1]//10, P[:,0]//10, color='r', alpha=0.75)
    ax[2].axvline(a_ast//10, color='r', ls='--', alpha=0.5, label='a*')
    ax[2].axvline(b_ast//10, color='r', ls='-.', alpha=0.5, label='b*')
    ax[2].invert_yaxis()
    ax[2].set_xlim([0, D.shape[1]//10])
    ax[2].set_ylim([0, D.shape[0]//10])
    ax[2].set_xlabel("Reference (index)")
    ax[2].set_ylabel("Query (index)")
    ax[2].legend()
    plt.show()

elif P_mat_exists:
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Subsequence DTW: Optimal Warping Path", fontweight='bold', fontsize=16)
    fig.subplots_adjust(top=0.9)
    ax.plot(P[:,1], P[:,0], color='r')
    ax.set_xlabel("Reference (index)")
    ax.set_ylabel("Query (index)")
    ax.grid(True)
    ax.set_title(f'Query: {query_file_name_str}     Reference: {reference_file_name_str}')
    plt.show()


#---- Scene Verification ----#
hold_time = 0.1
# create frames
M_query_start = np.zeros((rows, cols))
M_query_end = np.zeros((rows, cols))
M_ref_start = np.zeros((rows, cols))
M_ref_end = np.zeros((rows, cols))

#---- Comparing the start ----#
M_query_start = event_visualisation(query_full, query_start_time)
M_query_end = event_visualisation(query_full, query_end_time)

M_ref_start = event_visualisation(reference_full, reference_start_time)
M_ref_end = event_visualisation(reference_full, reference_end_time)

fig, ax = plt.subplots(2, 2, figsize=(12,9)) 
fig.suptitle("Comparing the Alignment Achieved by Subsequence DTW", fontweight='bold', fontsize=20)
fig.text(0.5, 0.05, f'Query: {query_file_name_str}     Reference: {reference_file_name_str}', ha='center', fontsize=14)
# start alignment
ax[0][0].imshow(M_query_start)
ax[0][0].axis('off')
ax[0][0].set_title('Query Start', fontweight='bold', fontsize=16)
ax[0][1].imshow(M_ref_start)
ax[0][1].axis('off')
ax[0][1].set_title('Reference Start', fontweight='bold', fontsize=16)
# end alignment
ax[1][0].imshow(M_query_end)
ax[1][0].axis('off')
ax[1][0].set_title('Query End', fontweight='bold', fontsize=16)
ax[1][1].imshow(M_ref_end)
ax[1][1].axis('off')
ax[1][1].set_title('Reference End', fontweight='bold', fontsize=16)
plt.show()

