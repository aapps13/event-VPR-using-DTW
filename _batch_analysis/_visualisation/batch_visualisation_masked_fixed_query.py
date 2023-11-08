import sys 
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import os
from _functions.determine_ground_truth import calc_ground_truth
from _functions.visualisation import event_visualisation, play_video_frame
from mpl_toolkits.axes_grid1 import make_axes_locatable


file_dir = 'sunset1_night_1/14_[29-30]/'

single_time = 1
show_plot = 1
start = 1
save = 0

#--- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'


## https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html ##

filename_dict = {'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                 'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                 'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                 'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                 'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                 'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

video_offset = {'sunset1' : 5,
                'sunset2' : -5.5,
                'daytime' : 0,
                'sunrise' : 4.5,
                'morning' : 0,
                'night'    : 2}

data_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/fixed_query/master_batch/' + file_dir
mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'

if not os.path.exists(os.path.join(data_dir, 'aligned_images')):
    os.mkdir(os.path.join(data_dir, 'aligned_images'))
    save = 1

filename = file_dir.split('_')[0] + '_' + file_dir.split('_')[1] + '.txt'
reference_file_name = file_dir.split('_')[0]
query_file_name = file_dir.split('_')[1]

data_array = np.loadtxt(data_dir+filename, delimiter=',')

reference = loadmat(os.path.join(mat_dir, filename_dict[reference_file_name]))['data']
query = loadmat(os.path.join(mat_dir, filename_dict[query_file_name]))['data']
mask = np.load(os.path.join(data_dir, 'M.npy'))
test_times = np.load(data_dir + '/test_setup.npy')

q_time = test_times[0,:]
ref_start = test_times[1:,0]
ref_end = test_times[1:,1]


#----- Plot Accumulated Cost -----#
if data_array.shape[1] > 4: # if the accumulated cost is in the data array

    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle("Subsequence DTW: Accumulated Cost", fontweight='bold', fontsize=16)
    fig.text(0.5, 0.91, f"Query: {query_file_name}    Reference: {reference_file_name}    Query Length:{data_array[0,1]-data_array[0,0]}s", ha='center', fontsize=12, fontweight='normal')   
    fig.subplots_adjust(top=0.82)
    ax[0].plot(data_array[:,4], marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
    ax[0].set_title('Original', fontweight='bold')
    ax[0].set_xlabel('Query Start Time (s)', fontweight='bold')
    ax[0].set_ylabel('Accumulated Cost (arb.)', fontweight='bold')
    ax[0].set_xticks(range(data_array.shape[0]), data_array[:,0], rotation=90)
    ax[0].grid(which='both')
    ax[1].plot(data_array[:,4]/data_array[:,5], marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
    ax[1].set_title('Normalised by Query Length', fontweight='bold')
    ax[1].set_xlabel('Query Start Time (s)', fontweight='bold')
    ax[1].set_ylabel('Accumulated Cost (arb.)', fontweight='bold')
    ax[1].set_xticks(range(data_array.shape[0]), data_array[:,0], rotation=90)
    ax[1].grid(which='both')
    if save:
        plt.savefig(os.path.join(data_dir, 'aligned_images','_accumulated_cost'))
    elif show_plot:
        plt.show()
    else:
        plt.ioff()
    plt.close()


#----- Plot Position Estimate -----#
if start:
    query_time_array = data_array[:,0]
    reference_time_array = data_array[:,2]
else:
    query_time_array = data_array[:,1]
    reference_time_array = data_array[:,2]  

# initialise storage arrays
estimated_positions = np.zeros((data_array.shape[0], 2))
actual_positions = np.zeros((data_array.shape[0], 2))
difference_distance = np.zeros((data_array.shape[0], 1))

for i in range(data_array.shape[0]):
    if i == 0:
        ground_truth_path, actual_position, estimated_position, distance, closest_time = calc_ground_truth(query_file_name, query_time_array[i], reference_file_name, reference_time_array[i])
    else:
        _, actual_position, estimated_position, distance, closest_time = calc_ground_truth(query_file_name, query_time_array[i], reference_file_name, reference_time_array[i])
    estimated_positions[i,:] = estimated_position
    actual_positions[i,:] = actual_position
    difference_distance[i] = distance

# Plot
fig, ax = plt.subplots(1,2, figsize=(12,6))
fig.suptitle("Subsequence DTW: Estimated vs Ground Truth Position", fontweight='bold', fontsize=16)
fig.text(0.5, 0.91, f"Query: {query_file_name}    Reference: {reference_file_name}    Query Length:{data_array[0,1]-data_array[0,0]}s", ha='center', fontsize=12, fontweight='normal')   
fig.subplots_adjust(top=0.82)
ax[0].plot(ground_truth_path[:,1], ground_truth_path[:,0], color=peter_river, alpha=0.5)
ax[0].scatter(estimated_positions[:,1], estimated_positions[:,0], marker='*', color=pomegranate, label='Esimated Position')
ax[0].scatter(actual_positions[:,1], actual_positions[:,0], marker='^', color=emerald, label='Ground Truth Position')
for i in range(estimated_positions.shape[0]):
    ax[0].text(estimated_positions[i,1], estimated_positions[i,0], str(i), color='red', fontsize=8)
    ax[0].text(actual_positions[i,1], actual_positions[i,0], str(i), color='green', fontsize=8)
ax[0].set_title('Location in Traverse', fontweight='bold')
ax[0].set_xlabel('Latitude (deg)', fontweight='bold')
ax[0].set_ylabel('Longitude (deg)', fontweight='bold')
ax[0].grid(which='both')
ax[0].ticklabel_format(style='plain') 
ax[0].legend()
ax[1].plot(difference_distance, marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
ax[1].set_title('Distance between Estimate and Ground Truth Position', fontweight='bold')
ax[1].set_xlabel('Query Start Time (s)', fontweight='bold')
ax[1].set_ylabel('Distance (m)', fontweight='bold')
ax[1].set_xticks(range(data_array.shape[0]), data_array[:,0], rotation=90)
ax[1].grid(which='both')
if save:
    plt.savefig(os.path.join(data_dir, 'aligned_images','_position'))
elif show_plot:
    plt.show()
else:
    plt.ioff()
plt.close()


#----- Plot the expanding window -----#
plt.figure(figsize=(12,8))
plt.subplots_adjust(top=0.9)
for i in range(len(test_times)-1):
    plt.scatter(data_array[i,2], i, marker='*', color='r', s=150, zorder=3)#, label='Estimated Time')
    plt.scatter(data_array[i,3], i, marker='*', color='b', s=150, zorder=3)#, label='Estimated Time')
    plt.plot([ref_start[i], ref_end[i]], [i, i], color='k', linestyle='--', marker='+', markersize=10, zorder=2)
plt.suptitle('Subsequence DTW: Fixed Query with an Expanding Reference Window', fontweight='bold', fontsize=16)
plt.title(f"Query: {query_file_name}      Reference: {reference_file_name}      Query Time:{q_time[0]}-{q_time[1]}s", ha='center', fontsize=12, fontweight='normal')   
plt.xlabel('Window Time (s)', fontweight='bold', fontsize=14)
plt.ylabel('Window Number', fontweight='bold', fontsize=14)
plt.legend(['Estimated Start', 'Estimated Finish'])
plt.grid(which='both', zorder=1, alpha=0.4)
if save:
    plt.savefig(os.path.join(data_dir, 'aligned_images','_expanding_window'))
elif show_plot:
    plt.show()
else:
    plt.ioff()
plt.close()


#----- Plot event and video frames
if start:
    query_time_array = data_array[:,0]
    reference_time_array = data_array[:,2]
else:
    query_time_array = data_array[:,1]
    reference_time_array = data_array[:,2]

query_length = data_array[0,1] - data_array[0,0]

for i in range(len(data_array)):
    print(f"{i+1} of {len(data_array)}")

    query_time = query_time_array[i]
    reference_time = reference_time_array[i]

    # create visualisation array
    M_event_query = event_visualisation(query, query_time, hold_time=0.1, count_threshold=10)
    M_event_reference = event_visualisation(reference, reference_time, hold_time=0.1, count_threshold=10)

    M_video_query = play_video_frame(query_file_name, query_time + video_offset[query_file_name])
    M_video_reference = play_video_frame(reference_file_name, reference_time + video_offset[reference_file_name])


    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(4,6)
    # gs.tight_layout(figure=fig)
    fig.suptitle('Localisation via Subsequence DTW using a Mask', fontweight='bold', fontsize=16)
    fig.text(0.5, 0.91, f"Query: {query_file_name}     Query Time: {query_time}s     Query Length: {query_length:.2f}s", ha='center', fontsize=12, fontweight='normal')
    fig.text(0.5, 0.87, f"Reference: {reference_file_name}     Reference Time: {reference_time:.2f}s", ha='center', fontsize=12, fontweight='normal')

    gs.update(wspace=0.4)
    ax1 = plt.subplot(gs[0:2, 0:2]) # query event stream
    ax2 = plt.subplot(gs[2:4, 0:2]) # reference event stream 
    ax3 = plt.subplot(gs[1:3, 2:4]) # mask
    ax4 = plt.subplot(gs[0:2, 4:6]) # mask
    ax5 = plt.subplot(gs[2:4, 4:6]) # mask

    im1 = ax1.imshow(M_event_query)
    ax1.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)
    ax1.set_ylabel('Query', fontweight='bold', fontsize=14, labelpad=10)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0)
    cbar = plt.colorbar(im1, cax=cax)

    im2 = ax2.imshow(M_event_reference)
    ax2.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)
    ax2.set_ylabel('Reference', fontweight='bold', fontsize=14, labelpad=10)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0)
    cbar = plt.colorbar(im2, cax=cax)

    im3 = ax3.imshow(mask)#, vmin=0, vmax=10)
    ax3.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)
    ax3.set_title('Mask', fontweight='bold', fontsize=12)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="2%", pad=0)
    cbar = plt.colorbar(im3, cax=cax)

    ax4.imshow(M_video_query)
    ax4.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)

    ax5.imshow(M_video_reference)
    ax5.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)

    if save == 1:
        plt.savefig(os.path.join(data_dir, 'aligned_images',f'image_{i+1}'))
        plt.ioff()
    elif show_plot:
        plt.show()
    plt.close()
