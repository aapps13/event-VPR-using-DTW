import sys 
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import os
import argparse
from _functions.determine_ground_truth import calc_ground_truth
from _functions.visualisation import event_visualisation, play_video_frame
from _functions.subsequence_dtw_functions import analyse_cost
from _utilities.organise_files import sort_files

file_dir = 'resolution_[5-5]/'
resolution = file_dir.split('_')[1][:-1]

query_file_name = 'sunset2'
reference_file_name = 'sunset1'

parser = argparse.ArgumentParser(description='Visualise the outputs of the Subsequence DTW localisation algorithm')
parser.add_argument('-f', '--file_directory', type=str, nargs='?', required=False, help='Name of the resolution data directory')
parser.add_argument('-c', '--cost', type=int, nargs='?', required=False, default=1, help='Access to cost data')
parser.add_argument('-p', '--plot', type=int, nargs='?', required=False, default=0, help='Show plot')
parser.add_argument('-s', '--save', type=int, nargs='?', required=False, default=0, help='Save figures')
parser.add_argument('-a', '--alpha', type=float, nargs='?', required=False, default=1, help='Cost analysis alpha')
parser.add_argument('-b', '--beta', type=float, nargs='?', required=False, default=100, help='Cost analysis beta')
args = parser.parse_args()

if args.file_directory:
    file_dir = args.file_directory + '/'
resolution = file_dir.split('_')[1][:-1]

plot_cost = args.cost
show_plot = args.plot
start = 1
save = args.save
alpha = args.alpha
beta = args.alpha

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

data_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/Batch Output/threshold_test/' + file_dir
mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'

sorted_thresh_dirs = sort_files(data_dir, 'threshold')

compiled_distances = []

for thresh_dir in sorted_thresh_dirs:
    data_dir_thresh = data_dir + thresh_dir + '/'

    print(f'\nProcessing {thresh_dir}')

    if not os.path.exists(os.path.join(data_dir_thresh, 'aligned_images')):
        os.mkdir(os.path.join(data_dir_thresh, 'aligned_images'))
        save = 1

    # filename = file_dir.split('_')[0] + '_' + file_dir.split('_')[1] + '.txt'
    # reference_file_name = file_dir.split('_')[0]
    # query_file_name = file_dir.split('_')[1]

    data_text_filename = reference_file_name + '_' + query_file_name + '.txt'
    data_array = np.loadtxt(data_dir_thresh+data_text_filename, delimiter=',')

    reference = loadmat(os.path.join(mat_dir, filename_dict[reference_file_name]))['data']
    query = loadmat(os.path.join(mat_dir, filename_dict[query_file_name]))['data']

    if plot_cost:
        cost_array = np.load(os.path.join(data_dir_thresh, 'accumulated_cost.npy'))

    # #----- Plot Accumulated Cost -----#
    # if data_array.shape[1] > 4: # if the accumulated cost is in the data array

    #     plt.figure(figsize=(12,8))
    #     plt.suptitle("Subsequence DTW: Accumulated Cost for Each Window", fontweight='bold', fontsize=16)
    #     plt.title(f"Query: {query_file_name}    Reference: {reference_file_name}    Query Length:{data_array[0,1]-data_array[0,0]}s", ha='center', fontsize=12, fontweight='normal')

    #     plt.subplots_adjust(hspace=0.98)
    #     plt.plot(data_array[:,4], marker='*', linestyle='--', color=peter_river, markerfacecolor=pomegranate, markeredgecolor=pomegranate, markersize=10)
    #     plt.xlabel('Query Start Time (s)', fontweight='bold')
    #     plt.ylabel('Accumulated Cost (arb.)', fontweight='bold')
    #     plt.xticks(range(data_array.shape[0]), data_array[:,0], rotation=90)
    #     plt.grid(which='both')
        
    #     if save:
    #         plt.savefig(os.path.join(data_dir_thresh, 'aligned_images', 'accumulated_cost'))
    #     elif show_plot:
    #         plt.show()
    #     plt.close()

    # #----- Plot Position Estimate -----#
    # if start:
    #     query_time_array = data_array[:,0]
    #     reference_time_array = data_array[:,2]
    # else:
    #     query_time_array = data_array[:,1]
    #     reference_time_array = data_array[:,3]  

    # # initialise storage arrays
    # estimated_positions = np.zeros((data_array.shape[0], 2))
    # actual_positions = np.zeros((data_array.shape[0], 2))
    # difference_distance = np.zeros((data_array.shape[0], ))

    # for i in range(data_array.shape[0]):
    #     if i == 0:
    #         ground_truth_path, actual_position, estimated_position, distance, _ = calc_ground_truth(query_file_name, query_time_array[i], reference_file_name, reference_time_array[i])  
    #     else:
    #         _, actual_position, estimated_position, distance, _ = calc_ground_truth(query_file_name, query_time_array[i], reference_file_name, reference_time_array[i])
    #     estimated_positions[i,:] = estimated_position
    #     actual_positions[i,:] = actual_position
    #     difference_distance[i] = distance

    # compiled_distances.append(difference_distance)

    # # Set up plotting
    # plt.figure(figsize=(12,10))
    # plt.suptitle("Subsequence DTW: Estimated vs Ground Truth ", fontweight='bold', fontsize=16)
    # plt.plot(ground_truth_path[:,1], ground_truth_path[:,0], color=peter_river, alpha=0.5)
    # plt.scatter(estimated_positions[:,1], estimated_positions[:,0], marker='*', color=pomegranate, label='Esimated Position')
    # plt.scatter(actual_positions[:,1], actual_positions[:,0], marker='^', color=emerald, label='Ground Truth Position')
    # for i in range(estimated_positions.shape[0]):
    #     plt.text(estimated_positions[i,1], estimated_positions[i,0], str(i), color='red', fontsize=8)
    #     plt.text(actual_positions[i,1], actual_positions[i,0], str(i), color='green', fontsize=8)
    # plt.grid(which='both')
    # plt.legend()
    # if save:
    #     plt.savefig(os.path.join(data_dir_thresh, 'aligned_images','position'))
    # elif show_plot:
    #     plt.show()
    # plt.close()

    # # plot distance
    # plt.figure(figsize=(12,10))
    # plt.suptitle("Subsequence DTW: Distance between Estimated and Ground Truth Position", fontweight='bold', fontsize=16)
    # plt.title(f"Query: {query_file_name}    Reference: {reference_file_name}    Query Length:{data_array[0,1]-data_array[0,0]}s", ha='center', fontsize=12, fontweight='normal')
    # plt.plot(difference_distance, color=peter_river)
    # plt.xlabel('Query Start Time (s)', fontweight='bold')
    # plt.ylabel('Distance (m)', fontweight='bold')
    # plt.grid(which='both')
    # if save:
    #     plt.savefig(os.path.join(data_dir_thresh, 'aligned_images','distance'))
    # elif show_plot:
    #     plt.show()
    # plt.close()

    #----- Plot event and video frames
    if start:
        query_time_array = data_array[:,0]
        reference_time_array = data_array[:,2]
    else:
        query_time_array = data_array[:,1]
        reference_time_array = data_array[:,2]

    query_length = data_array[0,1] - data_array[0,0]


    for i in range(len(data_array)):

        print(f"{i} of {len(data_array)-1}")

        query_time = query_time_array[i]
        reference_time = reference_time_array[i]

        # create visualisation array
        M_event_query = event_visualisation(query, query_time, hold_time=0.1, count_threshold=10)
        M_event_reference = event_visualisation(reference, reference_time, hold_time=0.1, count_threshold=10)

        M_video_query = play_video_frame(query_file_name, query_time + video_offset[query_file_name])
        M_video_reference = play_video_frame(reference_file_name, reference_time + video_offset[reference_file_name])

        fig = plt.figure(figsize=(12,8))
        gs = gridspec.GridSpec(4,6)
        fig.suptitle('Localisation via Subsequence DTW using a Mask', fontweight='bold', fontsize=16)
        fig.text(0.5, 0.91, f"Query: {query_file_name}     Query Time: {query_time}s     Query Length: {query_length:.2f}s", ha='center', fontsize=12, fontweight='normal')
        fig.text(0.5, 0.87, f"Reference: {reference_file_name}     Reference Time: {reference_time:.2f}s", ha='center', fontsize=12, fontweight='normal')
        
        gs.update(wspace=1)
        ax1 = plt.subplot(gs[0:2, 0:2]) # query event stream
        ax2 = plt.subplot(gs[2:4, 0:2]) # reference event stream 
        ax3 = plt.subplot(gs[0:2, 2:4]) # mask
        ax4 = plt.subplot(gs[2:4, 2:4]) # mask
        ax5 = plt.subplot(gs[1:3, 4:6]) # mask

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

        ax3.imshow(M_video_query)
        ax3.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)

        ax4.imshow(M_video_reference)
        ax4.tick_params(left= False, bottom= False, labelbottom=False, labelleft=False)

        if plot_cost:
            cost = cost_array[i]
            cost = cost/np.mean(cost)
            min_index, R, nearest_points, _ = analyse_cost(cost, alpha, beta)
            ax5.plot(cost[100:-101], zorder=1)
            ax5.scatter(min_index, cost[min_index], marker='*', color='r', s=50, zorder=3, label='Minimum Cost')
            ax5.axvline(min_index, ls='--', color='k', alpha=0.75, zorder=2)
            ax5.axhline(cost[min_index], ls='--', color='k', alpha=0.75, zorder=2)
            ax5.grid(which='both')
            ax5.set_axisbelow(True)
            ax5.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            # ax5.tick_params(left= True, bottom= True, labelbottom=False, labelleft=False)
            ax5.set_title('Accumulated Cost \n (Normalised by Mean)', fontweight='bold', fontsize=12)

            ax5.set_xlabel('Reference Index')
            ax5.set_ylabel('Accumulated Cost')
            ax5.legend()
            divider = make_axes_locatable(ax5)

        if save == 1:
            plt.savefig(os.path.join(data_dir_thresh, 'aligned_images',f'image_{i+1}'))
            plt.ioff()
        elif show_plot:
            plt.show()
        plt.close()

np.save(data_dir+f'/compiled_distances.npy', compiled_distances)

if os.path.exists(data_dir+f'/compiled_distances.npy'):
    save = 1
    compiled_distances = np.load(data_dir+f'/compiled_distances.npy')
    query_start_array = np.arange(15,35,1) 

    plt.figure(figsize=(12,8))
    for i in range(compiled_distances.shape[0]):
        thresh_string = sorted_thresh_dirs[i].split('_')[1]
        distance_data = compiled_distances[i,:]
        plt.plot(query_start_array, distance_data, label=thresh_string)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'Comparison of Estimated Position Error using Different Compression Thresholds', fontweight='bold', fontsize=14)
    # plt.suptitle(f'Threshold Comparison for Resolution {resolution}', fontweight='bold', fontsize=14)
    plt.title(f'Resolution: {resolution}    Query: {query_file_name}    Reference: {reference_file_name}     Query Length: 1 s', fontsize=12)
    plt.grid(which='both')
    plt.xlabel('Query Start Time (s)')
    plt.ylabel('Distance Error (m)')
    plt.legend(title='Threshold')

    if save == 1:
        plt.savefig(data_dir + '/distance_plot')
        plt.ioff()
    elif show_plot:
        plt.show()
    plt.close()