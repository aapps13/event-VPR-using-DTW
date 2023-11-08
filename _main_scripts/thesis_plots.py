#---- Append function directory to the file path ----#
import sys
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)

#---- Import Modules ----#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import _functions.subsequence_dtw_functions as subsequence_dtw_functions
import _functions.visualisation as visualisation


#---- Set up data structures ----#
linux = 0
if linux:
    print('Directories need to be defined..') # update when you know the directory
else:
    master_mat_dir = 'D:/Honours/datasets/compressed_data/batch/'
    master_subdtw_data_dir = 'D:/Honours/datasets/compressed_data/batch/output/'

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}

#---- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'


#---- Plot Params ----#
suptitle_weight = 'bold'
suptitle_size = 20
title_weight = 'bold'
title_size = 16
label_weight = 'normal'
label_size = 14

plot_linewidth = 2
plot_boxwidth = 1.5


#---- show_plot dict ----#
show_plot = {'video_and_event_integration'          : 0,
             'original_vs_compressed_event_rate'    : 0,
             'original_vs_compressed_event_stream'  : 0,
             'compression_techniques'               : 0,
             'subdtw_match_1'                       : 0,
             'subdtw_match_2'                       : 0, 
             'subdtw_match_3'                       : 0,
             'subdtw_match_4'                       : 0,
             'video_frame_comparison'               : 0,
             'run_time'                             : 1} 



###################################################################################################
####                               Video and Event Integration                                 ####
###################################################################################################

if show_plot['video_and_event_integration']:
    mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
    filename = 'sunset1'
    data = loadmat(mat_dir + filename_dict[filename])['data']

    time = 15
    hold_time = 0.1
    event_frame = visualisation.event_visualisation(data, time, hold_time)
    video_frame = visualisation.get_video_frame(filename, time)

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    fig.suptitle('Event Visualisation via Integration', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.5, 0.885, f'Dataset: {filename}       Time: {time} s       Integration Time: {hold_time} s', ha='center', fontweight=label_weight, fontsize=label_size)
    fig.subplots_adjust(top=0.850,bottom=0.01,left=0.03,right=0.93,hspace=0.2,wspace=0.031)
    #---- Video ----#
    ax[0].imshow(video_frame)
    ax[0].axis('off')
    im = ax[1].imshow(event_frame, cmap='bwr', vmax=5, vmin=-5)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Accumulated Polarity')
    ax[1].axis('off')
    plt.show()

###################################################################################################
####                         Original and Compression - Event Stream                           ####
###################################################################################################
hold_time = 0.25
time = 15
resolution = 100
pos_threshold = 20
neg_threshold = -15

if show_plot['original_vs_compressed_event_stream']:
    filename = 'sunset2'
    master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
    master_data_dir = 'G:/Honours/spatial_compression/resolution_[100-100]/compressed/pos_20_neg_-15/'

    #---- Original Data ----# 
    original_data = loadmat(master_mat_dir + filename_dict[filename])['data']
    original_data[:,0] -= original_data[0,0]
    M_original = visualisation.event_visualisation(original_data, time, hold_time, )
    original_threshold = np.max([abs(np.min(M_original)), np.max([M_original])])

    #---- Compressed Data ----#
    compressed_data = loadmat(master_data_dir + filename_dict[filename])['data']
    compressed_data[:,0] -= compressed_data[0,0]
    M_compressed = visualisation.event_visualisation(compressed_data, time, hold_time, rows=resolution, cols=resolution)
    compressed_threshold = np.max([abs(np.min(M_compressed)), np.max([M_compressed])])

    #---- Plotting ----#
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    fig.suptitle('Comparison of Event Stream Before and After Compression', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.50, 0.91, f'Dataset: {filename}    Compressed Resolution: ({resolution},{resolution})    Positive Threshold: {pos_threshold}    Negative Threshold: {neg_threshold}    Integration Time: {hold_time}', ha='center', fontweight=label_weight, fontsize=label_size)

    fig.subplots_adjust(top=0.845, bottom=0.11, left=0.06, right=0.95, hspace=0.2, wspace=0.33)

    ax[0].set_title('Original', fontweight=title_weight, fontsize=title_size)
    im0 = ax[0].imshow(M_original, cmap='bwr', vmin=-original_threshold//2, vmax=original_threshold//2)
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="2%", pad=0.05)
    cbar0 = plt.colorbar(im0, cax=cax0)
    cbar0.set_label('Accumulated Polarity')

    ax[1].set_title('Compressed', fontweight=title_weight, fontsize=title_size)
    im1 = ax[1].imshow(M_compressed, cmap='bwr', vmin=-compressed_threshold, vmax=compressed_threshold)
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Accumulated Polarity')
    plt.show()

###################################################################################################
####                          Original and Compression - Event Rate                            ####
###################################################################################################
if show_plot['original_vs_compressed_event_rate']:
    filename = 'sunset1'
    master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
    master_data_dir = 'G:/Honours/spatial_compression/resolution_[100-100]/compressed/pos_20_neg_-15/'

    resolution = 100
    pos_threshold = 20
    neg_threshold = -15

    #---- Original Data ----# 
    original_data = loadmat(master_mat_dir + filename_dict[filename])['data']
    original_data[:,0] -= original_data[0,0]
    original_data_rounded = np.floor(original_data[:,0])
    original_data_step_index = np.where(np.diff(original_data_rounded)==1)[0]
    original_events_per_second = np.diff(original_data_step_index)

    #---- Compressed Data ----#
    compressed_data = loadmat(master_data_dir + filename_dict[filename])['data']
    compressed_data[:,0] -= compressed_data[0,0]
    compresssed_data_rounded = np.floor(compressed_data[:,0])
    compressed_data_step_index = np.where(np.diff(compresssed_data_rounded)==1)[0]
    compressed_events_per_second = np.diff(compressed_data_step_index)

    #---- Plotting ----#
    xlims = [0, 32]
    fig, ax = plt.subplots(1,3, figsize=(12,5))
    fig.suptitle('Comparison of Event Rate Before and After Compression', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.50, 0.89, f'Dataset: {filename}    Compressed Resolution: ({resolution},{resolution})    Positive Threshold: {pos_threshold}    Negative Threshold: {neg_threshold}', ha='center', fontweight=label_weight, fontsize=label_size)
    fig.subplots_adjust(top=0.8, bottom=0.11, left=0.06, right=0.965, hspace=0.2, wspace=0.33)

    ax[0].set_title('Combined', fontweight=title_weight, fontsize=title_size)
    ax[0].set_xlabel('Time (s)', fontweight=label_weight, fontsize=label_size)
    ax[0].set_ylabel('Event Rate (ev/s)', fontweight=label_weight, fontsize=label_size)
    ax[0].plot(original_events_per_second, color=peter_river, label='Original', linewidth=plot_linewidth)
    ax[0].plot(compressed_events_per_second, color=pomegranate, label='Compressed', linewidth=plot_linewidth)
    [ax[0].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]
    ax[0].set_xlim(xlims)
    ax[0].grid(which='both')
    ax[0].legend()

    ax[1].set_title('Original', fontweight=title_weight, fontsize=title_size)
    ax[1].set_xlabel('Time (s)', fontweight=label_weight, fontsize=label_size)
    ax[1].set_ylabel('Event Rate (ev/s)', fontweight=label_weight, fontsize=label_size)
    ax[1].plot(original_events_per_second, color=peter_river, linewidth=plot_linewidth)
    [ax[1].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]
    ax[1].grid(which='both')
    ax[1].set_xlim(xlims)

    ax[2].set_title('Compressed', fontweight=title_weight, fontsize=title_size)
    ax[2].set_xlabel('Time (s)', fontweight=label_weight, fontsize=label_size)
    ax[2].set_ylabel('Event Rate (ev/s)', fontweight=label_weight, fontsize=label_size)
    ax[2].plot(compressed_events_per_second, color=pomegranate, linewidth=plot_linewidth)
    [ax[2].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]
    ax[2].grid(which='both')
    ax[2].set_xlim(xlims)
    ax[2].ticklabel_format(axis='y', style='scientific')
    plt.show()

###################################################################################################
####                                 Compression Techniques                                    ####
###################################################################################################
if show_plot['compression_techniques']:
    def generate_filter_spacing_points(filter_gap, rows, cols):
        row_init_spacing = (rows%filter_gap)//2
        col_init_spacing = (cols%filter_gap)//2

        row_pixels = np.arange(row_init_spacing, rows+1, filter_gap)
        col_pixels = np.arange(col_init_spacing, cols+1, filter_gap)

        chosen_pixels = [] 
        for i in row_pixels:
            for j in col_pixels:
                chosen_pixels.append([i, j])
        pixel_array = np.asarray(chosen_pixels)
        return pixel_array

    rows = 260
    cols = 346

    pixel_array_10 = generate_filter_spacing_points(10, rows, cols)
    pixel_array_20 = generate_filter_spacing_points(20, rows, cols)
    pixel_array_50 = generate_filter_spacing_points(50, rows, cols)

    #---- Spatial Decimation ----#
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    fig.suptitle("Data Compression via Spatial Decimation", fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.subplots_adjust(top=0.825, bottom=0.155,left=0.05,right=0.96,hspace=0.2,wspace=0.2)
    for i in range(rows):
        ax[0].axhline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[1].axhline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[2].axhline(i, alpha=0.2, color=peter_river, zorder=0)

    for i in range(cols):
        ax[0].axvline(i, alpha=0.2, color=peter_river, zorder=0)    
        ax[1].axvline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[2].axvline(i, alpha=0.2, color=peter_river, zorder=0)

    ax[0].scatter(pixel_array_10[:,1], pixel_array_10[:,0], color=pomegranate, marker='s', s=5, zorder=2)
    ax[0].set_title("Keeping Every 10th Pixel", fontweight=label_weight, fontsize=label_size)
    fig.text(0.175, 0.03, f"Pixels Considered: {pixel_array_10.shape[0]} ({(pixel_array_10.shape[0]/(rows*cols))*100:.2f}%)", ha='center', fontweight=label_weight, fontsize=label_size)
    ax[0].set_xlim([0,cols])
    ax[0].set_ylim([0,rows])
    ax[0].invert_yaxis()
    ax[1].scatter(pixel_array_20[:,1], pixel_array_20[:,0], color=pomegranate, marker='s', s=5, zorder=2)
    ax[1].set_title("Keeping Every 20th Pixel", fontweight=label_weight, fontsize=label_size)
    fig.text(0.5, 0.03, f"Pixels Considered: {pixel_array_20.shape[0]} ({(pixel_array_20.shape[0]/(rows*cols))*100:.2f}%)", ha='center', fontweight=label_weight, fontsize=label_size)
    ax[1].set_xlim([0,cols])
    ax[1].set_ylim([0,rows])
    ax[1].invert_yaxis()
    ax[2].scatter(pixel_array_50[:,1], pixel_array_50[:,0], color=pomegranate, marker='s', s=5, zorder=2)
    ax[2].set_title("Keeping Every 50th Pixel", fontweight=label_weight, fontsize=label_size)
    fig.text(0.825, 0.03, f"Pixels Considered: {pixel_array_50.shape[0]} ({(pixel_array_50.shape[0]/(rows*cols))*100:.2f}%)", ha='center', fontweight=label_weight, fontsize=label_size)
    ax[2].set_xlim([0,cols])
    ax[2].set_ylim([0,rows])
    ax[2].invert_yaxis()
    plt.show()

    #---- Spatio-Temporal Pooling ----#
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    fig.suptitle("Data Compression via Spatio-Temporal Pooling", fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.subplots_adjust(top=0.825, bottom=0.155, left=0.05, right=0.96, hspace=0.2,wspace=0.2)
    for i in range(rows):
        ax[0].axhline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[1].axhline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[2].axhline(i, alpha=0.2, color=peter_river, zorder=0)
    for i in range(cols):
        ax[0].axvline(i, alpha=0.2, color=peter_river, zorder=0)    
        ax[1].axvline(i, alpha=0.2, color=peter_river, zorder=0)
        ax[2].axvline(i, alpha=0.2, color=peter_river, zorder=0)
    regions_1 = 50
    for i in np.linspace(0, rows, regions_1+1):
        ax[0].axhline(i, color=pomegranate)
    for i in np.linspace(0, cols, regions_1+1):
        ax[0].axvline(i, color=pomegranate)
    ax[0].set_title(f"Pooled Resolution: ({regions_1},{regions_1})", fontweight=label_weight, fontsize=label_size)
    ax[0].set_xlim([0,cols])
    ax[0].set_ylim([0,rows])
    ax[0].invert_yaxis()
    regions_2 = 20
    for i in np.linspace(0, rows, regions_2+1):
        ax[1].axhline(i, color=pomegranate)
    for i in np.linspace(0, cols, regions_2+1):
        ax[1].axvline(i, color=pomegranate)
    ax[1].set_title(f"Pooled Resolution: ({regions_2},{regions_2})", fontweight=label_weight, fontsize=label_size)
    ax[1].set_xlim([0,cols])
    ax[1].set_ylim([0,rows])
    ax[1].invert_yaxis()
    regions_3 = 5
    for i in np.linspace(0, rows, regions_3+1):
        ax[2].axhline(i, color=pomegranate)
    for i in np.linspace(0, cols, regions_3+1):
        ax[2].axvline(i, color=pomegranate)
    ax[2].set_title(f"Pooled Resolution: ({regions_3},{regions_3})", fontweight=label_weight, fontsize=label_size)
    ax[2].set_xlim([0,cols])
    ax[2].set_ylim([0,rows])
    ax[2].invert_yaxis()
    plt.show()


###################################################################################################
####                                Subsequence DTW Matching                                   ####
###################################################################################################

####---------------------------------- Matching - Data 1 --------------------------------------####
if show_plot['subdtw_match_1']:
    D = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_1_decimation_20/D_mat.npy')
    P = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_1_decimation_20/P_mat.npy')
    b_ast = P[-1,1]
    ## UPDATE FROM TEXT FILE INFO ##
    query_name = 'morning'
    reference_name = 'sunset1'
    query_end_time = 16
    estimated_time = 16.728965044021606

    dataset_string_1 = f'Query: morning \nTime: 15 - 16s (3952 events) \nReference: sunset1 \nTime: 0 - 30s (42934 events)'
    dataset_string_2 = f'Decimation Spacing: 20 \nLocation: Reference Index {b_ast} \nMatching Distance Error: 6.93m \nDTW Runtime: 241.25s'

    master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'   
    reference_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[reference_name]}')['data']
    query_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[query_name]}')['data']

    hold_time = 0.1
    M_event_query_full = visualisation.event_visualisation(query_dataset_full, query_end_time-hold_time, hold_time)
    M_event_estimate_full = visualisation.event_visualisation(reference_dataset_full, estimated_time-hold_time, hold_time)

    M_video_query = visualisation.get_video_frame(query_name, query_end_time)
    M_video_estimate = visualisation.get_video_frame(reference_name, estimated_time)

    #---- Plotting ----#
    fig = plt.figure(figsize=(10,7))
    fig.suptitle('SubDTW using Spatial Decimation: Taking every 20th Pixel', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.1, 0.22, dataset_string_1, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.text(0.4, 0.22, dataset_string_2, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95,hspace=0.25,wspace=0.55)
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0, 0:4]) # Accumulated Cost Array
    ax2 = plt.subplot(gs[1:3, 0:2]) # query event stream 
    ax3 = plt.subplot(gs[1:3, 2:4]) # query video 

    im1 = ax1.imshow(D, aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Accumulated Cost (arb.)')
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

    ax1.plot(P[:,1], P[:,0], color=pomegranate, linewidth=plot_linewidth)
    ax1.scatter(P[-1,1], 0.98*P[-1,0], marker='*', color=carrot, s=75, zorder=3, label='Estimated Postion')
    ax1.axvline(P[-1,1], linestyle='--', color=carrot,linewidth=plot_linewidth)
    ax1.set_title('SubDTW: Accumulated Cost', fontweight=title_weight, fontsize=title_size)
    ax1.set_xlabel('Reference (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.set_ylabel('Query (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.invert_yaxis()
    ax1.legend()

    ax2.imshow(M_video_query)
    ax2.set_title("Query Position", fontweight=title_weight, fontsize=title_size)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

    ax3.imshow(M_video_estimate)
    ax3.set_title("Estimated Position", fontweight=title_weight, fontsize=title_size)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()

####---------------------------------- Matching - Data 2 --------------------------------------####
if show_plot['subdtw_match_2']:
    D = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_2_decimation_50/D_mat.npy')
    P = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_2_decimation_50/P_mat.npy')
    b_ast = P[-1,1]

    ## UPDATE FROM TEXT FILE INFO ##
    query_name = 'morning'
    reference_name = 'sunset1'
    query_end_time = 16
    estimated_time = 25.816565990447998

    dataset_string_1 = f'Query: morning \nTime: 15 - 16s (547 events) \nReference: sunset1 \nTime: 0 - 30s (7508 events)'
    dataset_string_2 = f'Decimation Spacing: 50 \nLocation: Reference Index {b_ast} \nMatching Distance Error: 141.90m \nDTW Runtime: 4.45s'

    master_mat_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'   
    reference_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[reference_name]}')['data']
    query_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[query_name]}')['data']

    hold_time = 0.1
    M_event_query_full = visualisation.event_visualisation(query_dataset_full, query_end_time-hold_time, hold_time)
    M_event_estimate_full = visualisation.event_visualisation(reference_dataset_full, estimated_time-hold_time, hold_time)

    M_video_query = visualisation.get_video_frame(query_name, query_end_time)
    M_video_estimate = visualisation.get_video_frame(reference_name, estimated_time)
    
    #---- Plotting ----#
    fig = plt.figure(figsize=(10,7))
    fig.suptitle('SubDTW using Spatial Decimation: Taking every 50th Pixel', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.1, 0.22, dataset_string_1, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.text(0.4, 0.22, dataset_string_2, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95,hspace=0.25,wspace=0.55)
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0, 0:4]) # Accumulated Cost Array
    ax2 = plt.subplot(gs[1:3, 0:2]) # query event stream 
    ax3 = plt.subplot(gs[1:3, 2:4]) # query video 

    im1 = ax1.imshow(D, aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Accumulated Cost (arb.)')
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

    ax1.plot(P[:,1], P[:,0], color=pomegranate, linewidth=plot_linewidth)
    ax1.scatter(P[-1,1], 0.98*P[-1,0], marker='*', color=carrot, s=75, zorder=3, label='Estimated Postion')
    ax1.axvline(P[-1,1], linestyle='--', color=carrot, linewidth=plot_linewidth)
    ax1.set_title('SubDTW: Accumulated Cost', fontweight=title_weight, fontsize=title_size)
    ax1.set_xlabel('Reference (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.set_ylabel('Query (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.invert_yaxis()
    ax1.legend()

    ax2.imshow(M_video_query)
    ax2.set_title("Query Position", fontweight=title_weight, fontsize=title_size)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

    ax3.imshow(M_video_estimate)
    ax3.set_title("Estimated Position", fontweight=title_weight, fontsize=title_size)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()

####---------------------------------- Matching - Data 4 --------------------------------------####
if show_plot['subdtw_match_3']:
    D = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_8_pooling_44_32/D_mat.npy')
    P = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_8_pooling_44_32/P_mat.npy')
    b_ast = P[-1,1]

    ## UPDATE FROM TEXT FILE INFO ##
    query_name = 'sunset2'
    reference_name = 'sunset1'
    query_end_time = 38.5
    estimated_time = 39

    dataset_string_1 = f'Query: sunset2 \nTime: 38.5 - 39s (308 events) \n\nReference: sunset1 \nTime: 0 - 120 (208609 events)'
    dataset_string_2 = f'Positive Threshold: 44 \nNegative Threshold: -32 \nLocation: Reference Index {b_ast} \nMatching Distance Error: 5.64m \nDTW Runtime: 65.62s'

    master_mat_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32/'   
    reference_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[reference_name]}')['data']
    query_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[query_name]}')['data']

    hold_time = 0.1
    M_event_query_full = visualisation.event_visualisation(query_dataset_full, query_end_time-hold_time, hold_time)
    M_event_estimate_full = visualisation.event_visualisation(reference_dataset_full, estimated_time-hold_time, hold_time)

    M_video_query = visualisation.get_video_frame(query_name, query_end_time)
    M_video_estimate = visualisation.get_video_frame(reference_name, estimated_time)
    
    #---- Plotting ----#
    fig = plt.figure(figsize=(10,7))
    fig.suptitle('SubDTW using Spatio-Temporal Pooling', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.1, 0.22, dataset_string_1, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.text(0.45, 0.22, dataset_string_2, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95,hspace=0.25,wspace=0.55)
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0, 0:4]) # Accumulated Cost Array
    ax2 = plt.subplot(gs[1:3, 0:2]) # query event stream 
    ax3 = plt.subplot(gs[1:3, 2:4]) # query video 

    im1 = ax1.imshow(D, aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Accumulated Cost (arb.)')
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

    ax1.plot(P[:,1], P[:,0], color=pomegranate, linewidth=plot_linewidth)
    ax1.scatter(P[-1,1], 0.98*P[-1,0], marker='*', color=carrot, s=75, zorder=3, label='Estimated Postion')
    ax1.axvline(P[-1,1], linestyle='--', color=carrot, linewidth=plot_linewidth)
    ax1.set_title('SubDTW: Accumulated Cost', fontweight=title_weight, fontsize=title_size)
    ax1.set_xlabel('Reference (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.set_ylabel('Query (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.invert_yaxis()
    ax1.legend()

    ax2.imshow(M_video_query)
    ax2.set_title("Query Position", fontweight=title_weight, fontsize=title_size)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

    ax3.imshow(M_video_estimate)
    ax3.set_title("Estimated Position", fontweight=title_weight, fontsize=title_size)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()

####---------------------------------- Matching - Data 5 --------------------------------------####
if show_plot['subdtw_match_4']:
    D = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_11_pooling_44_32/D_mat.npy')
    P = np.load('C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_subsequenceDTW/figure_data/data_11_pooling_44_32/P_mat.npy')
    b_ast = P[-1,1]

    ## UPDATE FROM TEXT FILE INFO ##
    query_name = 'sunset2'
    reference_name = 'sunset1'
    query_end_time = 100.25 -2.25
    estimated_time = 101.98443508148193

    dataset_string_1 = f'Query: sunset2 \nTime: 100 - 100.25s (1243 events) \n\nReference: sunset1 \nTime: 0 - 120 (208609 events)'
    dataset_string_2 = f'Positive Threshold: 44 \nNegative Threshold: -32 \nLocation: Reference Index {b_ast} \nMatching Distance Error: 0.23m \nDTW Runtime: 305.23s'

    master_mat_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32/'   
    reference_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[reference_name]}')['data']
    query_dataset_full = loadmat(f'{master_mat_dir}{filename_dict[query_name]}')['data']

    hold_time = 0.1
    M_event_query_full = visualisation.event_visualisation(query_dataset_full, query_end_time-hold_time, hold_time)
    M_event_estimate_full = visualisation.event_visualisation(reference_dataset_full, estimated_time-hold_time, hold_time)

    M_video_query = visualisation.get_video_frame(query_name, query_end_time)
    M_video_estimate = visualisation.get_video_frame(reference_name, estimated_time)
    
    #---- Plotting ----#
    fig = plt.figure(figsize=(10,7))
    fig.suptitle('SubDTW using Spatio-Temporal Pooling', fontweight=suptitle_weight, fontsize=suptitle_size)
    fig.text(0.1, 0.22, dataset_string_1, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.text(0.45, 0.22, dataset_string_2, fontweight=label_weight, fontsize=label_size, linespacing=1.25, va='top')
    fig.subplots_adjust(top=0.9,bottom=0.2,left=0.1,right=0.95,hspace=0.25,wspace=0.55)
    gs = gridspec.GridSpec(3,4)
    ax1 = plt.subplot(gs[0, 0:4]) # Accumulated Cost Array
    ax2 = plt.subplot(gs[1:3, 0:2]) # query event stream 
    ax3 = plt.subplot(gs[1:3, 2:4]) # query video 

    im1 = ax1.imshow(D, aspect='auto')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('Accumulated Cost (arb.)')
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

    ax1.plot(P[:,1], P[:,0], color=pomegranate, linewidth=plot_linewidth)
    ax1.scatter(P[-1,1], 0.98*P[-1,0], marker='*', color=carrot, s=75, zorder=3, label='Estimated Postion')
    ax1.axvline(P[-1,1], linestyle='--', color=carrot, linewidth=plot_linewidth)
    ax1.set_title('SubDTW: Accumulated Cost', fontweight=title_weight, fontsize=title_size)
    ax1.set_xlabel('Reference (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.set_ylabel('Query (Index)', fontweight=label_weight, fontsize=label_size)
    ax1.invert_yaxis()
    ax1.legend()

    ax2.imshow(M_video_query)
    ax2.set_title("Query Position", fontweight=title_weight, fontsize=title_size)
    ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)

    ax3.imshow(M_video_estimate)
    ax3.set_title("Estimated Position", fontweight=title_weight, fontsize=title_size)
    ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.show()
###################################################################################################
####                                Spatio-Temporal Pooling                                    ####
###################################################################################################








###################################################################################################
####                                Video Frame Comparison                                     ####
###################################################################################################
    
if show_plot['video_frame_comparison']:
    sunset1_time = 16.7
    sunset2_time = 15
    night_time = 19
    daytime_time = 16.5
    morning_time = 16.5
    sunrise_time = 14.8

    #---- Get frame from each video

    M_sunset1 = visualisation.get_video_frame('sunset1', sunset1_time)
    M_sunset2 = visualisation.get_video_frame('sunset2', sunset2_time)
    M_night = visualisation.get_video_frame('night', night_time)
    M_daytime = visualisation.get_video_frame('daytime', daytime_time)
    M_morning = visualisation.get_video_frame('morning', morning_time)
    M_sunrise = visualisation.get_video_frame('sunrise', sunrise_time)

    #---- Plotting ----#
    fig, ax = plt.subplots(2,3, figsize=(12,6))
    fig.subplots_adjust(top=0.87,bottom=0.01,left=0.01,right=0.99,hspace=0.178,wspace=0.01)
    # fig.tight_layout(top=0.87,bottom=0.01,left=0.01,right=0.99,hspace=0.178,wspace=0.01)
    fig.suptitle("Visual Comparison of each Data Set", fontweight=suptitle_weight, fontsize=suptitle_size)

    ax[0][0].imshow(M_sunset1)
    ax[0][0].set_title("Sunset1", fontweight=title_weight, fontsize=title_size)
    ax[0][0].axis('off')

    ax[0][1].imshow(M_sunset2)
    ax[0][1].set_title("Sunset2", fontweight=title_weight, fontsize=title_size)
    ax[0][1].axis('off')

    ax[0][2].imshow(M_night)
    ax[0][2].set_title("Night", fontweight=title_weight, fontsize=title_size)
    ax[0][2].axis('off')

    ax[1][0].imshow(M_daytime)
    ax[1][0].set_title("Daytime", fontweight=title_weight, fontsize=title_size)
    ax[1][0].axis('off')

    ax[1][1].imshow(M_morning)
    ax[1][1].set_title("Morning", fontweight=title_weight, fontsize=title_size)
    ax[1][1].axis('off')

    ax[1][2].imshow(M_sunrise)
    ax[1][2].set_title("Sunrise", fontweight=title_weight, fontsize=title_size)
    ax[1][2].axis('off')
    plt.show()


###################################################################################################
####                                       Run Time                                            ####
###################################################################################################

if show_plot['run_time']:

    filenames = ['run_time_all_data',
                 'run_time_large_batch',
                 'run_time_large_batch_2',
                 'run_time_batch_data_combined',
                 'run_time_batch_data_1',
                 'run_time_batch_data_2',
                 'run_time_batch_data_3',]
    
    data_index = 2

    run_time_data_directory = f'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/thesis_results/run_time/{filenames[data_index]}.csv'

    df_run_time = pd.read_csv(run_time_data_directory, delimiter=',')
    N = df_run_time['query_points']
    M = df_run_time['reference_points']
    run_time = df_run_time['runtime']
    NM = N*M

    fig, ax = plt.subplots(figsize=(12,7))
    fig.subplots_adjust(top=0.92,bottom=0.09,left=0.075,right=0.95,hspace=0.2,wspace=0.2)
    fig.suptitle('Algorithm Run Time as a Function of Total Events', fontweight=suptitle_weight, fontsize=suptitle_size)
    ax.scatter(NM, run_time, color=peter_river)
    ax.set_xlabel('Total Number of Events (N x M events)', fontweight=label_weight, fontsize=label_size)
    ax.set_ylabel('DTW Run Time (s)', fontweight=label_weight, fontsize=label_size)
    ax.grid(which='both')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((6, 6))  # Adjust power limits to control when scientific notation is used

    # Apply the custom ticker format to the axis
    ax.xaxis.set_major_formatter(formatter)


    plt.show()

