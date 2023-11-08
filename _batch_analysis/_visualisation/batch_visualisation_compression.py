import sys 
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#---- Colours ----#
green_sea = '#16a085'
emerald = '#2ecc71'
peter_river = '#3498db'
amythest = '#9b59b6'
carrot = '#e67e22'
pomegranate = '#c0392b'
wisteria = '#8e44ad'
colour_list = [emerald, peter_river, amythest, carrot, pomegranate, wisteria,green_sea]

linux = 0
if linux == 1:
    exit() # UPDATE to /media/aapps... when you know what it is
else:
    function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
    sys.path.append(function_dir)
    # master_data_dir = 'D:/Honours/datasets/compressed_data/batch/output/'
    master_data_dir = 'F:/Honours/output_ds1/'
    

#---- Import Function Modules ----#
import _utilities.organise_files as organise_files

#---- Sort the files in by resolution then threshold ----#
files = os.listdir(master_data_dir)
files = sorted(files, key=organise_files.get_resolution_and_threshold) # sort by resolution and threshold


#---- Read and concatenate data csv's into a single Pandas Dataframe ----#
headers = ['Resolution', 'Threshold', 'Reference', 'Query', 'Data Selection', 'Query Start (s)', 'Query End (s)', 'Query Points', 'Reference Points' ,'Estimated Start (s)', 'Estimated End (s)', 'Distance (m)','Runtime (s)','Memory (B)', 'DTW Successful']
df_global = pd.DataFrame(columns=headers)
resolution_list = []
threshold_list = []
query_start_list = []
for file in files:
    data_dir = master_data_dir + file
    df_temp = pd.read_csv(data_dir + '/batch_data.csv')
    df_global = pd.concat([df_global, df_temp], ignore_index=True)

query_name = df_global['Query'][0]
reference_name = df_global['Reference'][0]
query_length = float(df_global['Query End (s)'][0]) - float(df_global['Query Start (s)'][0])


# Find the resolutions and thresholds of the test
for val in df_global['Resolution']:
    if val not in resolution_list:
        resolution_list.append(val)
for val in df_global['Threshold']:
    if val not in threshold_list:
        threshold_list.append(val)
for val in df_global['Query Start (s)']:
    if val not in query_start_list:
        query_start_list.append(val)

#---- Generate arrays for plotting ----#
runtime_2d_res_thresh = np.zeros([len(resolution_list), len(threshold_list)]) 
distance_2d_res_thresh = np.zeros([len(resolution_list), len(threshold_list)]) 

runtime_selection = 'mean'
distance_selection = 'mean'

runtime_list = []
distance_list = [] 

for res_index, resolution in enumerate(resolution_list):
    for thresh_index, threshold in enumerate(threshold_list):
        # runtime
        runtime_vals = (df_global[(df_global['Resolution'] == resolution) & (df_global['Threshold'] == threshold)]['Runtime (s)'])
        runtime_list.append([resolution]+[threshold] + runtime_vals.tolist())
        if runtime_selection == 'mean':
            runtime_2d_res_thresh[res_index, thresh_index] = np.mean(runtime_vals)
        elif runtime_selection == 'median':
            runtime_2d_res_thresh[res_index, thresh_index] = np.median(runtime_vals)
        else:
            ValueError("Runtime selection parameter is invalid")
        # distance
        distance_vals = (df_global[(df_global['Resolution'] == resolution) & (df_global['Threshold'] == threshold)]['Distance (m)'])
        distance_list.append([resolution]+[threshold] + distance_vals.tolist())
        if distance_selection == 'mean':
            distance_2d_res_thresh[res_index, thresh_index] = np.mean(distance_vals)
        elif distance_selection == 'median':
            distance_2d_res_thresh[res_index, thresh_index] = np.median(distance_vals)
        else:
            ValueError("Distance selection parameter is invalid")

runtime_array = np.asarray(runtime_list)
distance_array = np.asarray(distance_list)


########################################################################################################################################
####                                                       Plotting                                                                #####
########################################################################################################################################

#---- Global Parameters ----#

show_plot = {"runtime_resolution"   : 0,
             "distance_resolution"  : 0,
             "query_length_runtime" : 0,
             "res_thresh_runtime"   : 0,
             "res_thresh_distance"  : 1}

suptitle_size = 16
suptitle_weight = 'bold'
title_size = 12
title_weight = 'bold'
label_size = 12
label_weight = 'normal'
plot_linewidth = 2
plot_boxwidth = 1.5

########################################################################################################################################
####                                              Runtime for each resolution                                                      #####
ylims=[-2,75]

fig, ax = plt.subplots(1,2, figsize=(12,6))
fig.subplots_adjust(top=0.84, bottom=0.15, left=0.07, right=0.9, hspace=0.35, wspace=0.155)
fig.suptitle("SubDTW Compression: Runtime Comparison for Different Compression Configurations", fontweight=suptitle_weight, fontsize=suptitle_size)
fig.text(0.5, 0.9, f'Reference: {reference_name}        Query: {query_name}        Query Length: {query_length} s', ha='center', fontsize=title_size)
for i in range(len(resolution_list)):
 
    a = i*len(threshold_list)
    b = a + len(threshold_list)
    resolution = runtime_array[a,0]
    threshold = runtime_array[a:b,1]
    runtime_data = runtime_array[a:b,2:]

    for j in range(len(threshold)):
        ax[i].plot(runtime_data[j,:], label=f'{threshold[j]:.0f}', color=colour_list[j], linewidth=plot_linewidth)

    if i == 1: # ony have one legend on the top left corner
        legend = ax[i].legend(bbox_to_anchor=(1.25, 1.025))
        legend.set_title('Threshold', {'size': 'medium', 'weight': 'normal', 'style': 'normal'})

    ax[i].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
    ax[i].set_ylabel('Runtime (s)', fontweight=label_weight, fontsize=label_size)
    ax[i].set_xlabel('Query Start Time (s)', fontweight=label_weight, fontsize=label_size)
    ax[i].grid(which='both')
    ax[i].set_ylim(ylims)
    x_ticks = np.arange(0, len(query_start_list), 5)
    x_labels = query_start_list[::5]  # Slice the list to get every 5th entry
    ax[i].set_xticks(x_ticks)
    ax[i].set_xticklabels(x_labels, rotation=90)
    [ax[i].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]

if show_plot['runtime_resolution']:
    plt.show()
else:
    plt.close()



########################################################################################################################################
####                                              Distance for each resolution                                                     #####
ylims=[-10,2900]

fig, ax = plt.subplots(1,2, figsize=(12,6))
fig.subplots_adjust(top=0.84, bottom=0.1, left=0.07, right=0.9, hspace=0.35, wspace=0.2)
fig.suptitle("SubDTW Compression: Comparison of Matching Accuracy", fontweight=suptitle_weight, fontsize=suptitle_size)
fig.text(0.5, 0.9, f'Reference: {reference_name}        Query: {query_name}        Query Length: {query_length} s', ha='center', fontsize=title_size)

for i in range(len(resolution_list)):
 
    a = i*len(threshold_list)
    b = a + len(threshold_list)
    resolution = distance_array[a,0]
    threshold = distance_array[a:b,1]
    runtime_data = distance_array[a:b,2:]

    for j in range(len(threshold)):
        ax[i].plot(runtime_data[j,:], label=f'{threshold[j]:.0f}', color=colour_list[j], linewidth=plot_linewidth)

    if i == 1: # ony have one legend on the top left corner
        legend = ax[i].legend(bbox_to_anchor=(1.25, 1.025))
        legend.set_title('Threshold', {'size': 'medium', 'weight': 'normal', 'style': 'normal'})

    ax[i].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
    ax[i].set_ylabel('Distance (m)', fontweight=label_weight, fontsize=label_size)
    ax[i].set_xlabel('Query Measurement', fontweight=label_weight, fontsize=label_size)
    ax[i].grid(which='both')
    ax[i].set_ylim(ylims)
    x_ticks = np.arange(0, len(query_start_list), 5)
    x_labels = query_start_list[::5]  # Slice the list to get every 5th entry
    ax[i].set_xticks(x_ticks)
    ax[i].set_xticklabels(x_labels, rotation=90)
    [ax[i].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]

if show_plot['distance_resolution']:
    plt.show()
else:
    plt.close()


########################################################################################################################################
####                                                Query Length vs Runtime                                                        #####
fig, ax = plt.subplots(figsize=(10,10))
fig.suptitle('SubDTW: Comparison of Query Length and Runtime', fontweight=title_weight, fontsize=suptitle_size)
fig.text(0.5, 0.92, f'Reference: {reference_name}        Query: {query_name}        Query Length: {query_length} s', ha='center', fontsize=title_size)
fig.subplots_adjust(top=0.9, bottom=0.09, left=0.1, right=0.95, hspace=0.26, wspace=0.2)
ax.scatter(df_global['Query Points'], df_global['Runtime (s)'], s=20, color=wisteria, zorder=2)
ax.set_xlabel('Query Length (number of points)', fontweight=label_weight, fontsize=label_size)
ax.set_ylabel('Algorithm Runimte (s)', fontweight=label_weight, fontsize=label_size)
ax.grid(which='both', zorder=0)
[ax.spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]

if show_plot['query_length_runtime']:
    plt.show()
else:
    plt.close()


########################################################################################################################################
####                                            Resolution vs Threshold vs Runtime                                                 #####
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('SubDTW Compression: Effect of Resolution \nand Threshold on Runtime', fontweight=suptitle_weight, fontsize=suptitle_size)
fig.subplots_adjust(top=0.97, bottom=0.015, left=0.11, right=0.91, hspace=0.2, wspace=0.2)
fig.text(0.5, 0.87, f'Reference: {reference_name}        Query: {query_name}        Query Length: {query_length} s', ha='center', fontsize=title_size)

ax.imshow(runtime_2d_res_thresh)
ax.set_title(f'Data Selection: {runtime_selection}')
ax.set_ylabel('Resolution (n,n)', fontweight=label_weight, fontsize=label_size)
ax.set_xlabel('Threshold (Event Count)', fontweight=label_weight, fontsize=label_size)
ax.set_xticks(range(len(threshold_list)))
ax.set_xticklabels(threshold_list)
ax.set_yticks(range(len(resolution_list)))
ax.set_yticklabels(resolution_list)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(ax.imshow(runtime_2d_res_thresh), cax=cax)
cbar.set_label('Algorithm Runtime (s)', fontweight=label_weight, fontsize=label_size)
if show_plot['res_thresh_runtime']:
    plt.show()
else:
    plt.close()


########################################################################################################################################
####                                            Resolution vs Threshold vs Distance                                                #####
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('SubDTW Compression: Effect of Resolution \nand Threshold on Accuracy', fontweight=suptitle_weight, fontsize=suptitle_size)
fig.subplots_adjust(top=0.93, bottom=0.015, left=0.1, right=0.9, hspace=0.2, wspace=0.2)
fig.text(0.5, 0.87, f'Reference: {reference_name}        Query: {query_name}        Query Length: {query_length} s', ha='center', fontsize=title_size)

ax.imshow(distance_2d_res_thresh)
ax.set_title(f'Data Selection: {distance_selection}')
ax.set_ylabel('Resolution (n,n)', fontweight=label_weight, fontsize=label_size)
ax.set_xlabel('Threshold (Event Count)', fontweight=label_weight, fontsize=label_size)
ax.set_xticks(range(len(threshold_list)))
ax.set_xticklabels(threshold_list)
ax.set_yticks(range(len(resolution_list)))
ax.set_yticklabels(resolution_list)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar.set_label('Matching Distance (m)', fontweight=label_weight, fontsize=label_size)
cbar = fig.colorbar(ax.imshow(distance_2d_res_thresh), cax=cax)
if show_plot['res_thresh_distance']:
    plt.show()
else:
    plt.close()









########################################################################################################################################
####                                                Old Plots (NO GENERIC)                                                         #####

#---- Distance vs Resolution ----#
# ylims=[-10,2700]
# fig, ax = plt.subplots(2,3, figsize=(12,8))
# fig.subplots_adjust(left=0.065, right=0.88,top=0.9, bottom=0.11, wspace=0.35, hspace=0.35)
# fig.suptitle("SubDTW Compression: Matching Accuracy Comparison", fontweight=suptitle_weight, fontsize=suptitle_size)

# for i in range(6):
#     # slice off data for the resolution
#     a = 6*i
#     b = a + 5
#     resolution = distance_array[a,0]
#     threshold = distance_array[a:b,1]
#     runtime_data = distance_array[a:b,2:]

#     if i < 3:
#         for j in range(len(threshold)):
#             ax[0][i].plot(distance_array[j,:], label=f'{threshold[j]}', color=colour_list[j], linewidth=plot_linewidth)
#             ax[0][i].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
#             ax[0][i].set_ylabel('Distance (m)', fontweight=label_weight, fontsize=label_size)
#             ax[0][i].set_xlabel('Query Measurement', fontweight=label_weight, fontsize=label_size)
#             ax[0][i].grid(which='both')
#             ax[0][i].set_ylim(ylims)
#             ax[0][i].set_xticks([0,2,4,6,8,10,12])
#             [ax[0][i].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]
#             if i == 2: # ony have one legend on the top left corner
#                 legend = ax[0][i].legend(title='Threshold', bbox_to_anchor=(1.5, 1.025))
#                 legend.set_title('Legend Title', {'size': 'medium', 'weight': 'normal', 'style': 'normal'})
#     else:
#         for j in range(len(threshold)):
#             ax[1][i-3].plot(distance_array[j,:], label=f'{threshold[j]}', color=colour_list[j], linewidth=plot_linewidth)
#             ax[1][i-3].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
#             ax[1][i-3].set_ylabel('Distance (m)', fontweight=label_weight, fontsize=label_size)
#             ax[1][i-3].set_xlabel('Query Measurement', fontweight=label_weight, fontsize=label_size)
#             ax[1][i-3].grid(which='both')
#             ax[1][i-3].set_ylim(ylims)
#             ax[1][i-3].set_xticks([0,2,4,6,8,10,12])
#             [ax[1][i-3].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]




#---- Runtime vs Resolution ----#

# ylims=[-2,55]
# fig, ax = plt.subplots(2,3, figsize=(12,8))
# fig.subplots_adjust(left=0.065, right=0.88,top=0.9, bottom=0.11, wspace=0.35, hspace=0.35)
# fig.suptitle("SubDTW Compression: Runtime Comparison using Resolution and Threshold", fontweight=suptitle_weight, fontsize=suptitle_size)
# for i in range(6):
#     a = 6*i
#     b = a + 5
#     resolution = runtime_array[a,0]
#     threshold = runtime_array[a:b,1]
#     runtime_data = runtime_array[a:b,2:]

#     if i < 3:
#         for j in range(len(threshold)):
#             ax[0][i].plot(runtime_data[j,:], label=f'{threshold[j]}', color=colour_list[j], linewidth=plot_linewidth)
#             ax[0][i].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
#             ax[0][i].set_ylabel('Runtime (s)', fontweight=label_weight, fontsize=label_size)
#             ax[0][i].set_xlabel('Query Measurement', fontweight=label_weight, fontsize=label_size)
#             ax[0][i].grid(which='both')
#             ax[0][i].set_ylim(ylims)
#             ax[0][i].set_xticks([0,2,4,6,8,10,12])
#             [ax[0][i].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]
#             if i == 2: # ony have one legend on the top left corner
#                 legend = ax[0][i].legend(title='Threshold', bbox_to_anchor=(1.5, 1.025))
#                 legend.set_title('Legend Title', {'size': 'medium', 'weight': 'normal', 'style': 'normal'})

#     else:
#         for j in range(len(threshold)):
#             ax[1][i-3].plot(runtime_data[j,:], label=f'{threshold[j]}', color=colour_list[j], linewidth=plot_linewidth)
#             ax[1][i-3].set_title(f'Resolution: ({resolution:.0f}-{resolution:.0f})', fontweight=title_weight, fontsize=title_size)
#             ax[1][i-3].set_ylabel('Runtime (s)', fontweight=label_weight, fontsize=label_size)
#             ax[1][i-3].set_xlabel('Query Measurement', fontweight=label_weight, fontsize=label_size)
#             ax[1][i-3].grid(which='both')
#             ax[1][i-3].set_ylim(ylims)
#             ax[1][i-3].set_xticks([0,2,4,6,8,10,12])
#             [ax[1][i-3].spines[border].set_linewidth(plot_boxwidth) for border in ['top', 'bottom', 'left', 'right']]