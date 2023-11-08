import sys
import os
function_dir = 'C:/Users/angus/Documents/git_repositories/ENGN4350_Honours/subsequence_dtw/'
sys.path.append(function_dir)
import argparse
import numpy as np
from scipy.io import loadmat  
import cv2 
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import _functions.visualisation as visualisation

# plt.rcParams['text.usetex'] = True

#---- Parser Arguments ----#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to create a video stream of the event sequence")
    parser.add_argument('-f', '--file', type=str, nargs='?', required=False, help='Filepath of event data')
    parser.add_argument('-t', '--hold_time', type=float, nargs='?', required=False, help='Hold time for event integration')
    parser.add_argument('-s', '--save', type=int, nargs='?', required=False, help='Save animation: Yes[1]/No[0]')
    parser.add_argument('-pt', '--positive_threshold', type=int, nargs='?', required=False, help='Positive Threshold')
    parser.add_argument('-nt', '--negative_threshold', type=int, nargs='?', required=False, help='Negative Threshold')
    parser.add_argument('-c', '--compressed', type=int, nargs='?', required=True, help='Is data compressed: Yes[1]/No[0]')
    parser.add_argument('-r', '--resolution', type=int, nargs='?', required=False, help='Resolution of the image')
    args = parser.parse_args()

file_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/mat_files/'
filename = 'sunrise'
save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_event_visualisation/'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}


#---- Apply args ----#
if args.file and args.compressed:
    file_directory_string = args.file
    compression_info = file_directory_string.split('\\')[-2]
    compression_info_split = compression_info.split('_')
    resolution = compression_info_split[1]
    threshold = compression_info_split[3]
else:
    resolution = -1
    threshold = -1 

if args.resolution:
    resolution = args.resolution

if args.file:
    filename = args.file
    mat_file_name = filename.split("\\")[-1][:-4]
    video_name = mat_file_name
else:
    mat_file_name = filename_dict[filename][:-4]
    filename = file_directory + filename_dict[filename]
    video_name = mat_file_name

if args.hold_time:
    hold_time = args.hold_time
else:
    hold_time = 0.1

if args.save:
    save = 1
else:
    save = 0

filename_string = [key for key, value in filename_dict.items() if value == mat_file_name+'.mat'][0]

if args.compressed and args.save:
    save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_event_visualisation/_compressed/'+compression_info+'/'
elif not args.compressed and args.save:
    save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_event_visualisation/'
    save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_camera_bias/_accumulation_events/'
else:
    save_directory = -1
    
if not os.path.exists(save_directory) and args.save:
    os.makedirs(save_directory)

#---- Load Data ----#
data_dir = 'F:/Data/Output_Data/full_dataset/spatial_compression/resolution_[100-100]/pos_44_neg_-32/dvs_vpr_2020-04-22-17-24-21.mat'
save_directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Figures/_event_visualisation/_compressed/resolution_[100-100]_pthresh_44_nthresh_-32/'
filename_string = 'sunset2'


data = loadmat(data_dir)['data']

resolution = 100
threshold = [44, -32]


# subtract time offset if t_0 != 0
if data[0,0] != 0:
    data[:,0] -= data[0,0]

anim = visualisation.event_video_stream(data, filename_string, hold_time=hold_time, resolution=resolution, threshold=threshold, save=save, save_directory=save_directory)
# anim = visualisation.event_camera_bias(data, filename_string, hold_time=hold_time, resolution=resolution, threshold=threshold, save=save, save_directory=save_directory)
