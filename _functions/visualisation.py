import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

filename_dict ={'sunset1' : 'dvs_vpr_2020-04-21-17-03-03.mat',
                'sunset2' : 'dvs_vpr_2020-04-22-17-24-21.mat',
                'night'   : 'dvs_vpr_2020-04-27-18-13-29.mat',
                'daytime' : 'dvs_vpr_2020-04-24-15-12-03.mat',
                'morning' : 'dvs_vpr_2020-04-28-09-14-11.mat',
                'sunrise' : 'dvs_vpr_2020-04-29-06-20-23.mat'}


def event_visualisation(dataset, start_time, hold_time=0.1, cols=346, rows=260, count_threshold=30):
    M = np.zeros((rows, cols))
    start_index = np.argmax(dataset[:,0]>=start_time)
    index = start_index
    while True:
        event = dataset[index]
        # print(event)
        if (event[0] <= start_time + hold_time):
            if event[3] == 1: # increment if positive polarity
                M[int(event[2]), int(event[1])] += 1
            elif event[3] == 0 or event[3]==-1: # decrement if negative polarity
                M[int(event[2]), int(event[1])] -= 1
        else:
            break
        index += 1
        
        # print every 1e6 points
        # if index % 1e6 == 0:
        #     print(f'File Time: {dataset[index][0]}')
        
        # check the new index is within bounds before starting the next loop
        if index >= dataset.shape[0]:
            break

    if count_threshold != -1:
        M[M>count_threshold] = count_threshold
        M[M<(-count_threshold)] = -count_threshold

    return M

def get_video_frame(file_name, time, fps=30):
    video_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/videos/'

    # filename_dict ={'sunset1' : '20200421_170039-sunset1_concat.mp4',
    #                 'sunset2' : '20200422_172431-sunset2_concat.mp4'} 
    
    # video_offset = {'sunset1' : 4.4,
    #                 'sunset2' : -4.4}
    

    filename_dict ={'sunset1' : '20200421_170039-sunset1_concat.mp4',
                    'sunset2' : '20200422_172431-sunset2_concat.mp4',
                    'night'   : '20200427_181204-night_concat.mp4',
                    'daytime' : '20200424_151015-daytime_concat.mp4',
                    'morning' : '20200428_091154-morning_concat.mp4',
                    'sunrise' : '20200429_061912-sunrise_concat.mp4'} 
    
    video_offset = {'sunset1' : 5,
                    'sunset2' : -5,
                    'daytime' : 0,
                    'sunrise' : 4.5,
                    'morning' : 0,
                    'night'   : 1.5}
    
    # print(f'{time} \t {file_name} \t {video_offset[file_name]}')
    time = time + video_offset[file_name] 
    start_frame_index = time * fps

    #----- Open Video Object using cv2 -----#
    cap = cv2.VideoCapture(os.path.join(video_dir, filename_dict[file_name]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index) # set the start time

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    else: 
        ret, frame = cap.read() # read frame
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame from BGR to RGB
        else:
            print('ret=0')
            frame = None
    
    return frame


# def get_video_frame(file_name, time, fps=30):
#     video_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/videos/'

#     filename_dict ={'sunset1' : '20200421_170039-sunset1_concat.mp4',
#                     'sunset2' : '20200422_172431-sunset2_concat.mp4',
#                     'night'   : '20200427_181204-night_concat.mp4',
#                     'daytime' : '20200424_151015-daytime_concat.mp4',
#                     'morning' : '20200428_091154-morning_concat.mp4',
#                     'sunrise' : '20200429_061912-sunrise_concat.mp4'} 
    
#     video_offset = {'sunset1' : 5,
#                     'sunset2' : -5.5,
#                     'daytime' : 0,
#                     'sunrise' : 4.5,
#                     'morning' : 0,
#                     'night'    : 2}
    
#     time = time + video_offset[file_name] 
#     start_frame_index = time * fps

#     #----- Open Video Object using cv2 -----#
#     cap = cv2.VideoCapture(os.path.join(video_dir, filename_dict[file_name]))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index) # set the start time

#     # Check if camera opened successfully
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")
#     else: 
#         ret, frame = cap.read() # read frame
#         if ret == True:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame from BGR to RGB
#         else:
#             print('ret=0')
#             frame = None
    
#     return frame




def event_video_stream(data_set, filename, threshold=-1, resolution=-1, hold_time=0.1, save=0, save_directory=-1):

    #---- Unpack the parameters
    if save==1 and save_directory==-1:
        raise  ValueError('Please provide a save directory')

    if save:
        print_en = 1
    else:
        print_en = 0

    mat_name = filename_dict[filename][:-4]


    if resolution==-1:
        cols=346
        rows=260
        resolution_string = 'Compression: Full Dataset'
    else:
        if isinstance(resolution, int) == True:
            cols = resolution
            rows = resolution
            resolution_string = f'Resolution: ({resolution}, {resolution})'
        else:
            res_split = resolution.split('-')
            cols = int(res_split[0][1:])+1
            rows = int(res_split[1][:-1])+1

    if threshold==-1:
        threshold_string = ''
    else:
        if isinstance(threshold, int):
            threshold_string = f'Threshold: {threshold}'
        elif isinstance(threshold, list):
            threshold_string = f'Threshold:  Postive: {threshold[0]} Negative: {threshold[1]}'

    
    compression_string = f'{resolution_string}          {threshold_string}'

    #---- Loop through file create frames ----#
    time_array = np.arange(0,data_set[-1,0]-hold_time,hold_time)
    fps = 1/hold_time
    # Plot params
    suptitle_size = 16
    suptitle_weight ='bold'

    M_list = []
    max_val = 0
    for start_time in tqdm(time_array, "Creating video frames"):
        M = event_visualisation(data_set, start_time, hold_time, cols=cols, rows=rows, count_threshold=10)
        current_max_val = np.max(abs(M))
        # current_min_val = np.min(M)
        if current_max_val > max_val:
            max_val = current_max_val
        M_list.append(M)
        # print(f'Max: {current_max_val} \t Min: {current_min_val}')
    M_array = np.asarray(M_list)

    def update_frame(frame, print_en=print_en):
        plt.clf()
        plt.suptitle(f'{filename}: {mat_name}', fontweight=suptitle_weight, fontsize=suptitle_size)  
        plt.title(f'Time: {time_array[frame]:.1f} s \n'+compression_string, ha='left', x=-0)
        plt.axis('off')
        im = plt.imshow(M_array[frame], cmap='bwr', vmax=max_val, vmin=-max_val)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Accumulated Polarity')
        if print_en:
            print(f'Time: {time_array[frame]:.1f}')

    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.2)

    num_frames = M_array.shape[0]
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames, repeat=False, blit=False)
    writervideo = animation.FFMpegWriter(fps=fps) 
    if save:
        anim.save(filename=save_directory+mat_name+'.mp4', writer=writervideo)
    else:
        plt.show()
    return anim



def event_camera_bias(data_set, filename, threshold=-1, resolution=-1, hold_time=1, save=0, save_directory=-1, bias_threshold=100):

    #---- Unpack the parameters
    if save==1 and save_directory==-1:
        raise  ValueError('Please provide a save directory')

    if save:
        print_en = 1
    else:
        print_en = 0

    mat_name = filename_dict[filename][:-4]

    if threshold==-1 or resolution==-1:
        compression_string = 'Compression: Full Dataset'
    else:
        compression_string = f'Resolution: {resolution} \t Threshold: {threshold}'

    if resolution==-1:
        cols=346
        rows=260
    else:
        res_split = resolution.split('-')
        cols = int(res_split[0][1:])+1
        rows = int(res_split[1][:-1])+1

    #---- Loop through file create frames ----#
    time_array = np.arange(0,data_set[-1,0]-hold_time,hold_time)
    # time_array = [10]
    fps = 1/hold_time
    # Plot params
    suptitle_size = 16
    suptitle_weight ='bold'

    M_accumulator = np.zeros((rows, cols))
    M_list = []
    max_val = 0
    max_val_list = []

    # for start_time in tqdm(time_array, "Creating video frames"):
    for start_time in time_array:
        if start_time % 5 == 0:
            print(start_time)
        M = event_visualisation(data_set, start_time, hold_time, cols=cols, rows=rows, count_threshold=-1)
        current_max_val = np.max(abs(M))
        max_val_list.append(np.max(abs(M)))
        # current_min_val = np.min(M)
        if current_max_val > max_val:
            max_val = current_max_val
        M_accumulator = M_accumulator + M
        M_list.append(M_accumulator)
        max_val_list.append(np.max(abs(M_accumulator)))

        # print(f'Max: {current_max_val} \t Min: {current_min_val}')
    M_array = np.asarray(M_list)
    max_val_array = np.asarray(max_val_list)    


    def update_frame(frame, print_en=print_en):
        plt.clf()
        plt.suptitle(f'{filename}: {mat_name}', fontweight=suptitle_weight, fontsize=suptitle_size)  
        plt.title(f'Time: {time_array[frame]:.2f} s \n'+compression_string, ha='left', x=-0)
        plt.axis('off')
        im = plt.imshow(M_array[frame], cmap='bwr', vmax=bias_threshold, vmin=-bias_threshold)
        # im = plt.imshow(M_array[frame], cmap='bwr', vmax=max_val_array[frame], vmin=-max_val_array[frame])
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Accumulated Polarity')
        if print_en:
            print(f'Time: {time_array[frame]:.1f}')



    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.2)

    num_frames = M_array.shape[0]
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames, repeat=False, blit=False)
    writervideo = animation.FFMpegWriter(fps=fps) 
    if save:
        anim.save(filename=save_directory+mat_name+f'_{hold_time}.mp4', writer=writervideo)
    else:
        plt.show()
    return anim


def gps_visualisation(filename, save=0):
    gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/'
    gps_file_dict = {'sunset1' : 'sunset1_gps.csv',
                     'sunset2' : 'sunset2_gps.csv',
                     'night'   : 'night_gps.csv',
                     'daytime' : 'daytime_gps.csv',
                     'morning' : 'morning_gps.csv',
                     'sunrise' : 'sunrise_gps.csv'}
    
    gps_data = np.genfromtxt(gps_dir + gps_file_dict[filename], delimiter=',')


    #---- Loop through file create frames ----#

    time_array = gps_data[:,6]
    fps = 1
    # Plot params
    suptitle_size = 16
    suptitle_weight ='bold'

    

    def update_frame(frame):
        plt.clf()
        plt.suptitle(f'{filename}', fontweight=suptitle_weight, fontsize=suptitle_size)  
        plt.title(f'Time: {time_array[frame]:.1f} s', ha='left', x=-0)
        plt.axis('off')
        plt.plot(gps_data[:,1], gps_data[:,0])
        plt.scatter(gps_data[frame,1], gps_data[frame,0], color='red', zorder=4)
        print(time_array[frame])


    fig = plt.figure(figsize=(10,8))
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.9, hspace=0.2, wspace=0.2)

    num_frames = len(time_array)
    
    anim = animation.FuncAnimation(fig, update_frame, frames=num_frames, repeat=False, blit=False)
    writervideo = animation.FFMpegWriter(fps=fps) 
    if save:
        anim.save(filename=filename+'.mp4', writer=writervideo)
    else:
        plt.show()

    return anim

