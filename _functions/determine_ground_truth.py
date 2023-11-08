import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import pynmea2

linux = 0
if linux == 1:
    gps_dir = '/media/aapps/Elements/Data/Brisbane_Event_Data/gps_files/'
else:
    # gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/'
    gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_full_data_interp/'
    # gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_compression_interpolation/v2/'

gps_file_dict = {'sunset1' : 'sunset1_gps.csv',
                 'sunset2' : 'sunset2_gps.csv',
                 'night'   : 'night_gps.csv',
                 'daytime' : 'daytime_gps.csv',
                 'morning' : 'morning_gps.csv',
                 'sunrise' : 'sunrise_gps.csv'}

video_offset = {'sunset1' : 5,
                'sunset2' : -5.5,
                'daytime' : 0,
                'sunrise' : 4.5,
                'morning' : 0,
                'night'   : 2}

gps_offset = {'sunset1' : 4.5,
              'sunset2' : -5,
              'daytime' : -2,
              'sunrise' : 4.5,
              'morning' : -2,
              'night'   : 0}





def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0

    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                # print(msg.timestamp, msg.latitude, msg.longitude)
                # print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T



def calc_ground_truth(query_name, query_time, reference_name, reference_time, linux=0):
    '''
    GPS csv header order:
        latiude(deg), longitude(deg), altitude(m), distance travelled(m), speed(m/s)
        time of day(HHMMSS), elapsed time(s)     
        
    '''
    if linux == 1:
        gps_dir = '/media/aapps/Elements/Data/Brisbane_Event_Data/gps_files/'
    else:
        # gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/'
        gps_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_full_data_interp/'

    #---- Load Data ----#
    query_gps_path = os.path.join(gps_dir, gps_file_dict[query_name])
    reference_gps_path = os.path.join(gps_dir, gps_file_dict[reference_name])
    query_gps = np.genfromtxt(query_gps_path, delimiter=',')
    reference_gps = np.genfromtxt(reference_gps_path, delimiter=',')
    # round time to nearest integer to match the GPS data
    query_time_int = round(query_time + gps_offset[query_name])
    reference_time_int = round(reference_time + gps_offset[reference_name])


    #---- Determine Ground Truth Values ----#
    # reference_path = reference_gps[:,0:2]
    # query_position = query_gps[query_time_int-1,0:2]
    # estimated_position = reference_gps[reference_time_int-1,0:2]
    reference_path = reference_gps[:,1:3]
    query_position = query_gps[query_time_int-1,1:3]
    estimated_position = reference_gps[reference_time_int-1,1:3]

    # calculate the distance between the points
    distance = haversine_distance(estimated_position[1], estimated_position[0], query_position[1], query_position[0])

    # find the timestamp of the position in the reference
    closest_distance = float('inf')
    for index, coordinate in enumerate(reference_path):
        current_distance = haversine_distance(query_position[0], query_position[1], coordinate[0], coordinate[1])
        if current_distance < closest_distance:
                closest_distance = current_distance
                closet_index = index
    # closet_reference_time = reference_gps[closet_index,6] - gps_offset[reference_name]
    closet_reference_time = reference_gps[closet_index,0] - gps_offset[reference_name]


    return reference_path, query_position, estimated_position, distance, closet_reference_time


def get_position(dataset_name, time):
    # load data
    gps_filepath = os.path.join(gps_dir, gps_file_dict[dataset_name])
    gps_data = np.genfromtxt(gps_filepath, delimiter=',')
    # find the closest time
    time_int = round(time + gps_offset[dataset_name])
    # find gps position (lat, lon)
    gps_position = gps_data[time_int, 0:2]
    return gps_position


def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate differences between latitudes and longitudes
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance


def calc_ground_truth_interp(query_name, query_time, reference_name, reference_time):

    #---- Load Data ----#
    # gps_interp_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_compression_interpolation/v4/'
    gps_interp_dir = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Datasets/gps_files/_compression_interpolation/pos_44_neg_-32/backup/'

    gps_file_dict = {'sunset1' : 'sunset1_gps.csv',
                     'sunset2' : 'sunset2_gps.csv',
                     'daytime' : 'daytime_gps.csv',
                     'morning' : 'morning_gps.csv',
                     'night'   : 'night_gps.csv'}
    
    query_gps = np.genfromtxt(gps_interp_dir+gps_file_dict[query_name], delimiter=',')
    reference_gps = np.genfromtxt(gps_interp_dir+gps_file_dict[reference_name], delimiter=',')

    #---- Determine Ground Truth Values ----#
    query_index = np.argmax(query_gps[:,0] >= query_time)
    reference_index = np.argmax(reference_gps[:,0] >= reference_time)

    # print(f'Query Time: {query_time} \t Index: {query_index} \t GPS at Index: {query_gps[query_index,:]}')
    # print(f'Estimated Time: {reference_time} \t Index: {reference_index} \t GPS at Index: {reference_gps[reference_index,:]}')

    reference_path = reference_gps[:,1:3]
    query_position = query_gps[query_index,1:3]
    estimated_position = reference_gps[reference_index,1:3]

    # calculate the distance between the points
    distance = haversine_distance(estimated_position[1], estimated_position[0], query_position[1], query_position[0])
    # print(distance)

    # find the timestamp of the position in the reference
    closest_distance = float('inf')
    closet_reference_index = -1
    for index, coordinate in enumerate(reference_path):
        current_distance = haversine_distance(query_position[0], query_position[1], coordinate[0], coordinate[1])
        if current_distance < closest_distance:
                closest_distance = current_distance
                closet_reference_index = index # reference_path[index,0]

    return reference_path, query_position, estimated_position, distance, closet_reference_index

