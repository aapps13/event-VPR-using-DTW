import numpy as np
import os

def sort_files(data_dir, identifier):
    assert os.path.exists(data_dir), "Data filepath does not exist"

    #---- Identify files ----#
    identified_files = [file for file in os.listdir(data_dir) if file.startswith(identifier)]

    #---- Sort_files ----#
    start_time = []
    for file in identified_files:
        # index_1 = file.find('[')
        # index_2 = file.find('-')
        thresh = file.split('_')[1]
        start_time.append(float(thresh))
        # start_time.append(float(file[index_1+1:index_2]))
    # sort the start time and mask filenames in ascending order
    sorted_indices = [i for i, v in sorted(enumerate(start_time), key=lambda x: x[1])]
    sorted_files = [identified_files[i] for i in sorted_indices]

    return sorted_files 


def combine_files(data_dir, identifier, save_name):
    assert os.path.exists(data_dir), "Data filepath does not exist"

    sorted_files = sort_files(data_dir, identifier)

    accumulated_list = []

    for i, file in enumerate(sorted_files):
        data = np.load(os.path.join(data_dir,file))
        accumulated_list.append(data)

    np.save(data_dir+f'/{save_name}', accumulated_list)

    return


# Define a custom sorting key function
def get_resolution_and_threshold(item):
    # Split the string by underscores
    parts = item.split('_')

    # Extract the resolution part
    resolution_part = parts[1]
    # Remove the square brackets and split into integers
    resolution = tuple(map(int, resolution_part[1:-1].split('-')))
    
    # Extract the threshold part
    threshold_part = parts[-1]
    threshold = int(threshold_part)
    
    return (resolution, -threshold)

