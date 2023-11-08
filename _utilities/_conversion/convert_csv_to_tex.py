import pandas as pd
import argparse

# Manually enter the file
directory = 'C:/Users/angus/OneDrive - Australian National University/Honours/Data/Output Data/subsequence_dtw/compression/resolution_[100-100]/dtw_output/'
filename = 'data_1/batch_data.csv'

# Set up parser arguments to interact from the commandline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print the contents of a CSV file in Latex format.')
    parser.add_argument('-f', '--file_path', type=str, nargs='?', required=False, help='Path to the CSV file.')
    args = parser.parse_args()

# Use the file path provided by the command line if it is available
if args.file_path:
    file_to_load = args.file_path
    print('Loading file path from command line \n \n')
else:
    file_to_load = directory + filename
    print('Using the file path provided in the script \n \n')


# Load the CSV file as a pandas dataframe
df = pd.read_csv(file_to_load, delimiter=',')

# Print the CSV in Latex format
print(df.to_latex(index=False))