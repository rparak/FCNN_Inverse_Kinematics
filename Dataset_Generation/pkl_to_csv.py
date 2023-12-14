# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Pandas (Data analysis and manipulation) [pip3 install pandas]
import pandas as pd
# Custom Lib.:
#   ../Lib/Utilities/File_IO
import Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 100000

def main():
    """
    Description:
        A program to convert generated data from 'generate.py' in 'pkl' format to 'zip' format, which 
        contains a 'csv' file with the data.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Create a file path to read/write the data.
    file_path_rw = f'{project_folder}/Data/Dataset/Config_N_{CONST_NUM_OF_DATA}'

    # Read the data from a file (*.pkl).
    data = File_IO.Load(file_path_rw, 'pkl', ',')

    # Create the structure of the output dataset.
    #   Note:
    #       'x_coord', 'y_coord': 
    #           Coordinates of the x-axis, y-axis (in meters) corresponding to the 
    #           absolute positions of the joints.
    #       'cfg': The configuration of the solution. The IK for the RR robotic structure 
    #              has two solutions.
    #       'th_0', 'th_1': Absolute position of the robot's joints.
    df = pd.DataFrame({'x_coord': data[:, 0].astype('float32'),
                       'y_coord': data[:, 1].astype('float32'),
                       'cfg': data[:, 2].astype('uint8'),
                       'th_0': data[:, 3].astype('float32'),
                       'th_1': data[:, 4].astype('float32')})

    # Define the compression of the output data.
    compression_dict = dict(method='zip', archive_name=f'Config_N_{CONST_NUM_OF_DATA}.csv')  

    # Save the data to the file (*.zip).
    #   Note:
    #       f'{file_path_rw}.zip' contains f'Config_N_{CONST_NUM_OF_DATA}.csv'
    df.to_csv(f'{file_path_rw}.zip', index=False, compression=compression_dict)

if __name__ == "__main__":
    sys.exit(main())

