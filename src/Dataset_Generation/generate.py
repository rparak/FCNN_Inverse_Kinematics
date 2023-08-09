# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Library:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO


"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str

# ....
CONST_DATASET_CONFIG = {'Number_of_Data': 100, 'Type': 1, 'Id': [0, 1, 2]}

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    file_path = []
    for _, id_i in enumerate(CONST_DATASET_CONFIG['Id']):
        file_path_tmp = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_CONFIG['Type']}/Config_N_{CONST_DATASET_CONFIG['Number_of_Data']}_ID_{id_i}"

        # Remove the '*.urdf' file if it already exists.
        if os.path.isfile(f'{file_path_tmp}.txt'):
            os.remove(f'{file_path_tmp}.txt')

        # Create a file path to save the data.
        file_path.append(file_path_tmp)

    # Start the timer.
    t_0 = time.time()

    # ...
    theta_rand = np.round(np.float32(np.random.uniform(Robot_Str.Theta.Limit[:, 0], Robot_Str.Theta.Limit[:, 1])), 
                          decimals=4)

    # Save the data to the file.
    # ...

    # Display information.
    print('[INFO] The data generation has been successfully completed.')
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

if __name__ == "__main__":
    sys.exit(main())

