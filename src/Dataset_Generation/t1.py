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
CONST_DATASET_CONFIG = {'Type': 2, 'Number_of_Data': 100}

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to save the data.
    file_path = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_CONFIG['Type']}/Config_N_{CONST_DATASET_CONFIG['Number_of_Data']}_ID_1"

    # Save the data to the file.
    data = File_IO.Load(file_path, 'pkl', ',')

    print(np.array(data, dtype=np.float32))

if __name__ == "__main__":
    sys.exit(main())

