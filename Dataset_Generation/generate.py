# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Kinematics/Core
import Kinematics.Core
#   ../Parameters/Robot
import Parameters.Robot
#   ../Lib/Utilities/File_IO
import Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 100000

def main():
    """
    Description:
        The main program that generates the dataset of the selected robotic structure.

        The structure of the dataset is described below.
            Input of the NN:  x -> Position(x, y); configuration_id(0, 1)
            Output of the NN: y -> theta(0 .. n)
            
            Where n is the number of absolute joint positions.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to save the data.
    file_path = f"{project_folder}/Data/Dataset/Config_N_{CONST_NUM_OF_DATA}"

    # Remove the '*.urdf' file if it already exists.
    if os.path.isfile(f'{file_path}.pkl'):
        os.remove(f'{file_path}.pkl')

    # Initialization of data to show the process flow.
    percentage_offset = CONST_NUM_OF_DATA/10; percentage_idx = 1

    # Start the timer.
    t_0 = time.time()
    print('[INFO] The generation of the dataset is in progress.')

    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0; data = []; tolerance = 4
    while CONST_NUM_OF_DATA > i:
        # Random generation of absolute joint orientations.
        #   Note:
        #       The boundaries of the random generation are defined in the object structure.
        theta_rand = np.random.uniform(Robot_Str.Theta.Limit[:, 0], Robot_Str.Theta.Limit[:, 1])

        # Obtain the x, y coordinates using forward kinematics.
        p_tmp = np.round(Kinematics.Core.Forward_Kinematics(theta_rand, Robot_Str)[1], tolerance).astype('float32')

        # Obtain the solutions of the absolute positions of the joints.
        th_tmp = Kinematics.Core.Inverse_Kinematics(p_tmp, Robot_Str)[1]

        # If there is a duplicate of the input data, skip to the next step.
        if data != []:
            for _, x_i in enumerate(data):
                if p_tmp in x_i[0:2]:
                    continue

        # Data structure.
        #   Position (p), configuration(0: theta solution 0, 1: theta solution 1) and absolute position of the joint (theta).
        for id, th_tmp_i in enumerate(th_tmp):
            data_x = np.concatenate((p_tmp, np.array([id]).astype('float32')))
            data_y = np.round(th_tmp_i, tolerance).astype('float32')
            # Store the acquired data.
            data.append(np.concatenate((data_x, data_y), dtype=np.float32))

        i += 2
        if i > (percentage_offset * percentage_idx):
            print(f'[INFO]  Percentage: {int(100 * float(i)/float(CONST_NUM_OF_DATA))} %')
            print(f'[INFO]  Time: {(time.time() - t_0):.3f} in seconds')
            percentage_idx += 1

    # Display information (1).
    print(f'[INFO]  Percentage: {int(100 * float(i)/float(CONST_NUM_OF_DATA))} %')
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

    # Save the data to the file.
    File_IO.Save(file_path, data, 'pkl', ',')
    print(f'[INFO] The file has been successfully saved.')

    # Display information (2).
    print(f'[INFO] Number of processed data: {i}')
    if len(data) == CONST_NUM_OF_DATA:
        print('[INFO] The data generation has been successfully completed.')
    else:
        print(f'[WARNING] Insufficient number of combinations.')

if __name__ == "__main__":
    sys.exit(main())

