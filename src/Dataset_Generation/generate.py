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
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000
#   Number of dataset types.
CONST_NUM_OF_DATASET_TYPES = 2
#   The number of datasets in each type.
#       Note:
#           Index 0: Dataset Type 1
#           Index 1: Dataset Type 2
CONST_NUM_OF_DATASETS = [1, 2]


def main():
    """
    Description:
        The main program that generates the dataset of the selected robotic structure.

        The program creates two types of datasets, which are described below.
            Dataset Type 1.
                DNN ID 0:
                    Input of the NN:  x -> Position(x, y, z); Orientation(quaternion)
                    Output of the NN: y -> theta(0 .. n)
            Dataset Type 2.
                DNN ID 0:
                    Input of the NN:  x -> Position(x, y, z), Orientation(quaternion)
                    Output of the NN: y -> theta(0 .. n - 1) 
                DNN ID 1:
                    Input of the NN:  x -> Position(x, y, z), Orientation(quaternion), theta(0 .. n - 1)
                    Output of the NN: y -> theta(n)
            
            Where n is the number of absolute joint positions.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to save the data.
    file_path = []
    for i in range(CONST_NUM_OF_DATASET_TYPES):
        for j in range(CONST_NUM_OF_DATASETS[i]):
            file_path_tmp = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{i + 1}/Config_N_{CONST_NUM_OF_DATA}_ID_{j}"

            # Remove the '*.urdf' file if it already exists.
            if os.path.isfile(f'{file_path_tmp}.pkl'):
                os.remove(f'{file_path_tmp}.pkl')

            # Store the path to the file.
            file_path.append(file_path_tmp)

    # Start the timer.
    t_0 = time.time()

    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0; data_t_1 = []; data_t_2_1 = []; data_t_2_2 = []
    while CONST_NUM_OF_DATA > i:
        # Random generation of absolute joint orientations.
        #   Note:
        #       The boundaries of the random generation are defined in the object structure.
        theta_rand = np.random.uniform(Robot_Str.Theta.Limit[:, 0], Robot_Str.Theta.Limit[:, 1])
        
        # Obtain the homogeneous transformation matrix using forward kinematics.
        T_rand = Kinematics.Forward_Kinematics(theta_rand, 'Fast', Robot_Str)[1]

        # Store the acquired data.
        #   Dataset Type 1.
        #       ID 0.   
        data_t_1.append(np.append(np.append(T_rand.p.all(), T_rand.Get_Rotation('QUATERNION').all()), 
                                theta_rand))
        #   Dataset Type 2.
        #       ID 0.
        data_t_2_1.append(np.append(np.append(T_rand.p.all(), T_rand.Get_Rotation('QUATERNION').all()), 
                                    theta_rand[0:Robot_Str.Theta.Zero.size - 1]))
        #       ID 1.
        data_t_2_2.append(np.append(np.append(T_rand.p.all(), T_rand.Get_Rotation('QUATERNION').all()), 
                                    theta_rand))

        i += 1

    # Save the data to the file.
    for _, (file_path_i, data_i) in enumerate(zip(file_path, [data_t_1, data_t_2_1, data_t_2_2])):
        File_IO.Save(file_path_i, data_i, 'pkl', ',')

    # Display information.
    print('[INFO] The data generation has been successfully completed.')
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

if __name__ == "__main__":
    sys.exit(main())

