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
#   Type: 
#       Type of dataset.
#   Num_of_Data: 
#       Number of data to be generated.
CONST_DATASET_CONFIG = {'Type': 1, 'Number_of_Data': 100}

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    file_path = []; N_nn = 1 if CONST_DATASET_CONFIG['Type'] == 1 else 2
    for i in range(N_nn):
        # Create a file path to save the data.
        file_path_tmp = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_CONFIG['Type']}/Config_N_{CONST_DATASET_CONFIG['Number_of_Data']}_ID_{i}"

        # Remove the '*.urdf' file if it already exists.
        if os.path.isfile(f'{file_path_tmp}.pkl'):
            os.remove(f'{file_path_tmp}.pkl')

        # Store the path to the file.
        file_path.append(file_path_tmp)


    # Start the timer.
    t_0 = time.time()

    i = 0; data_1 = []; data_2 = []
    while CONST_DATASET_CONFIG['Number_of_Data'] > i:
        # ...
        theta_rand = np.random.uniform(Robot_Str.Theta.Limit[:, 0], Robot_Str.Theta.Limit[:, 1])
        
        # ...
        T_rand = Kinematics.Forward_Kinematics(theta_rand, 'Fast', Robot_Str)[1]

        # ...
        if N_nn == 1:
            # ...
            data_1.append(np.append(np.append(T_rand.p.all(), T_rand.Get_Rotation('QUATERNION').all()), 
                                    theta_rand))
        else:
            if CONST_DATASET_CONFIG['Type'] == 2:
                # ...
                pass
            elif CONST_DATASET_CONFIG['Type'] == 3:
                # ...
                pass

        i += 1

    for _, (file_path_i, data_i) in enumerate(zip(file_path, [data_1, data_2])):
        # Save the data to the file.
        File_IO.Save(file_path_i, data_i, 'pkl', ',')

    # Display information.
    print('[INFO] The data generation has been successfully completed.')
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

if __name__ == "__main__":
    sys.exit(main())

