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
#   ../Parameters/Robot
import Parameters.Robot
#   ../FCNN_IK/Model
import FCNN_IK.Model

"""
Notes:
    A command to kill all Python processes within the GPU.
    $ ../>  sudo killall -9 python
"""

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# A dataset configuration that specifies the amount of data 
# generated to train the model.
CONST_NUM_OF_DATA = 100000

def main():
    """
    Description:
        A program for hyperparameter optimization of the Fully-Connected Neural Network (FCNN) to solve 
        Inverse Kinematics (IK) task.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to read/write the data.
    file_path_r = f'{project_folder}/Data/Dataset/Config_N_{CONST_NUM_OF_DATA}'
    file_path_w = f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}'

    # Read the data from a file (*.zip).
    #   Note:
    #       f'{file_path_r}.zip' contains f'Config_N_{CONST_NUM_OF_DATA}.csv'
    data = pd.read_csv(f'{file_path_r}.zip', compression='zip')

    # Assign data to variables.
    #   Input:
    #       'x_coord', 'y_coord': 
    #           Coordinates of the x-axis, y-axis (in meters) corresponding to the 
    #           absolute positions of the joints.
    #       'cfg': The configuration of the solution. The IK for the RR robotic structure 
    #              has two solutions.
    #   Output:
    #       'th_0', 'th_1': Absolute position of the robot's joints.
    x = data.iloc[:, 0:(Robot_Str.Theta.Zero.size + 1)].values; y = data.iloc[:, -Robot_Str.Theta.Zero.size:].values
    
    # Optimization of the hyperparameters for the Fully-Connected Neural Network (FCNN).
    #   1\ Initialization.
    FCNN_IK_Optimizer_Cls = FCNN_IK.Model.FCNN_Optimizer_Cls(x=x, y=y, train_size=0.80, test_size=0.20, 
                                                             file_path=file_path_w)
    #   2\ Optimization.
    FCNN_IK_Optimizer_Cls.Optimize(num_of_trials=100, epochs_per_trial=100, batch_size=64, 
                                   save_results=True)

if __name__ == "__main__":
    sys.exit(main())
