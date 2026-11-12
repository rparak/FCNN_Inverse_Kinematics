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
#   ../Hyperparameters/Utilities
import Hyperparameters.Utilities

"""
Notes:
    A command to kill all Python processes within the GPU.
    $ ../> sudo killall -9 python
"""

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# A dataset configuration that specifies the amount of data 
# generated to optimize the model.
#   Note:
#       The optimization is run once (on N = 10000) and the results apply to the single
#       hyperparameter structure used for all dataset sizes, see ../Training/Hyperparameters/Utilities.py
CONST_NUM_OF_DATA = 10000
# Optimization configuration.
#   Note:
#       The trials use the same callbacks as the training (relative early stopping and relative
#       reduction of the learning rate, with the same relative thresholds 'early_stopping_min_delta'
#       and 'lr_min_delta'), only with shorter patience values and a limited number of epochs.
#   The number of trials.
CONST_NUM_OF_TRIALS = 30
#   The number of randomly sampled trials before the Bayesian optimization starts
#   (the remaining trials are selected by the Gaussian process).
CONST_NUM_OF_INITIAL_POINTS = 10
#   The maximum number of epochs per trial.
CONST_EPOCHS_PER_TRIAL = 400
#   The number of epochs without improvement of 'val_loss' after which the trial is stopped.
CONST_EARLY_STOPPING_PATIENCE = 100
#   The number of epochs without improvement of 'val_loss' after which the learning rate is reduced.
CONST_LR_PATIENCE = 20
#   The number of the best configurations to be displayed.
CONST_NUM_OF_TOP_TRIALS = 5

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
    #   Note:
    #       The results are saved to f'{file_path_w}_Optimizer_Best_Results.txt'
    file_path_w = f'{project_folder}/Training/Hyperparameters/FCNN_IK'

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
    
    # The batch size, the epsilon of the Adam optimizer and the relative improvement thresholds of the callbacks
    # are taken from the hyperparameter structure, so the optimization uses the same settings as the training.
    Hyperparameters_Str = Hyperparameters.Utilities.Get_Hyperparameter_Structure()

    # Optimization of the hyperparameters for the Fully-Connected Neural Network (FCNN).
    #   1\ Initialization.
    FCNN_IK_Optimizer_Cls = FCNN_IK.Model.FCNN_Optimizer_Cls(x=x, y=y, train_size=0.80, test_size=0.20, 
                                                             file_path=file_path_w)
    #   2\ Optimization.
    FCNN_IK_Optimizer_Cls.Optimize(num_of_trials=CONST_NUM_OF_TRIALS, epochs_per_trial=CONST_EPOCHS_PER_TRIAL, batch_size=Hyperparameters_Str['batch_size'],
                                   early_stopping_patience=CONST_EARLY_STOPPING_PATIENCE, early_stopping_min_delta=Hyperparameters_Str['early_stopping_min_delta'],
                                   lr_patience=CONST_LR_PATIENCE, lr_min_delta=Hyperparameters_Str['lr_min_delta'],
                                   num_of_initial_points=CONST_NUM_OF_INITIAL_POINTS, save_results=True, adam_epsilon=Hyperparameters_Str['adam_epsilon'],
                                   num_of_top_trials=CONST_NUM_OF_TOP_TRIALS)

if __name__ == "__main__":
    sys.exit(main())
