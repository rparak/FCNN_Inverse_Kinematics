# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Trajectory/Utilities
import Lib.Trajectory.Utilities
#   ../Configuration/Parameters
import Configuration.Parameters
#   ../Lib/FCNN_IK/Model
import Lib.FCNN_IK.Model


"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# Save the data to a file.
CONST_SAVE_DATA = False
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000
#   Method to be used for training.
#       Method 0: No test (validation) partition.
#       Method 1: With test (validation) partition.
CONST_DATASET_METHOD = 0

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Prediction of the absolute joint position of the robotic arm.
    #   1\ Initialization.
    if CONST_DATASET_METHOD == 0:
        FCNN_IK_Predictor_Cls = Lib.FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False.h5')
    elif CONST_DATASET_METHOD == 1:
        FCNN_IK_Predictor_Cls = Lib.FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True.h5')
    
    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Lib.Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.01)
    
    # Obtain the constraints for absolute joint positions in order to generate multi-axis position trajectories.
    (abs_j_pos_0, abs_j_pos_1) = Configuration.Parameters.Get_Absolute_Joint_Positions(Robot_Str.Name)

    # The tolerance of the data.
    tolerance = 4

    # Generation of multi-axis position trajectories from input parameters.
    theta_arr = []
    for _, (th_actual, th_desired) in enumerate(zip(abs_j_pos_0, abs_j_pos_1)):
        (theta_arr_i, _, _) = Polynomial_Cls.Generate(th_actual, th_desired, 0.0, 0.0, 0.0, 0.0,
                                                      Configuration.Parameters.CONST_T_0, Configuration.Parameters.CONST_T_1)
        theta_arr.append(theta_arr_i)

    # Obtain the homogeneous transformation matrix using forward kinematics from 
    # the generated multi-axis position trajectories.
    x = []; y = []; z = []; q_w = []; q_x = []; q_y = []; q_z = []
    for _, theta_arr_i in enumerate(np.array(theta_arr, dtype=np.float64).T):
        T = Kinematics.Forward_Kinematics(theta_arr_i, 'Fast', Robot_Str)[1]
        # Store the acquired data.
        #   Position (x, y, z).
        x.append(np.round(T.p.x, tolerance)); y.append(np.round(T.p.y, tolerance))
        z.append(np.round(T.p.z, tolerance))
        #   Quaternion (w, x, y, z)
        q = T.Get_Rotation('QUATERNION')
        q_w.append(np.round(q.w, tolerance)); q_x.append(np.round(q.x, tolerance))
        q_y.append(np.round(q.y, tolerance)); q_z.append(np.round(q.z, tolerance))

if __name__ == "__main__":
    sys.exit(main())
