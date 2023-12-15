# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' not in sys.path:
    sys.path.append('../../')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Parameters/Robot
import Parameters.Robot
#   ../Kinematics/Core
import Kinematics.Core as Kinematics
#   ../Trajectory/Utilities
import Trajectory.Utilities
#   ../Configuration/Parameters
import Configuration.Parameters
#   ../FCNN_IK/Model
import FCNN_IK.Model

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# A dataset configuration that specifies the amount of data 
# generated to train the model.
CONST_NUM_OF_DATA = 1000

def main():
    """
    Description:
        The program to visualize both the desired and predicted Tool Center Point (TCP) coordinates in meters.

        The observation is tested on trajectories of the absolute positions of the robot's joints, generated 
        using a multi-axis polynomial profile.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Prediction of the absolute joint position of the robotic arm.
    #   1\ Initialization.
    FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_x.pkl', 
                                                             f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_y.pkl', 
                                                             f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}_use_val_True.h5')
    

    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.01)
    
    # Obtain the constraints for absolute joint positions in order to generate multi-axis position trajectories.
    (abs_j_pos_0, abs_j_pos_1) = Configuration.Parameters.Get_Absolute_Joint_Positions(Robot_Str.Name)

    # Generation of multi-axis position trajectories from input parameters.
    theta_arr = []
    for _, (th_actual, th_desired) in enumerate(zip(abs_j_pos_0, abs_j_pos_1)):
        (theta_arr_i, _, _) = Polynomial_Cls.Generate(th_actual, th_desired, 0.0, 0.0, 0.0, 0.0,
                                                      Configuration.Parameters.CONST_T_0, Configuration.Parameters.CONST_T_1)
        theta_arr.append(theta_arr_i)

    # The tolerance of the data.
    tolerance = 4

    # Obtain the x, y coordinates (TCP: Tool Center Point) using forward kinematics from the generated 
    # multi-axis position trajectories.
    x = []; y = []; x_1 = []; y_1 = []; 
    for _, theta_arr_i in enumerate(np.array(theta_arr, dtype=np.float64).T):
        # Obtain the x, y coordinates of the desired absolute positions of the joints using forward kinematics.
        p = np.round(Kinematics.Forward_Kinematics(theta_arr_i, Robot_Str)[1], tolerance).astype('float32')
        x.append(p[0]); y.append(p[1])

        # Predict the absolute joint position of the robotic arm from the input position of the end-effector 
        # and configuration of the solution.
        theta_predicted = FCNN_IK_Predictor_Cls.Predict([0.0, 0.0, 1.0])[0]

        # Obtain the x, y coordinates of the predicted absolute positions of the joints using forward kinematics.
        p_1 = np.round(Kinematics.Forward_Kinematics(theta_predicted, Robot_Str)[1], tolerance).astype('float32')
        x_1.append(p_1[0]); y_1.append(p_1[1])
        
    # Display TCP(Tool Center Point) parameters.
    y_label = [r'x(t) in meters', r'y(t) in meters']
    for i, (TPC_i, TCP_1_i) in enumerate(zip([x, y], [x_1, y_1])):
        # Create a figure.
        _, ax = plt.subplots()

        # Visualization of relevant structures.
        ax.plot(Polynomial_Cls.t, TPC_i, '.-', color='#d0d0d0', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, markerfacecolor = '#ffffff', label='Desired Data')
        ax.plot(Polynomial_Cls.t, TCP_1_i, '.-', color='#ffbf80', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, markerfacecolor = '#ffffff', label='Predicted Data')

        # Set parameters of the graph (plot).
        #   Set the x ticks.
        ax.set_xticks(np.arange(np.min(Polynomial_Cls.t) - 0.1, np.max(Polynomial_Cls.t) + 0.1, 0.1))
        #   Set the y ticks.
        ax.set_yticks(np.arange(np.min(TPC_i) - 0.1, np.max(TPC_i) + 0.1, 0.1))
        #   Label.
        ax.set_xlabel(r't in seconds', fontsize=15, labelpad=10)
        ax.set_ylabel(f'{y_label[i]}', fontsize=15, labelpad=10) 
        #   Set parameters of the visualization.
        ax.grid(which='major', linewidth = 0.15, linestyle = '--')
        # Get handles and labels for the legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        # Remove duplicate labels.
        legend = dict(zip(labels, handles))
        # Show the labels (legends) of the graph.
        ax.legend(legend.values(), legend.keys(), fontsize=10.0)

        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())
