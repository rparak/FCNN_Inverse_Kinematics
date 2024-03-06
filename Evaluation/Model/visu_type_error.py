# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
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
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics
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
# The configuration ID of the inverse kinematics (IK) solution.
CONST_IK_CONFIGURATION = 1

def main():
    """
    Description:
        A program to compare the absolute orientation error for an individual dataset configuration. 
        
        The comparison is tested on trajectories of both the x and y coordinates, generated 
        using a multi-axis trapezoidal profile.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE
    
    # The tolerance of the data.
    tolerance = 4
    
    # Obtain the constraints for absolute joint positions in order to generate multi-axis position trajectories.
    (abs_j_pos_0, abs_j_pos_1) = Configuration.Parameters.Get_Absolute_Joint_Positions(Robot_Str.Name)

    # Obtain the x, y coordinates of the absolute positions of the joints using forward kinematics.
    #   p_0: Initial coordinates.
    #   p_1: Desired coordinates.
    p_0 = Kinematics.Forward_Kinematics(abs_j_pos_0, Robot_Str)[1].astype('float32')
    p_1 = Kinematics.Forward_Kinematics(abs_j_pos_1, Robot_Str)[1].astype('float32')

    # Initialization of the class to generate trajectory.
    Trapezoidal_Cls = Trajectory.Utilities.Trapezoidal_Profile_Cls(delta_time=0.01)

    # Generation of multi-axis position trajectories from input parameters.
    p_arr = []
    for _, (p_0_i, p_1_i) in enumerate(zip(p_0, p_1)):
        (p_arr_i, _, _) = Trapezoidal_Cls.Generate(p_0_i, p_1_i, 0.0, 0.0,
                                                Configuration.Parameters.CONST_T_0, Configuration.Parameters.CONST_T_1)
        p_arr.append(p_arr_i)

    e_o = []
    for _, N_i in enumerate([1000, 10000, 100000]):
        # Prediction of the absolute joint position of the robotic arm.
        #   1\ Initialization.
        FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/Data/Model/Config_N_{N_i}_Scaler_x.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}_Scaler_y.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}.h5')
        

        # Obtain the Absolute Orientation Error (APE) using inverse kinematics from the generated 
        # multi-axis position trajectories.
        e_o_i = []
        for _, p_arr_i in enumerate(np.array(p_arr, dtype=np.float32).T):
            p_tmp = np.round(p_arr_i.astype('float32'), tolerance)

            # Compute the solution of the inverse kinematics (IK) using an analytical method.
            (_, theta) = Kinematics.Inverse_Kinematics(p_tmp, Robot_Str)

            # Predict the absolute joint position of the robotic arm from the input position of the end-effector 
            # and configuration of the solution.
            theta_predicted = FCNN_IK_Predictor_Cls.Predict(np.array([p_tmp[0], p_tmp[1], CONST_IK_CONFIGURATION], dtype=np.float32))[0]

            # Obtain the absolute orientation error.
            e_o_i.append(Mathematics.Euclidean_Norm(theta_predicted - theta[CONST_IK_CONFIGURATION, :]))

        # Store the data.
        e_o.append(e_o_i)

        # Release class object.
        del FCNN_IK_Predictor_Cls

    # Set the parameters for the scientific style.
    plt.style.use('science')

    label = [r'$e(\hat{t})$']; title = ['Comparison of Absolute Orientation Error (AOE) in Individual Datasets']

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of relevant structures.
    box_plot_out = ax.boxplot(e_o, labels=['N = 1000', 'N = 10000', 'N = 100000'], showmeans=True, meanline = True, showfliers=False)
    #   Auxiliary structures.
    ax.plot([], [], '-', linewidth=1, color='#8ca8c5', label='Mean')
    ax.plot([], [], '-', linewidth=1, color='#ffbf80', label='Median')

    # Set the properties of the box plot.
    plt.setp(box_plot_out['boxes'], color='#8d8d8d',)
    plt.setp(box_plot_out['whiskers'], color='#8d8d8d')
    plt.setp(box_plot_out['fliers'], color='#8d8d8d')
    plt.setp(box_plot_out['means'], color='#8ca8c5')
    plt.setp(box_plot_out['medians'], color='#ffbf80')
    plt.setp(box_plot_out['caps'], color='#8d8d8d')

    # Set parameters of the graph (plot).
    ax.set_title(f'{title[0]}', fontsize=25, pad=25.0)
    #   Label
    ax.set_xlabel(r'A dataset configuration that specifies the amount of data generated to train the model', fontsize=15, labelpad=10)
    ax.set_ylabel(f'Error {label[0]} in radians', fontsize=15, labelpad=10) 
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
