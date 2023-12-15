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
# A dataset configuration that specifies the amount of data 
# generated to train the model.
CONST_NUM_OF_DATA = 1000

def main():
    """
    Description:
        A program to compare the absolute position error for an individual dataset configuration. 
        
        The comparison is tested on trajectories of the absolute positions of the robot's joints, generated 
        using a multi-axis polynomial profile.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE
    
    # The tolerance of the data.
    tolerance = 4
    
    e_p = []
    for _, N_i in enumerate([1000, 10000, 100000]):
        # Prediction of the absolute joint position of the robotic arm.
        #   1\ Initialization.
        FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/Data/Model/Config_N_{N_i}_use_val_True_Scaler_x.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}_use_val_True_Scaler_y.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}_use_val_True.h5')

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

        # Obtain the Absolute Position Error (APE) using forward kinematics from the generated 
        # multi-axis position trajectories.
        e_p_i = []
        for _, theta_arr_i in enumerate(np.array(theta_arr, dtype=np.float64).T):
            # Obtain the x, y coordinates of the desired absolute positions of the joints using forward kinematics.
            p = np.round(Kinematics.Forward_Kinematics(theta_arr_i, Robot_Str)[1], tolerance).astype('float32')

            # Predict the absolute joint position of the robotic arm from the input position of the end-effector 
            # and configuration of the solution.
            theta_predicted = FCNN_IK_Predictor_Cls.Predict([0.0, 0.0, 1.0])[0]

            # Obtain the x, y coordinates of the predicted absolute positions of the joints using forward kinematics.
            p_1 = np.round(Kinematics.Forward_Kinematics(theta_predicted, Robot_Str)[1], tolerance).astype('float32')
            
            # Obtain he absolute position error.
            e_p_i.append(Mathematics.Euclidean_Norm(p_1 - p))

        # Store the data.
        e_p.append(e_p_i)

        # Release class object.
        del FCNN_IK_Predictor_Cls

    # Set the parameters for the scientific style.
    plt.style.use('science')

    label = [r'$e_{p}(\hat{t})$']; title = ['Absolute Position Error (APE)']

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of relevant structures.
    box_plot_out = ax.boxplot(e_p, labels=['N = 1000', 'N = 10000', 'N = 100000'], showmeans=True, meanline = True, showfliers=False)
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
    ax.set_ylabel(f'Absolute error {label[0]} in millimeters', fontsize=15, labelpad=10) 
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
