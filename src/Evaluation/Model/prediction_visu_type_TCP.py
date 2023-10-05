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

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Lib.Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.01)
    
    # Set the parameters for the scientific style.
    plt.style.use(['science'])

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

    # Display TCP(Tool Center Point) parameters.
    y_label = [r'x(t) in meters', r'y(t) in meters', r'z(t) in meters', r'$q_{w}(t)$ in [-]', 
               r'$q_{x}(t)$ in [-]', r'$q_{y}(t)$ in [-]', r'$q_{z}(t)$ in [-]']
    for i, TPC_i in enumerate([x, y, z, q_w, q_x, q_y, q_z]):
        # Create a figure.
        _, ax = plt.subplots()

        # Visualization of relevant structures.
        ax.plot(Polynomial_Cls.t, TPC_i, '.-', color='#d0d0d0', linewidth=1.0, markersize = 3.0, 
                markeredgewidth = 1.5, markerfacecolor = '#ffffff', label='Desired Data')

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

        if CONST_SAVE_DATA == True:
            # Set the full scree mode.
            plt.get_current_fig_manager().full_screen_toggle()

            # Save the results.
            plt.savefig(f'{project_folder}/src/Data/Prediction/{Robot_Str.Name}/TCP_{i}_Config_N_{CONST_NUM_OF_DATA}.png', format='png', dpi=300)
        else:
            # Show the result.
            plt.show()

if __name__ == "__main__":
    sys.exit(main())
