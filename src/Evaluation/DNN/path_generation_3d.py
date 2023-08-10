# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# Interpolation stop(t_0), start(t_1) time in seconds.
CONST_T_0 = 0.0
CONST_T_1 = 1.0
# FPS (Frames Per Seconds) value.
CONST_FPS = 30
# Absolute joint position start(id: 0) and stop(id: 1).
CONST_ABS_J_POS_0 = np.array([0.0, 0.0, 0.0, 0.0], dtype = np.float32)
CONST_ABS_J_POS_1 = np.array([Mathematics.Degree_To_Radian(90.0), Mathematics.Degree_To_Radian(0.0), 0.10, Mathematics.Degree_To_Radian(45.0)],
                              dtype = np.float32)

def main():
    """
    Description:
        ...
    """

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Obtain a linear interpolation between the actual and desired absolute joint position.
    theta_arr = np.linspace(CONST_ABS_J_POS_0, CONST_ABS_J_POS_1, np.int32((CONST_T_1 - CONST_T_0) * CONST_FPS))

    x = []; y = []; z = []
    for _, theta_arr_i in enumerate(theta_arr):
        # Obtain the homogeneous transformation matrix using forward kinematics.
        T = Kinematics.Forward_Kinematics(theta_arr_i, 'Fast', Robot_Str)[1]
    
        # Store the acquired data.
        x.append(T.p.x); y.append(T.p.y); z.append(T.p.z)

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # ...
    ax.plot(np.round(x, 4), np.round(y, 4), np.round(z, 4), '--o', color='#d0d0d0', linewidth=1.0, markersize = 3.0, 
            markeredgewidth = 1.5, markerfacecolor = '#ffffff', label='...')

    # Set parameters of the graph (plot).
    ax.set_title(f'Title ...', fontsize=25, pad=25.0)
    #   Limits.
    ax.set_xlim(np.minimum.reduce(x) - 0.1, np.maximum.reduce(x) + 0.1)
    ax.xaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_ylim(np.minimum.reduce(y) - 0.1, np.maximum.reduce(y) + 0.1)
    ax.yaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    ax.set_zlim(np.minimum.reduce(z) - 0.1, np.maximum.reduce(z) + 0.1)
    ax.zaxis.pane.set_color((1.0, 1.0, 1.0, 1.0))
    #   Label.
    ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10); ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
    ax.set_zlabel(r'z-axis in meters', fontsize=15, labelpad=10) 
    #   Set parameters of the visualization.
    ax.xaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.yaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    ax.zaxis._axinfo['grid'].update({'linewidth': 0.15, 'linestyle': '--'})
    #   Set the Axes box aspect.
    ax.set_box_aspect(None, zoom=0.90)
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
