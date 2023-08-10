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

def main():
    """
    Description:
        A program for visualization the 3D positions (x, y, z) of a dataset of an individual robot structure.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a file path to save the data.
    file_path = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_1/Config_N_{CONST_NUM_OF_DATA}_ID_0"
    
    # Read the data from a file.
    data = File_IO.Load(file_path, 'pkl', ',')

    # Express the data as x, y, z.
    x = data[:, 0]; y = data[:, 1]; z = data[:, 2]

    # Create a figure.
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')

    # Plot the robot's dataset dependent on the input data from the file.
    ax.plot(np.round(x, 4), np.round(y, 4), np.round(z, 4), 'o', linewidth=1, markersize=2.5, color = [0,0.9,0.3,1.0],
            label=f'3D positions (x, y, z): N = {CONST_NUM_OF_DATA}')

    # Set parameters of the graph (plot).
    ax.set_title(f'The Dataset of a {Robot_Str.Theta.Zero.size}-axis robotic arm {Robot_Str.Name}', fontsize=25, pad=25.0)
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
