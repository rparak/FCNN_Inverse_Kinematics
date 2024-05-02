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
#   ../Lib/Utilities/File_IO
import Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000

def main():
    """
    Description:
        A program to visualize the workspace (2D positions in x, y coordinates) of a defined dataset 
        for a selected robotic structure.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a file path to read the data.
    file_path = f'{project_folder}/Data/Dataset/Config_N_{CONST_NUM_OF_DATA}'
    
    # Read the data from a file.
    data = File_IO.Load(file_path, 'pkl', ',')

    # The tolerance of the data.
    tolerance = 4

    # Express the data as x, y coordinates in meters.
    x = np.round(data[:, 0], tolerance); y = np.round(data[:, 1], tolerance)

    # Create a figure.
    figure = plt.figure()
    figure.tight_layout()
    ax = figure.add_subplot()

    # Plot the robot's dataset dependent on the input data from the file.
    ax.plot(x, y, 'o', linewidth=1, markersize=2.5, color = [0,0.9,0.3,1.0], label=f'2D Coordinates (x, y): N = {CONST_NUM_OF_DATA}')

    # Set parameters of the graph (plot).
    ax.set_title(f'The Dataset of a {Robot_Str.Theta.Zero.size}-axis robotic arm {Robot_Str.Name}: Work Envelope', fontsize=45, pad=25.0)
    # Set parameters of the graph (plot).
    #   Set the x ticks.
    ax.set_xticks(np.arange(np.min(x) - 0.1, np.max(x) + 0.1, 0.1))
    #   Set the y ticks.
    ax.set_yticks(np.arange(np.min(y) - 0.1, np.max(y) + 0.1, 0.1))
    #   Label.
    ax.set_xlabel(r'x-axis in meters', fontsize=25, labelpad=10)
    ax.set_ylabel(r'y-axis in meters', fontsize=25, labelpad=10) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    # Show the result.
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
