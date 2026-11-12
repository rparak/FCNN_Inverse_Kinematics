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
#   ../Kinematics/Core
import Kinematics.Core
#   ../FCNN_IK/Model
import FCNN_IK.Model
#   ../Utilities/Dataset
import Utilities.Dataset as Dataset

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# The models (dataset configurations) to be visualized.
CONST_NUM_OF_DATA = [1000, 10000, 100000]
# The name of the fixed test set (without extension).
CONST_TEST_SET_NAME = 'Test_N_10000'
# Number of test samples (the first rows of the test set) to be visualized.
CONST_NUM_OF_SAMPLES = 500
# Save the figures to the files (*.png) for the README.
#   Note:
#       f'{project_folder}/images/Prediction_NN_Type_{i}.png'
CONST_SAVE_FIGURE = False

def main():
    """
    Description:
        The program to visualize both the desired and predicted coordinates of the robot's end-effector
        using an inverse kinematics neural-network predictor.

        The comparison is tested on the first samples of the fixed test set (both configurations of the
        solution).
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Read the fixed test set from a file (*.zip).
    data = Dataset.Load(f'{project_folder}/Data/Dataset/{CONST_TEST_SET_NAME}').values[0:CONST_NUM_OF_SAMPLES]
    p = data[:, 0:2]

    # Set the parameters for the scientific style.
    plt.style.use('science')

    for i, (N_i, color_i) in enumerate(zip(CONST_NUM_OF_DATA, ['#3d85c6', '#6aa84f', '#674ea7'])):
        file_path = f'{project_folder}/Data/Model/Config_N_{N_i}'
        if not os.path.isfile(f'{file_path}.keras'):
            print(f'[WARNING] The model {file_path}.keras does not exist. The model is skipped.')
            continue

        # Prediction of the absolute joint position of the robotic arm.
        #   1\ Initialization.
        FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{file_path}_Scaler_x.pkl', f'{file_path}_Scaler_y.pkl',
                                                                 f'{file_path}.keras', Robot_Str)
        #   2\ Prediction (batch).
        (_, theta_predicted) = FCNN_IK_Predictor_Cls.Predict(data[:, 0:3])

        # Obtain the predicted end-effector coordinates and the absolute position error in millimeters.
        p_predicted = Kinematics.Core.Forward_Kinematics(theta_predicted, Robot_Str)[1]
        e_p = 1000.0 * np.linalg.norm(p - p_predicted, axis=1)

        # Release class object.
        del FCNN_IK_Predictor_Cls

        # Create a figure.
        figure, ax = plt.subplots(figsize=(8, 8))

        # Visualization of the straight line between the desired and predicted coordinates of the robot's end effector.
        for _, (d_i, p_i) in enumerate(zip(p, p_predicted)):
            ax.plot([d_i[0], p_i[0]], [d_i[1], p_i[1]], '-', color='#e06666', linewidth=0.8, label=r'$\Delta$e')

        # Visualization of relevant structures.
        ax.plot(p[:, 0], p[:, 1], 'o', color='#8d8d8d', markersize=5.0, markeredgewidth=1.0, markerfacecolor='#ffffff',
                label=f'Desired Coordinates: N = {p.shape[0]}')
        ax.plot(p_predicted[:, 0], p_predicted[:, 1], 'o', color=color_i, markersize=3.0, markeredgewidth=1.0, markerfacecolor=color_i,
                label=f'Predicted Coordinates: mean $e_p$ = {np.mean(e_p):.2f} mm, max $e_p$ = {np.max(e_p):.2f} mm')

        # Set parameters of the graph (plot).
        ax.set_title(f'Predicted End-Effector Coordinates: Neural-Network Type {i} (N = {N_i})', fontsize=12, pad=10.0)
        #   Label.
        ax.set_xlabel(r'x-axis in meters', fontsize=12, labelpad=8)
        ax.set_ylabel(r'y-axis in meters', fontsize=12, labelpad=8)
        #   Set parameters of the visualization.
        ax.set_aspect('equal')
        ax.grid(which='major', linewidth = 0.15, linestyle = '--')
        # Get handles and labels for the legend.
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels.
        legend = dict(zip(labels, handles))
        # Show the labels (legends) of the graph.
        ax.legend(legend.values(), legend.keys(), fontsize=9.0, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)

        # Save the figure.
        if CONST_SAVE_FIGURE == True:
            figure.savefig(f'{project_folder}/images/Prediction_NN_Type_{i}.png', dpi=150, bbox_inches='tight')
            print(f'[INFO] >> file_path = {project_folder}/images/Prediction_NN_Type_{i}.png')

        # Show the result.
        plt.show()

if __name__ == "__main__":
    sys.exit(main())
