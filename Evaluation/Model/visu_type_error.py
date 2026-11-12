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
# The number of iterations of the hybrid refinement (NN + refinement).
#   Note:
#       0 -> pure neural-network prediction.
CONST_REFINE_ITERATIONS = 0
# Save the figure to the file (*.png) for the README.
#   Note:
#       f'{project_folder}/images/Error.png' or f'{project_folder}/images/Error_Refined.png'
CONST_SAVE_FIGURE = False

def main():
    """
    Description:
        A program to visualize the distribution of the absolute position error (APE) of the individual models
        on the fixed test set (both configurations of the solution).

        The box shows the interquartile range with the median, the whiskers show the 5th and 95th percentiles,
        and the points show the samples outside the whiskers.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Read the fixed test set from a file (*.zip).
    data = Dataset.Load(f'{project_folder}/Data/Dataset/{CONST_TEST_SET_NAME}').values

    e_p = []; labels = []
    for _, N_i in enumerate(CONST_NUM_OF_DATA):
        file_path = f'{project_folder}/Data/Model/Config_N_{N_i}'
        if not os.path.isfile(f'{file_path}.keras'):
            print(f'[WARNING] The model {file_path}.keras does not exist. The model is skipped.')
            continue

        # Prediction of the absolute joint position of the robotic arm.
        #   1\ Initialization.
        FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{file_path}_Scaler_x.pkl', f'{file_path}_Scaler_y.pkl',
                                                                 f'{file_path}.keras', Robot_Str)
        #   2\ Prediction (batch).
        (_, theta_predicted) = FCNN_IK_Predictor_Cls.Predict(data[:, 0:3], refine_iterations=CONST_REFINE_ITERATIONS)

        # Obtain the absolute position error in millimeters.
        e_p_i = 1000.0 * np.linalg.norm(data[:, 0:2] - Kinematics.Core.Forward_Kinematics(theta_predicted, Robot_Str)[1], axis=1)
        e_p.append(e_p_i); labels.append(f'N = {N_i}')
        print(f'[INFO] N = {N_i}: e_p mean = {np.mean(e_p_i):.4g} mm, median = {np.median(e_p_i):.4g} mm, '
              f'p95 = {np.percentile(e_p_i, 95):.4g} mm, max = {np.max(e_p_i):.4g} mm')

        # Release class object.
        del FCNN_IK_Predictor_Cls

    if e_p == []:
        print('[ERROR] No model was found.')
        return 1

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    figure, ax = plt.subplots(figsize=(8, 5))

    # Visualization of relevant structures.
    color = ['#3d85c6', '#6aa84f', '#674ea7']
    box = ax.boxplot(e_p, whis=(5, 95), widths=0.5, patch_artist=True,
                     medianprops=dict(color='#000000', linewidth=1.5),
                     flierprops=dict(marker='.', markersize=2.0, markerfacecolor='#8d8d8d', markeredgecolor='#8d8d8d', alpha=0.5))
    ax.set_xticks(np.arange(1, len(labels) + 1)); ax.set_xticklabels(labels)
    for i, patch_i in enumerate(box['boxes']):
        patch_i.set_facecolor(color[i % len(color)]); patch_i.set_alpha(0.6)
    # Mean of the position error.
    ax.plot(np.arange(1, len(e_p) + 1), [np.mean(e_p_i) for e_p_i in e_p], 'D', color='#e06666', markersize=6.0, label='Mean')

    # Set parameters of the graph (plot).
    method = 'NN' if CONST_REFINE_ITERATIONS == 0 else f'NN + refinement ({CONST_REFINE_ITERATIONS} it.)'
    ax.set_title(f'Absolute Position Error on the Test Set ({data.shape[0]} samples, both configurations): {method}', fontsize=12, pad=10.0)
    ax.set_yscale('log')
    #   Label.
    ax.set_xlabel(r'Model (training dataset size)', fontsize=12, labelpad=8)
    ax.set_ylabel(r'$e_{p}$ in millimeters (log scale)', fontsize=12, labelpad=8)
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.15, linestyle = '--')
    # Show the labels (legends) of the graph.
    ax.legend(fontsize=10.0, loc='upper right')

    # Save the figure.
    if CONST_SAVE_FIGURE == True:
        file_name = 'Error' if CONST_REFINE_ITERATIONS == 0 else 'Error_Refined'
        figure.savefig(f'{project_folder}/images/{file_name}.png', dpi=150, bbox_inches='tight')
        print(f'[INFO] >> file_path = {project_folder}/images/{file_name}.png')

    # Show the result.
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
