# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../' + 'src' not in sys.path:
    sys.path.append('../../../' + 'src')
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics
#   ../FCNN_IK/Model
import FCNN_IK.Model

"""
Description:
    Initialization of constants.
"""
# A dataset configuration that specifies the amount of data
# generated to train the model.
CONST_NUM_OF_DATA = 1000
# Metrics such as loss, mean squared error (MSE) and mean absolute
# error (MAE) to be plotted, and the column index of the training values.
#   Note:
#       The order of the columns is defined in ../src/FCNN_IK/Model.py -> CONST_HISTORY_KEYS
#       ['loss', 'mse', 'mae', 'val_loss', 'val_mse', 'val_mae', 'learning_rate']
CONST_METRICES = [(r'Loss', 0), (r'Mean Squared Error (MSE)', 1),
                  (r'Mean Absolute Error (MAE)', 2)]
# Save the figure to the file (*.png) for the README.
#   Note:
#       f'{project_folder}/images/Training/N_{CONST_NUM_OF_DATA}.png'
CONST_SAVE_FIGURE = False

def main():
    """
    Description:
        A program to show result data from training a dataset. Metrics such as Loss, Mean Squared Error, Mean Absolute
        Error and the learning rate were used to evaluate the performance of the proposed network.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Create a file path to read the data.
    file_path_r = f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}'

    # Read data from the file {*.txt}.
    data = File_IO.Load(f'{file_path_r}_History', 'txt', ',')
    # Get the index with the best loss (val_loss) selected by the trainer.
    (id, _) = Mathematics.Min(data[:, FCNN_IK.Model.CONST_HISTORY_KEYS.index('val_loss')])

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: Fully-Connected Neural Network (FCNN)')
    print(f'[INFO] The name of the dataset: Config_N_{CONST_NUM_OF_DATA}')
    print(f'[INFO] Number of epochs: {data.shape[0]}')
    print(f'[INFO] The best results were found in the {id} iteration.')
    for _, (metric_i, idx_i) in enumerate(CONST_METRICES):
        print(f'[INFO]  {metric_i}:')
        print(f'[INFO]  [train = {data[id, idx_i]:.08f}, valid = {data[id, idx_i + 3]:.08f}]')
    print(f'[INFO]  Learning rate: {data[id, 6]:.3e}')

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure with 4 subplots.
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'Training Results for the Dataset Containing {CONST_NUM_OF_DATA} Data Points', fontsize=25)

    # Subsample the epochs to approximately 1000 points.
    t = np.arange(0, data.shape[0], max(1, data.shape[0] // 1000))
    for i, (metric_i, idx_i) in enumerate(CONST_METRICES):
        # Visualization of relevant structures.
        ax[i].plot(t, data[t, idx_i], '-', color=[0.525,0.635,0.8,1.0], linewidth=1.0, label='train')
        ax[i].plot(t, data[t, idx_i + 3], '-', color=[1.0,0.75,0.5,1.0], linewidth=1.0, label='valid')
        ax[i].axvline(id, color='#8d8d8d', linestyle='--', linewidth=1.0, label='best')

        # Set parameters of the graph (plot).
        ax[i].set_yscale('log')
        #   Label.
        ax[i].set_xlabel(r'Epoch', fontsize=15, labelpad=10)
        ax[i].set_ylabel(metric_i, fontsize=15, labelpad=10)
        #   Set parameters of the visualization.
        ax[i].grid(which='major', linewidth = 0.15, linestyle = '--')
        # Show the labels (legends) of the graph.
        ax[i].legend(fontsize=10.0)

    # Visualization of the learning rate.
    ax[3].plot(t, data[t, 6], '-', color='#674ea7', linewidth=1.0)
    ax[3].set_yscale('log')
    ax[3].set_xlabel(r'Epoch', fontsize=15, labelpad=10)
    ax[3].set_ylabel(r'Learning Rate', fontsize=15, labelpad=10)
    ax[3].grid(which='major', linewidth = 0.15, linestyle = '--')

    # Save the figure.
    if CONST_SAVE_FIGURE == True:
        os.makedirs(f'{project_folder}/images/Training', exist_ok=True)
        fig.savefig(f'{project_folder}/images/Training/N_{CONST_NUM_OF_DATA}.png', dpi=150, bbox_inches='tight')
        print(f'[INFO] >> file_path = {project_folder}/images/Training/N_{CONST_NUM_OF_DATA}.png')

    # Show the result.
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
