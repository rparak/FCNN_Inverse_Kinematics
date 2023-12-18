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
# Integrate a system of ordinary differential equations (ODE) [pip3 install scipy]
import scipy 
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# A dataset configuration that specifies the amount of data 
# generated to train the model.
CONST_NUM_OF_DATA = 1000
# Matrices such as mean squared error (MSE), mean absolute 
# error (MAE) and accuracy to be plotted.
CONST_METRICES = [r'Accuracy', r'Mean Squared Error (MSE)', 
                  r'Mean Absolute Error (MAE)']

def main():
    """
    Description:
        A program to show result data from training a dataset. Metrics such as Mean Squared Error, Accuracy, 
        Mean Absolute Error, etc. were used to evaluate the performance of the proposed network.
    """
        
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Create a file path to read the data.
    file_path_r = f'{project_folder}/Data/Model/Config_N_{CONST_NUM_OF_DATA}'

    # Read data from the file {*.txt}.
    data = File_IO.Load(f'{file_path_r}_History', 'txt', ',')
    # Get the index with the best loss (val_mse) selected by the trainer.
    (id, _) = Mathematics.Min(data[:, 6])

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: Fully-Connected Neural Network (FCNN)')
    print(f'[INFO] The name of the dataset: Config_N_{CONST_NUM_OF_DATA}')
    print(f'[INFO] The best results were found in the {id} iteration.')
    print('[INFO]  Accuracy:')
    print(f'[INFO]  [train = {data[id, 1]:.08f}, valid = {data[id, 5]:.08f}]')
    print('[INFO]  Mean Squared Error (MSE):')
    print(f'[INFO]  [train = {data[id, 2]:.08f}, valid = {data[id, 6]:.08f}]')
    print('[INFO]  Mean Absolute Error (MAE):')
    print(f'[INFO]  [train = {data[id, 3]:.08f}, valid = {data[id, 7]:.08f}]')   
    
    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure with 3 subplots.
    fig, ax = plt.subplots(1, 3)
    fig.suptitle(f'Training Results for the Dataset Containing {CONST_NUM_OF_DATA} Data Points', fontsize=25)

    t = np.arange(0, len(data[:, 0]), 100)
    for i, mectric_i in enumerate(CONST_METRICES):
        # Interpolate a 1-D function.
        f_1 = scipy.interpolate.interp1d(np.arange(0,len(data[:, 0])), data[:, i + 1])
        f_2 = scipy.interpolate.interp1d(np.arange(0,len(data[:, 0])), data[:, i + 5])

        # Approximation of the function: y = f(x).
        y_1 = f_1(t)
        y_2 = f_2(t)

        # Visualization of relevant structures.
        ax[i].plot(t, y_1, '-', color=[0.525,0.635,0.8,0.5], linewidth=1.0, label='train')
        ax[i].plot(t, y_2, '-', color=[1.0,0.75,0.5,0.5], linewidth=1.0, label='valid')

        # Set parameters of the graph (plot).
        #   Label.
        ax[i].set_xlabel(r'Epoch', fontsize=15, labelpad=10)
        ax[i].set_ylabel(mectric_i, fontsize=15, labelpad=10) 
        #   Set parameters of the visualization.
        ax[i].grid(which='major', linewidth = 0.15, linestyle = '--')
        # Show the labels (legends) of the graph.
        ax[i].legend(fontsize=10.0)

    # Show the result.
    plt.show()
    
if __name__ == '__main__':
    sys.exit(main())
