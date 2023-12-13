# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../' not in sys.path:
    sys.path.append('../../../')
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# A dataset configuration that specifies the amount of data 
# generated to train the model.
CONST_NUM_OF_DATA = 10000

def main():
    """
    Description:
        A program to show result data from training a dataset. Metrics such as Mean Squared Error, Accuracy, 
        Mean Absolute Error, etc. were used to evaluate the performance of the proposed network.
    """
        
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to read/write the data.
    file_path_r = f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}'
    file_path_w = f'{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}'

    # Read data from the file {*.txt}.
    data = File_IO.Load(f'{file_path_r}_use_val_True_History', 'txt', ',')
    # Get the index with the best loss (val_mse) selected by the trainer.
    (id, _) = Mathematics.Min(data[:, 3])

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: Fully-Connected Neural Network (FCNN)')
    print(f'[INFO] The name of the dataset: {file_path_w}')
    print(f'[INFO] The best results were found in the {id} iteration.')
    print('[INFO]  Accuracy:')
    print(f'[INFO]  [train = {data[id, 1]:.08f}, valid = {data[id, 5]:.08f}]')
    print('[INFO]  Mean Squared Error (MSE):')
    print(f'[INFO]  [train = {data[id, 2]:.08f}, valid = {data[id, 6]:.08f}]')
    print('[INFO]  Mean Absolute Error (MAE):')
    print(f'[INFO]  [train = {data[id, 3]:.08f}, valid = {data[id, 7]:.08f}]')   
    
    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    _, ax = plt.subplots()

    # Visualization of relevant structures.
    ax.plot(np.arange(0,len(data[:, 0])), data[:, 2], '-', color=[0.525,0.635,0.8,0.5], linewidth=1.0, label='train')
    ax.plot(np.arange(0,len(data[:, 3])), data[:, 6], '-', color=[1.0,0.75,0.5,0.5], linewidth=1.0, label='valid')

    # Set parameters of the graph (plot).
    ax.set_title(f'The name of the Dataset: Config_N_{CONST_NUM_OF_DATA}', fontsize=25, pad=25.0)
    #   Label.
    ax.set_xlabel(r'Epoch', fontsize=15, labelpad=10)
    ax.set_ylabel(r'Mean Squared Error (MSE)', fontsize=15, labelpad=10) 
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
    
if __name__ == '__main__':
    sys.exit(main())
