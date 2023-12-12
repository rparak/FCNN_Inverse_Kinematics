# System (Default)
import sys
#   Add access if it is not in the system path.
if '../..' + 'src' not in sys.path:
    sys.path.append('../../..')
# OS (Operating system interfaces)
import os
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
# Save the data to a file.
CONST_SAVE_DATA = False
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000
#   Method to be used for training.
#       Method 0: No test (validation) partition.
#       Method 1: With test (validation) partition.
CONST_DATASET_METHOD = 0

def main():
    """
    Description:
        A program to save result data from training a dataset. Metrics such as Mean Squared Error, Accuracy, 
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
    if CONST_DATASET_METHOD == 0:
        data = File_IO.Load(f'{file_path_r}_use_val_False_History', 'txt', ',')
        # Get the index with the best loss (mse) selected by the trainer.
        (id, _) = Mathematics.Min(data[:, 0])
    elif CONST_DATASET_METHOD == 1:
        data = File_IO.Load(f'{file_path_r}_use_val_True_History', 'txt', ',')
        # Get the index with the best loss (val_mse) selected by the trainer.
        (id, _) = Mathematics.Min(data[:, 3])

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: Fully-Connected Neural Network (FCNN)')
    print(f'[INFO] The name of the dataset: {file_path_w}')
    print(f'[INFO] The best results were found in the {id} iteration.')
    if CONST_DATASET_METHOD == 0:
        print('[INFO]  Accuracy:')
        print(f'[INFO]  [train = {data[id, 1]:.08f}]')
        print('[INFO]  Mean Squared Error (MSE):')
        print(f'[INFO]  [train = {data[id, 2]:.08f}]')
        print('[INFO]  Mean Absolute Error (MAE):')
        print(f'[INFO]  [train = {data[id, 3]:.08f}]')
    elif CONST_DATASET_METHOD == 1:
        print('[INFO]  Accuracy:')
        print(f'[INFO]  [train = {data[id, 1]:.08f}, valid = {data[id, 5]:.08f}]')
        print('[INFO]  Mean Squared Error (MSE):')
        print(f'[INFO]  [train = {data[id, 2]:.08f}, valid = {data[id, 6]:.08f}]')
        print('[INFO]  Mean Absolute Error (MAE):')
        print(f'[INFO]  [train = {data[id, 3]:.08f}, valid = {data[id, 7]:.08f}]')   
    
if __name__ == '__main__':
    sys.exit(main())
