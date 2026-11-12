# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Custom Lib.:
#   ../Parameters/Robot
import Parameters.Robot
#   ../Utilities/Dataset
import Utilities.Dataset as Dataset

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# Dataset configuration.
#   Number of data (samples/rows) to be generated.
#       Note:
#           The number can also be passed as the first command line argument,
#           e.g. $ python generate.py 10000
CONST_NUM_OF_DATA = 100000
#   The type of the dataset ('train' or 'test') with the seed of the random number generator
#   and the prefix of the file name.
#       Note 1:
#           The type can also be passed as the second command line argument,
#           e.g. $ python generate.py 10000 test
#       Note 2:
#           The seed of the test set must be different from the seed of the training datasets.
CONST_DATASET_TYPE = 'train'
CONST_DATASET_TYPES = {'train': {'seed': 0, 'prefix': 'Config'},
                       'test':  {'seed': 1, 'prefix': 'Test'}}

def main():
    """
    Description:
        The main program that generates the dataset of the selected robotic structure, either
        a training dataset (Config_N_{N}.zip, seed 0) or the fixed test set (Test_N_{N}.zip, seed 1).

            $ python generate.py [N] [train|test]

        The test set has the same format and rules as the training datasets, but it is generated
        with a different seed. It is shared by all models, so the evaluation results are comparable.

        The structure of the dataset is described below.
            Input of the NN:  x -> Position(x, y); configuration_id(0, 1)
            Output of the NN: y -> theta(0 .. n)

            Where n is the number of absolute joint positions.

        Note 1:
            The positions are sampled uniformly in the Cartesian workspace. Only IK solutions within
            the joint limits are stored. For more information, see the function below:
                ../src/Utilities/Dataset.py -> Generate()

        Note 2:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # The number of data and the type of the dataset (command line arguments or constants).
    num_of_data = int(sys.argv[1]) if len(sys.argv) > 1 else CONST_NUM_OF_DATA
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else CONST_DATASET_TYPE
    if dataset_type not in CONST_DATASET_TYPES:
        print(f'[ERROR] Unknown dataset type \'{dataset_type}\'. Use one of {list(CONST_DATASET_TYPES)}.')
        return 1
    seed = CONST_DATASET_TYPES[dataset_type]['seed']; file_name = f'{CONST_DATASET_TYPES[dataset_type]["prefix"]}_N_{num_of_data}'

    # Create a file path to save the data.
    file_path = f'{project_folder}/Data/Dataset/{file_name}'

    # Start the timer.
    t_0 = time.time()
    print(f'[INFO] The generation of the dataset ({dataset_type}, seed = {seed}) is in progress.')

    # Generate the dataset.
    data = Dataset.Generate(num_of_data, seed, Robot_Str)

    # Display information (1).
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

    # Save the data to the file (*.zip).
    #   Note:
    #       f'{file_path}.zip' contains f'{file_name}.csv'
    Dataset.Save(file_path, data)
    print(f'[INFO] The file has been successfully saved.')
    print(f'[INFO] >> file_path = {file_path}.zip')

    # Display information (2).
    print(f'[INFO] Number of processed data: {len(data)}')
    if len(data) == num_of_data:
        print('[INFO] The data generation has been successfully completed.')
    else:
        print(f'[WARNING] Insufficient number of combinations.')

if __name__ == "__main__":
    sys.exit(main())
