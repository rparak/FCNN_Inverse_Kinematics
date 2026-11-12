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
CONST_NUM_OF_DATA = 10000
#   The seed of the random number generator.
#       Note:
#           The seed must be different from the seed used to generate
#           the training datasets (../Dataset_Generation/generate.py).
CONST_SEED = 1

def main():
    """
    Description:
        The main program that generates the fixed test set of the selected robotic structure.

        The test set has the same format and rules as the training datasets, but it is generated
        with a different seed. It is shared by all models, so the evaluation results are comparable.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to save the data.
    file_path = f'{project_folder}/Data/Dataset/Test_N_{CONST_NUM_OF_DATA}'

    # Start the timer.
    t_0 = time.time()
    print('[INFO] The generation of the test set is in progress.')

    # Generate the test set.
    data = Dataset.Generate(CONST_NUM_OF_DATA, CONST_SEED, Robot_Str)

    # Display information (1).
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

    # Save the data to the file (*.zip).
    #   Note:
    #       f'{file_path}.zip' contains f'Test_N_{CONST_NUM_OF_DATA}.csv'
    Dataset.Save(file_path, data)
    print(f'[INFO] The file has been successfully saved.')
    print(f'[INFO] >> file_path = {file_path}.zip')

    # Display information (2).
    print(f'[INFO] Number of processed data: {len(data)}')
    if len(data) == CONST_NUM_OF_DATA:
        print('[INFO] The test set generation has been successfully completed.')
    else:
        print(f'[WARNING] Insufficient number of combinations.')

if __name__ == "__main__":
    sys.exit(main())
