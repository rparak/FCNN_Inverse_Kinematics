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
#   The seed of the random number generator.
CONST_SEED = 0

def main():
    """
    Description:
        The main program that generates the dataset of the selected robotic structure.

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

    # The number of data to be generated (command line argument or constant).
    num_of_data = int(sys.argv[1]) if len(sys.argv) > 1 else CONST_NUM_OF_DATA

    # Create a file path to save the data.
    file_path = f'{project_folder}/Data/Dataset/Config_N_{num_of_data}'

    # Start the timer.
    t_0 = time.time()
    print('[INFO] The generation of the dataset is in progress.')

    # Generate the dataset.
    data = Dataset.Generate(num_of_data, CONST_SEED, Robot_Str)

    # Display information (1).
    print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

    # Save the data to the file (*.zip).
    #   Note:
    #       f'{file_path}.zip' contains f'Config_N_{num_of_data}.csv'
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
