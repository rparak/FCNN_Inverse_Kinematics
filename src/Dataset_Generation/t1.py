# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Library:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO


"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# ....
CONST_DATASET_CONFIG = {'Type': 1, 'Number_of_Data': 100}

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to save the data.
    file_path = f"{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_CONFIG['Type']}/Config_N_{CONST_DATASET_CONFIG['Number_of_Data']}_ID_0"

    # Save the data to the file.
    data = File_IO.Load(file_path, 'pkl', ',')

    import sklearn.preprocessing
    import sklearn.model_selection
    # Joblib (Lightweight pipelining) [pip3 install joblib]
    import joblib
    
    x = np.array(data, dtype=np.float32)[:, 0:7]
    y = np.array(data, dtype=np.float32)[:, 7::]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size = 0.7, test_size=0.3, random_state=0)

    print(x_train[0])

    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0))
    data_tmp = scaler.fit_transform(x_train)

    print(x_train[0])


if __name__ == "__main__":
    sys.exit(main())

