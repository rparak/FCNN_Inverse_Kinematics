# System (Default)
import sys
#   Add access if it is not in the system path.
if '..' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Sklearn (Simple and efficient tools for predictive data analysis) [pip3 install scikit-learn]
import sklearn.model_selection
# Joblib (Lightweight pipelining) [pip3 install joblib]
import joblib
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO
#   ../Lib/DNN_IK/Utilities
import Lib.DNN_IK.Utilities as Utilities
#   ../Lib/DNN_IK/Parameters
import Lib.DNN_IK.Parameters
#   ../Lib/DNN_IK/Model
import Lib.DNN_IK.Model

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000
#   Type of the dataset.
CONST_DATASET_TYPE = 1
#   The ID of the dataset in the selected type.
CONST_DATASET_ID = 0

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to read the data.
    file_path = f'{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_TYPE}/Config_N_{CONST_NUM_OF_DATA}_ID_{CONST_DATASET_ID}'
    
    # Read the data from a file.
    data = File_IO.Load(file_path, 'pkl', ',')

    # Assign data to variables.
    if CONST_DATASET_TYPE == 1:
        #   Input of the NN:  x -> Position(x, y, z); Orientation(quaternion)
        #   Output of the NN: y -> theta(0 .. n)
        x = data[:, 0:7].astype('float32'); y = data[:, -Robot_Str.Theta.Zero.size:].astype('float32')
    else:
        if CONST_DATASET_ID == 0:
            # Input of the NN:  x -> Position(x, y, z), Orientation(quaternion)
            # Output of the NN: y -> theta(0 .. n - 1) 
            x = data[:, 0:7].astype('float32'); y = data[:, 7:-1].astype('float32')
        else:
            # Input of the NN:  x -> Position(x, y, z), Orientation(quaternion), theta(0 .. n - 1)
            # Output of the NN: y -> theta(n)
            x = data[:, 0:-1].astype('float32')
            y = data[:, -1].astype('float32').reshape(-1, 1)

    # Split the data from the dataset (x, y) into random train and test subsets.
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, train_size=0.9, test_size=0.1, 
                                                                                random_state=0)

    # Find the scale parameter from the dataset and transform the data using this parameter.
    scaler_x, x_train_scaled = Utilities.Scale_Data([-1.0, 1.0], x_train)
    scaler_y, y_train_scaled = Utilities.Scale_Data([-1.0, 1.0], y_train)

    # Transform of data using an the scale parameter.
    x_test_transformed = Utilities.Transform_Data_With_Scaler(scaler_x, x_test)
    y_test_transformed = Utilities.Transform_Data_With_Scaler(scaler_y, y_test)

    # Save the scaler parameter for input/output data.
    #joblib.dump(scaler_x, f'{project_folder}/src/Data/Scaler/{Robot_Str.Name}/Type_{CONST_DATASET_TYPE}/Config_N_{CONST_NUM_OF_DATA}_ID_{CONST_DATASET_ID}_scaler_x.pkl')
    #joblib.dump(scaler_y, f'{project_folder}/src/Data/Scaler/{Robot_Str.Name}/Type_{CONST_DATASET_TYPE}/Config_N_{CONST_NUM_OF_DATA}_ID_{CONST_DATASET_ID}_scaler_y.pkl')

    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32,input_shape=(7,)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(150))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(75))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dropout(0.05))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Activation('tanh'))

    # Generate network
    opt = tf.keras.optimizers.experimental.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-06)
    model.compile(optimizer=opt, loss='mse')
    model.fit(x_train_scaled, y_train_scaled, epochs=10000, batch_size=128, validation_data=(x_test_transformed, y_test_transformed))

if __name__ == "__main__":
    sys.exit(main())
