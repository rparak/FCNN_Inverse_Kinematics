# System (Default)
import sys
#   Add access if it is not in the system path.
if '..' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO
#   ../Lib/FCNN_IK/Model
import Lib.FCNN_IK.Model
#   ../Parameters/Optimizer
import Parameters.Optimizer

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Lib.Parameters.Robot.EPSON_LS3_B401S_Str
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 1000
#   Type of the dataset.
CONST_DATASET_TYPE = 0
#   The ID of the dataset in the selected type.
CONST_DATASET_ID = 0

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Create a file path to read/write the data.
    file_path_r = f'{project_folder}/src/Data/Dataset/{Robot_Str.Name}/Type_{CONST_DATASET_TYPE}/Config_N_{CONST_NUM_OF_DATA}_ID_{CONST_DATASET_ID}'
    file_path_w = f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Type_{CONST_DATASET_TYPE}/Config_N_{CONST_NUM_OF_DATA}_ID_{CONST_DATASET_ID}'

    # Read the data from a file.
    data = File_IO.Load(file_path_r, 'pkl', ',')

    # Assign data to variables.
    if CONST_DATASET_TYPE == 0:
        #   Input of the NN:  x -> Position(x, y, z); Orientation(quaternion)
        #   Output of the NN: y -> theta(0 .. n)
        x = data[:, 0:7].astype('float32'); y = data[:, -Robot_Str.Theta.Zero.size:].astype('float32')
    else:
        if CONST_DATASET_ID == 0:
            # Input of the NN:  x -> Position(x, y, z), Orientation(quaternion)
            # Output of the NN: y -> theta(0 .. n - 1) 
            x = data[:, 0:7].astype('float32'); y = data[:, 7::].astype('float32')
        else:
            # Input of the NN:  x -> Position(x, y, z), Orientation(quaternion), theta(0 .. n - 1)
            # Output of the NN: y -> theta(n)
            x = data[:, 0:7].astype('float32')
            y = data[:, 7::].astype('float32').reshape(-1, 1)

    # https://github.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/blob/master/HPO_Classification.ipynb
    # ...
    #FCNN_IK_Optimizer_Cls = Lib.FCNN_IK.Model.FCNN_Optimizer_Cls(x=x, y=y, train_size=1.0, test_size=0.0, file_path=file_path_w)
    #   ...
    #FCNN_IK_Optimizer_Cls.Optimize(Parameters.Optimizer.FCNN_HYPERPARAMETERS_METHOD_0, 10, 2,False)

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist  # Replace with your custom dataset import
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from scikeras.wrappers import KerasClassifier
    from skopt import BayesSearchCV

    # Define a function to create a Keras model
    def create_model(learning_rate=0.01, units=64, activation='relu'):
        model = Sequential()
        model.add(Flatten(input_shape=(x.shape[1],)))
        model.add(Dense(units, activation=activation))
        model.add(Dense(y.shape[1], activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Create a KerasClassifier
    keras_classifier = KerasClassifier(build_fn=create_model, verbose=0, learning_rate=None, units=None, activation=None)

    print(keras_classifier.get_params().keys())
    param_grid = dict(learning_rate=(0.001, 0.1), units=(32, 256), activation=['relu', 'tanh', 'sigmoid'])

    # Create BayesSearchCV with KerasClassifier
    bayes_cv = BayesSearchCV(
        keras_classifier,
        param_grid,
        n_iter=20,  # Number of parameter combinations to try
        cv=3,       # Number of cross-validation folds
        n_jobs=-1   # Use all available CPU cores
    )

    # Perform hyperparameter tuning
    bayes_cv.fit(x, y)

    # Print the best hyperparameters and their corresponding score
    print("Best Hyperparameters: ", bayes_cv.best_params_)
    print("Best CV Score: {:.4f}".format(bayes_cv.best_score_))

    # Evaluate the model on the test set
    test_score = bayes_cv.score(x, y)
    print("Test Accuracy: {:.4f}".format(test_score))

if __name__ == "__main__":
    sys.exit(main())
