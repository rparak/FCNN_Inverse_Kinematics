# OS (Operating system interfaces)
import os
# Typing (Support for type hints)
import typing as tp
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf
# Joblib (Lightweight pipelining) [pip3 install joblib]
import joblib
# Sklearn (Simple and efficient tools for predictive data analysis) [pip3 install scikit-learn]
import sklearn.model_selection
# KerasTuner (Hyperparameter optimization) [pip3 install keras-tuner]
import keras_tuner as kt
# Shutil (High-level file operations)
import shutil
# Custom Lib.:
#   ../Lib/FCNN_IK/Utilities
import Lib.FCNN_IK.Utilities as Utilities
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# Locate the path to the project folder.
CONST_PROJECT_FOLDER = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

class FCNN_Trainer_Cls(object):
    """
    Description:
        A specific class for training an Inverse Kinematics (IK) task using a Fully-Connected Neural Network (FCNN).

    Initialization of the Class:
        Args:
            (1) x [Vector<float> nxm]: Input data.
                                        Note:
                                            Where n is the number of data and m is the number of input parameters.
            (2) y [Vector<float> nxk]: Output (target) data.
                                        Note:
                                            Where n is the number of data and k is the number of output parameters.
            (3) train_size [float]: The size of the training partition.
            (4) test_size [float]: The size of the validation partition.
            (5) file_path [string]: The specified path of the file without extension (format).

        Example:
            Initialization:
                # Assignment of the variables.
                #   Hyperparameters of a specific robotic structure 
                #   obtained from optimization.
                Hyperparameters_Str = Hyperparameters.EPSON_LS3_B401S
                #   In/Out data.
                x_in  = Dataset[:, ..]
                y_out = Dataset[:, ..]

                # Initialization of the class.
                Cls = FCNN_Trainer_Cls(x=x_in, y=y_out, train_size=0.8, test_size=0.2,
                                       '../..')
            Features:
                # Functions of the class.
                Cls.Compile(Hyperparameters_Str); Cls.Train(epochs=100, batch_size=64)
                Cls.Save()
    """
        
    def __init__(self, x: tp.List[float], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str) -> None:

        try:
            assert (train_size + test_size) == 1.0 and test_size > 0.0

            # The data (History: <loss, mean square error, mean absolute error>, etc.) from the training.
            self.__train_data = None

            # Split the data from the dataset (x, y) into random train and validation subsets.
            self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y, 
                                                                                                                                train_size=train_size, test_size=test_size, 
                                                                                                                                shuffle=1, random_state=0)

            # Find the scale parameter from the dataset and transform the data using this parameter.
            self.__scaler_x, self.__x_train_scaled = Utilities.Scale_Data([-1.0, 1.0], self.__x_train)
            self.__scaler_y, self.__y_train_scaled = Utilities.Scale_Data([-1.0, 1.0], self.__y_train)
            
            # The file path to save the data.
            self.__file_path = file_path

            # Set whether memory growth should be enabled.
            gpu_arr = tf.config.experimental.list_physical_devices('GPU')
            if gpu_arr:
                for _, gpu_i in enumerate(gpu_arr):
                    tf.config.experimental.set_memory_growth(gpu_i, True)
            else:
                print('[INFO] No GPU device was found.')

            # Initialization of a sequential neural network model.
            self.__model = tf.keras.models.Sequential()

            # Transform of data using an the scale parameter.
            self.__x_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_x, self.__x_validation)
            self.__y_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_y, self.__y_validation)

            # A callback to save the model with a specific frequency.
            self.__callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{self.__file_path}.h5', monitor='val_loss', 
                                                                 save_best_only=True, verbose=1)
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            if (train_size + test_size) != 1.0:
                print(f'[ERROR] Incorrectly selected test and training set size. The sum of the set sizes must equal 1.0 and not {(train_size + test_size)}.')
            if test_size <= 0.0:
                print(f'[ERROR] Incorrectly selected test set size. The size of the validation partition (test_size) must be greater than 0.0.')

    def __Release(self) -> None:
        """
        Description:
            Function to release GPU resources when the training process is already complete.
        """

        tf.keras.backend.clear_session()

    def Save(self) -> None:
        """
        Description:
            A function to save important files from the training.

            Note:
                The files can be found in the file path specified in the input parameters of the class.
        """

        # Save the scaler parameter for input/output data.
        joblib.dump(self.__scaler_x, f'{self.__file_path}_Scaler_x.pkl')
        joblib.dump(self.__scaler_y, f'{self.__file_path}_Scaler_y.pkl')
        print(f'[INFO] The input/output scalers have been successfully saved..')
        print(f'[INFO] >> file_path = {self.__file_path}_Scaler_x.pkl')
        print(f'[INFO] >> file_path = {self.__file_path}_Scaler_y.pkl')

        # Save a model (image) of the neural network architecture.
        tf.keras.utils.plot_model(self.__model, to_file=f'{self.__file_path}_Architecture.png', show_shapes=True, 
                                  show_layer_names=True)
        print(f'[INFO] The image of the neural network architecture has been successfully saved.')
        print(f'[INFO] >> file_path = {self.__file_path}_Architecture.png')
        
        # Save the data from the training.
        if self.__train_data != None:
            for _, data_i in enumerate(np.array(list(self.__train_data.history.values()), dtype=np.float64).T):
                File_IO.Save(f'{self.__file_path}_History', data_i, 'txt', ',')
            print(f'[INFO] The training data history has been successfully saved.')
            print(f'[INFO] >> file_path = {self.__file_path}_History.txt')

    def Compile(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            A function to compile the model.

            Note:
                The model will be configured with the losses and metrics.

        Args:
            (1) Hyperparameters [Dictionary {..}]: The structure of the hyperparameters.
                                                    For more information about hyperparameters, see the script below:
                                                        ../Training/Hyperparameters/{robot_name}.py,

                                                        where the {robot_name} is the name of the individual robotic 
                                                        structure.
        """

        # Set the input layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__x_train.shape[1], input_shape=(self.__x_train.shape[1], ), 
                                               activation=Hyperparameters['in_layer_activation'], use_bias=Hyperparameters['use_bias']))
        self.__model.add(tf.keras.layers.Dropout(Hyperparameters['layer_dropout']))
        
        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, Hyperparameters['num_of_hidden_layers']):
            self.__model.add(tf.keras.layers.Dense(Hyperparameters[f'hidden_layer_{i + 1}_units'], activation=Hyperparameters['hidden_layer_activation'], 
                                                   use_bias=Hyperparameters['use_bias']))
            self.__model.add(tf.keras.layers.Dropout(Hyperparameters['layer_dropout']))

        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['out_layer_activation'], 
                                               use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters['learning_rate']), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])
        
    def Train(self, epochs: int, batch_size: int) -> None:
        """
        Description:
            A function to train the Fully-Connected Neural Network (FCNN) model.

        Args:
            (1) epochs [int]: The number of epochs (iterations) to train the model.
            (2) batch_size [int]: The number of samples processed before the model is updated.
        """
 
        self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, 
                                             validation_data=(self.__x_validation_scaled, self.__y_validation_scaled), callbacks=[self.__callback])

        # Release GPU resources when the training process is already complete.
        self.__Release()

class FCNN_Predictor_Cls(object):
    """
    Description:
        A specific class for predicting the absolute joint position of the robotic arm using a Fully-Connected 
        Neural Network (FCNN) trained with Forward Kinematics (FK).

    Initialization of the Class:
        Args:
            (1) scaler_x_file_path [string]: The specified path to the file of the input data scaler.
            (2) scaler_y_file_path [string]: The specified path to the file of the output data scaler.
            (3) model_file_path [string]: The specified path to the file of the trained model.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = FCNN_Predictor_Cls('../..', '../..', 
                                         '../..')

            Features:
                # Functions of the class.
                Cls.Predict([p(x, y, z), q(w, x, y, z)])
    """
        
    def __init__(self, scaler_x_file_path: str, scaler_y_file_path: str, model_file_path: str) -> None:
        # Load the scaler parameter for input/output data.
        self.__scaler_x = joblib.load(scaler_x_file_path)
        self.__scaler_y = joblib.load(scaler_y_file_path)

        # Load the trained model from the folder.
        self.__model = tf.keras.models.load_model(model_file_path)

    def Predict(self, x: tp.List[float]) -> tp.List[float]:
        """
        Description:
            A function to predict the absolute joint position of the robotic arm from the input 
            position and orientation of the end-effector.

        Args:
            (1) x [Vector<float> 1x7]: Input data as TCP (Tool Center Point) of the robotic arm.
                                        Note:
                                            Defined as position (x, y, z) and orientation 
                                            in quaternions (w, x, y, z).

        Returns:
            (1) parameter [Vector<float> 1xm]: Output (target) data as absolute joint position of the robotic arm.
                                                Note:
                                                    Where m is the number of joints.
        """

        # Transform of data using the scale parameter.
        x_transfored = Utilities.Transform_Data_With_Scaler(self.__scaler_x, x.astype('float32').reshape(1, x.shape[0]))

        # Generates output predictions from input transformed data.
        y = self.__model.predict(x_transfored)

        return Utilities.Inverse_Data_With_Scaler(self.__scaler_y, y.astype('float32').reshape(1, y.shape[1]))

class FCNN_Optimizer_Cls(object):
    """
    Description:
        A specific class for hyperparameter optimization of the Fully-Connected Neural Network (FCNN).

        How can I calculate the number of hidden layers?
            A rough approximation can be obtained using the geometric pyramid rule proposed by Masters (Practical Neural Network 
            Recipies in C++, 1993).

                N_{h} = sqrt(N_{i}*N_{o}),

            where N_{h} is the number of hidden layers, N_{i} is the number of input neurons, and N_{o} is the number 
            of output neurons.

            To optimize the number of hidden layers, use the formula above with addition of 1 value.

        Reference:
            On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice, Li Yang and Abdallah Shami 
                https://arxiv.org/abs/2007.15745

    Initialization of the Class:
        Args:
            (1) x [Vector<float> nxm]: Input data.
                                        Note:
                                            Where n is the number of data and m is the number of input parameters.
            (2) y [Vector<float> nxk]: Output (target) data.
                                        Note:
                                            Where n is the number of data and k is the number of output parameters.
            (3) train_size [float]: The size of the training partition.
            (4) test_size [float]: The size of the validation partition.
            (5) file_path [string]: The specified path of the file without extension (format).

        Example:
            Initialization:
                # Assignment of the variables.
                #   In/Out data.
                x_in  = Dataset[:, ..]
                y_out = Dataset[:, ..]

                # Initialization of the class.
                Cls = FCNN_Optimizer_Cls(x=x_in, y=y_out, train_size=0.8, test_size=0.2,
                                         '../..')
            Features:
                # Functions of the class.
                Cls.Optimize(num_of_trials=100, epochs_per_trial=100, batch_size=64, 
                             save_results=True)
    """
        
    def __init__(self, x: tp.List[float], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str) -> None:
        try:
            assert (train_size + test_size) == 1.0 and test_size > 0.0

            # Split the data from the dataset (x, y) into random train and validation subsets.
            self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y, 
                                                                                                                                train_size=train_size, test_size=test_size, 
                                                                                                                                shuffle=1, random_state=0)

            # Find the scale parameter from the dataset and transform the data using this parameter.
            self.__scaler_x, self.__x_train_scaled = Utilities.Scale_Data([-1.0, 1.0], self.__x_train)
            self.__scaler_y, self.__y_train_scaled = Utilities.Scale_Data([-1.0, 1.0], self.__y_train)
            
            # The file path to save the data.
            self.__file_path = file_path

            # Set whether memory growth should be enabled.
            gpu_arr = tf.config.experimental.list_physical_devices('GPU')
            if gpu_arr:
                for _, gpu_i in enumerate(gpu_arr):
                    tf.config.experimental.set_memory_growth(gpu_i, True)
            else:
                print('[INFO] No GPU device was found.')
        
            # Transform of data using an the scale parameter.
            self.__x_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_x, self.__x_validation)
            self.__y_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_y, self.__y_validation)
            
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            if (train_size + test_size) != 1.0:
                print(f'[ERROR] Incorrectly selected test and training set size. The sum of the set sizes must equal 1.0 and not {(train_size + test_size)}.')
            if test_size <= 0.0:
                print(f'[ERROR] Incorrectly selected test set size. The size of the validation partition (test_size) must be greater than 0.0.')

    def __Release(self) -> None:
        """
        Description:
            Function to release GPU resources when the training process is already complete.
        """

        tf.keras.backend.clear_session()

    def __Save(self, parameters: tp.Dict) -> None:
        """
        Description:
            A function to save important files from the optimization.

            Note:
                The files can be found in the file path specified in the input parameters of the class.
        """

        file_name = f'{self.__file_path}_Optimizer_Best_Results.txt'

        # Remove the results file if it already exists.
        if os.path.isfile(file_name):
            os.remove(file_name)

        # Write the results obtained from the optimizer.
        with open(file_name, 'w') as f:
            for _, (key, value) in enumerate(parameters.items()):
                f.write(f'{key}: {value}\n')

        print(f'[INFO] The results obtained from the optimizer were successfully saved.')
        print(f'[INFO] >> file_path = {file_name}')

    def __Compile(self, Hyperparameters: kt.engine.hyperparameters.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        """
        Description:
            A function to compile the model.

            Note:
                The function will be called for each optimization trial.

        Args:
            (1) Hyperparameters [kt.engine.hyperparameters.hyperparameters.HyperParameters(object)]]: The structure of the hyperparameters 
                                                                                                      to be optimized.

        Returns:
            (1) parameter [tf.keras.Sequential]: A model grouping layers into an object with training and inference features.
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Defined general hyperparameters to be changed.
        #   Note:
        #       Other parameters are defined within each layer.
        use_bias = Hyperparameters.Choice('use_bias', values=[False, True]); hidden_layer_activation = Hyperparameters.Choice('hidden_layer_activation', 
                                                                                                                              values=['linear', 'relu', 'tanh'])
        layer_dropout = Hyperparameters.Float('layer_dropout', min_value=0.05, max_value=0.20, step=0.05)

        # Set the input layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__x_train.shape[1], input_shape=(self.__x_train.shape[1],), 
                                        activation=Hyperparameters.Choice('in_layer_activation', values=['linear', 'relu', 'tanh']), use_bias=use_bias))
        model.add(tf.keras.layers.Dropout(layer_dropout))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, Hyperparameters.Int('num_of_hidden_layers', min_value=1, max_value=int(((self.__x_train.shape[1] * self.__y_train.shape[1]) ** 0.5)) + 1, 
                                              step=1)):
            model.add(tf.keras.layers.Dense(Hyperparameters.Int(f'hidden_layer_{i + 1}_units', min_value=32, max_value=128, step=32), 
                                            activation=hidden_layer_activation, use_bias=use_bias))
            model.add(tf.keras.layers.Dropout(layer_dropout))
        
        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters.Choice('out_layer_activation', values=['linear', 'relu', 'tanh']), 
                                        use_bias=use_bias))

        # Finally, compile the model.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3])), loss='mse', 
                      metrics=['accuracy'])
        
        return model

    def Optimize(self, num_of_trials: int, epochs_per_trial: int, batch_size: int, save_results: bool) -> None:
        """
        Description:
            A function to optimize the Fully-Connected Neural Network (FCNN) model.

        Args:
            (1) num_of_trials [int]: The number of trials to be used in hyperparameter optimization.
            (2) epochs_per_trial [int]: The number of epochs (iterations) to train the model per trial.
            (3) batch_size [int]: The number of samples processed before the model is updated.
            (4) save_results [bool]: Information about whether the optimization results will be saved.
        """

        # Remove the unnecessary directory.
        directory_path = f'{CONST_PROJECT_FOLDER}/src/Training/FCNN_Inverse_Kinematics_Optimizer'
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)

        # Bayesian optimization with Gaussian process over the desired hyperparameters.
        optimizer = kt.BayesianOptimization(hypermodel=self.__Compile, objective=kt.Objective(name='val_accuracy', direction='max'), 
                                            max_trials=num_of_trials, directory='FCNN_Inverse_Kinematics_Optimizer', project_name='FCNN_Inverse_Kinematics')
        
        # Start the search for the most suitable model.
        optimizer.search(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs_per_trial, batch_size=batch_size, 
                            validation_data=(self.__x_validation_scaled, self.__y_validation_scaled))
        
        # Get the best hyperparameters determined by the objective function.
        #   objective = 'accuracy/val_accuracy'
        best_hps = optimizer.get_best_hyperparameters(num_trials=1)[0]

        # Save the results of the best parameters along with the score.
        if save_results == True:
            self.__Save(best_hps.values)

        # Release GPU resources when the optimization process is already complete.
        self.__Release()
