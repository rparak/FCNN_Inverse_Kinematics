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
# Custom Script:
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
# Number of hidden layers for training and optimization.
CONST_NUM_OF_HIDDEN_LAYERS = 6

class FCNN_Trainer_Cls(object):
    """
    Description:
        fully-connected neural network (FCNN)

    Initialization of the Class:
        Args:
            (1) ... [...]: ...
            (..) file_path [string]: The specified path of the file without extension (format).

        Example:
            Initialization:
                # Assignment of the variables.
                ...

                # Initialization of the class.
                Cls = FCNN_Trainer_Cls()

            Features:
                # Properties of the class.
                ...

                # Functions of the class.
                ...
    """
        
    def __init__(self, x: tp.List[float], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str) -> None:

        try:
            assert (train_size + test_size) == 1.0

            # A variable that indicates that validation data will also be used for training.
            self.__use_validation = False if test_size == 0.0 else True

            # The data (History: <loss, mean square error, mean absolute error>, etc.) from the training.
            self.__train_data = None

            # Split the data from the dataset (x, y) into random train and validation subsets.
            if self.__use_validation == True:
                self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y, train_size=train_size, 
                                                                                                                                    test_size=test_size, random_state=0)
            else:
                self.__x_train = x; self.__y_train = y

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

            if self.__use_validation == True:
                # Transform of data using an the scale parameter.
                self.__x_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_x, self.__x_validation)
                self.__y_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_y, self.__y_validation)

                # A callback to save the model with a specific frequency.
                self.__callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{self.__file_path}_use_val_{self.__use_validation}.h5', monitor='val_loss', 
                                                                     save_best_only=True, verbose=1)
            else:
                self.__callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{self.__file_path}_use_val_{self.__use_validation}.h5', monitor='loss', 
                                                                     save_best_only=True, verbose=1)   
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrectly selected test and training set size. The sum of the set sizes must equal 1.0 and not {(train_size + test_size)}.')

    def __Release(self) -> None:
        """
        Description:
            Function to release GPU resources when the training process is already complete.
        """

        tf.keras.backend.clear_session()

    def Save(self) -> None:
        """
        Description:
            ...
        """

        # Save the scaler parameter for input/output data.
        joblib.dump(self.__scaler_x, f'{self.__file_path}_use_val_{self.__use_validation}_Scaler_x.pkl')
        joblib.dump(self.__scaler_y, f'{self.__file_path}_use_val_{self.__use_validation}_Scaler_y.pkl')
        print(f'[INFO] TThe input/output scalers have been successfully saved..')
        print(f'[INFO] >> file_path = {self.__file_path}_use_val_{self.__use_validation}_Scaler_x.pkl')
        print(f'[INFO] >> file_path = {self.__file_path}_use_val_{self.__use_validation}_Scaler_y.pkl')

        # Save a model (image) of the neural network architecture.
        tf.keras.utils.plot_model(self.__model, to_file=f'{self.__file_path}_use_val_{self.__use_validation}_Architecture.png', show_shapes=True, 
                                  show_layer_names=True)
        print(f'[INFO] The image of the neural network architecture has been successfully saved.')
        print(f'[INFO] >> file_path = {self.__file_path}_use_val_{self.__use_validation}_Architecture.png')
        
        # Save the data from the training.
        if self.__train_data != None:
            for _, data_i in enumerate(np.array(list(self.__train_data.history.values()), dtype=np.float32).T):
                File_IO.Save(f'{self.__file_path}_use_val_{self.__use_validation}_History', data_i, 'txt', ',')
            print(f'[INFO] The training data history has been successfully saved.')
            print(f'[INFO] >> file_path = {self.__file_path}_use_val_{self.__use_validation}_History.txt')

    def __Compile_Method_0(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ...

        Args:
            (1) Hyperparameters [Dictionary {}]: ..
        """

        # Set the input layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(Hyperparameters['in_layer_units'], input_shape=(self.__x_train.shape[1], ), 
                                               activation=Hyperparameters['in_layer_activation']))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, CONST_NUM_OF_HIDDEN_LAYERS):
            self.__model.add(tf.keras.layers.Dense(Hyperparameters[f'hidden_layer_{i + 1}_units'], activation=Hyperparameters['hidden_layer_activation'], 
                                                   use_bias=Hyperparameters['use_bias']))

        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['out_layer_activation'], 
                                               use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters['learning_rate']), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])

    def __Compile_Method_1(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ...

        Args:
            (1) Hyperparameters [Dictionary {}]: ..
        """

        # Set the input layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(Hyperparameters['in_layer_units'], input_shape=(self.__x_train.shape[1], ), 
                                               activation=Hyperparameters['in_layer_activation']))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, CONST_NUM_OF_HIDDEN_LAYERS):
            self.__model.add(tf.keras.layers.Dense(Hyperparameters[f'hidden_layer_{i + 1}_units'], activation=Hyperparameters['hidden_layer_activation'], 
                                                   use_bias=Hyperparameters['use_bias']))
            self.__model.add(tf.keras.layers.Dropout(Hyperparameters['layer_dropout']))

        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['out_layer_activation'], 
                                               use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters['learning_rate']), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])

    def Compile(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ...

        Args:
            (1) Hyperparameters [Dictionary {}]: ..
        """

        try:
            assert (self.__use_validation == True and 'layer_dropout' in Hyperparameters.keys()) or \
                   (self.__use_validation == False and 'layer_dropout' not in Hyperparameters.keys())
            
            if self.__use_validation == True:
                self.__Compile_Method_1(Hyperparameters)
            else:
                self.__Compile_Method_0(Hyperparameters)

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] The incorrect hyperparameter configuration has been chosen.')
        
    def Train(self, epochs: int, batch_size: int) -> None:
        """
        Description:
            ...

        Args:
            (1) epochs [int]: ..
            (2) batch_size [int]: ..
        """
 
        if self.__use_validation == True:
            self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, 
                                                 validation_data=(self.__x_validation_scaled, self.__y_validation_scaled), callbacks=[self.__callback])
        else:
            self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1,
                                                 validation_data=None, callbacks=[self.__callback])

        # Release GPU resources when the training process is already complete.
        self.__Release()

class FCNN_Predictor_Cls(object):
    """
    Description:
        ...

    Initialization of the Class:
        Args:
            (1) ... [...]: ...

        Example:
            Initialization:
                # Assignment of the variables.
                ...

                # Initialization of the class.
                Cls = FCNN_Predictor_Cls()

            Features:
                # Properties of the class.
                ...

                # Functions of the class.
                ...
    """
        
    def __init__(self) -> None:
        pass  

    def __Release(self) -> None:
        """
        Description:
            Function to release GPU resources when the training process is already complete.
        """

        tf.keras.backend.clear_session()

class FCNN_Optimizer_Cls(object):
    """
    Description:
        ...

        Reference:
            On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice, Li Yang and Abdallah Shami 
                https://arxiv.org/abs/2007.15745

    Initialization of the Class:
        Args:
            (1) ... [...]: ...

        Example:
            Initialization:
                # Assignment of the variables.
                ...

                # Initialization of the class.
                Cls = FCNN_Optimizer_Cls()

            Features:
                # Properties of the class.
                ...

                # Functions of the class.
                ...
    """
        
    def __init__(self, x: tp.List[float], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str) -> None:
        try:
            assert (train_size + test_size) == 1.0

            # A variable that indicates that validation data will also be used for training.
            self.__use_validation = False if test_size == 0.0 else True

            # Split the data from the dataset (x, y) into random train and validation subsets.
            if self.__use_validation == True:
                self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y, train_size=train_size, 
                                                                                                                                    test_size=test_size, random_state=0)
            else:
                self.__x_train = x; self.__y_train = y

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
        
            if self.__use_validation == True:
                # Transform of data using an the scale parameter.
                self.__x_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_x, self.__x_validation)
                self.__y_validation_scaled = Utilities.Transform_Data_With_Scaler(self.__scaler_y, self.__y_validation)
            
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] Incorrectly selected test and training set size. The sum of the set sizes must equal 1.0 and not {(train_size + test_size)}.')

    def __Release(self) -> None:
        """
        Description:
            Function to release GPU resources when the training process is already complete.
        """

        tf.keras.backend.clear_session()

    def __Save(self, parameters: tp.Dict) -> None:
        """
        Description:
            ...

        Args:
            (1) parameters [Dictionary {}]: ..
        """

        file_name = f'{self.__file_path}_Optimizer_Best_Results_use_validation_{self.__use_validation}.txt'

        # Remove the results file if it already exists.
        if os.path.isfile(file_name):
            os.remove(file_name)

        # Write the results obtained from the optimizer.
        with open(file_name, 'w') as f:
            for _, (key, value) in enumerate(parameters.items()):
                f.write(f'{key}: {value}\n')

        print(f'[INFO] The results obtained from the optimizer were successfully saved.')
        print(f'[INFO] >> file_path = {file_name}')

    def __Compile_Method_0(self, Hyperparameters: kt.engine.hyperparameters.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        """
        Description:
            ....

        Args:
            (1) Hyperparameters [kt.engine.hyperparameters.hyperparameters.HyperParameters(object)]]: ...

        Returns:
            (1) parameter [tf.keras.Sequential]: ...
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Defined general hyperparameters to be changed.
        #   Note:
        #       Other parameters are defined within each layer.
        use_bias = Hyperparameters.Choice('use_bias', values=[False, True]); hidden_layer_activation = Hyperparameters.Choice('hidden_layer_activation', 
                                                                                                    values=['relu', 'tanh'])

        # Set the input layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(Hyperparameters.Int('in_layer_units', min_value=32, max_value=64, step=32), input_shape=(self.__x_train.shape[1],), 
                                        activation=Hyperparameters.Choice('in_layer_activation', values=['relu', 'tanh'])))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, CONST_NUM_OF_HIDDEN_LAYERS):
            model.add(tf.keras.layers.Dense(Hyperparameters.Int(f'hidden_layer_{i + 1}_units', min_value=32, max_value=256, step=32), 
                                            activation=hidden_layer_activation, use_bias=use_bias))
        
        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters.Choice('out_layer_activation', values=['relu', 'tanh']), 
                                        use_bias=use_bias))

        # Finally, compile the model.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3])), loss='mse', 
                      metrics=['accuracy'])
        
        return model

    def __Compile_Method_1(self, Hyperparameters: kt.engine.hyperparameters.hyperparameters.HyperParameters) -> tf.keras.Sequential:
        """
        Description:
            ....

        Args:
            (1) Hyperparameters [kt.engine.hyperparameters.hyperparameters.HyperParameters(object)]: ...

        Returns:
            (1) parameter [tf.keras.Sequential]: ...
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Defined general hyperparameters to be changed.
        #   Note:
        #       Other parameters are defined within each layer.
        use_bias = Hyperparameters.Choice('use_bias', values=[False, True]); hidden_layer_activation = Hyperparameters.Choice('hidden_layer_activation', 
                                                                                                    values=['relu', 'tanh'])
        layer_dropout = Hyperparameters.Float('layer_dropout', min_value=0.005, max_value=0.05, step=0.005)

        # Set the input layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(Hyperparameters.Int('in_layer_units', min_value=32, max_value=64, step=32), input_shape=(self.__x_train.shape[1],), 
                                        activation=Hyperparameters.Choice('in_layer_activation', values=['relu', 'tanh'])))
        model.add(tf.keras.layers.Dropout(layer_dropout))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, CONST_NUM_OF_HIDDEN_LAYERS):
            model.add(tf.keras.layers.Dense(Hyperparameters.Int(f'hidden_layer_{i + 1}_units', min_value=32, max_value=256, step=32), 
                                            activation=hidden_layer_activation, use_bias=use_bias))
            model.add(tf.keras.layers.Dropout(layer_dropout))
        
        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters.Choice('out_layer_activation', values=['relu', 'tanh']), 
                                        use_bias=use_bias))

        # Finally, compile the model.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3])), loss='mse', 
                      metrics=['accuracy'])
        
        return model

    def Optimize(self, num_of_trials: int, epochs_per_trial: int, batch_size: int, save_results: bool) -> None:
        """
        Description:
            ...

        Args:
            (1) num_of_trials [int]: ...
            (1) epochs_per_trial [int]: ...
            (1) batch_size [int]: ...
            (4) save_results [bool]:
        """

        # Remove the unnecessary directory.
        directory_path = f'{CONST_PROJECT_FOLDER}/src/Training/FCNN_Inverse_Kinematics_Optimizer'
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)

        # Bayesian optimization with Gaussian process over the desired hyperparameters.
        if self.__use_validation == True:
            optimizer = kt.BayesianOptimization(hypermodel=self.__Compile_Method_1, objective=kt.Objective(name='val_accuracy', direction='max'), 
                                                         max_trials=num_of_trials, directory='FCNN_Inverse_Kinematics_Optimizer', project_name='FCNN_Inverse_Kinematics')
            
            # Start the search for the most suitable model.
            optimizer.search(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs_per_trial, batch_size=batch_size, 
                             validation_data=(self.__x_validation_scaled, self.__y_validation_scaled))
        else:
            optimizer = kt.BayesianOptimization(hypermodel=self.__Compile_Method_0, objective=kt.Objective(name='accuracy', direction='max'), 
                                                         max_trials=num_of_trials, directory='FCNN_Inverse_Kinematics_Optimizer', project_name='FCNN_Inverse_Kinematics')
            
            # Start the search for the most suitable model.
            optimizer.search(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs_per_trial, batch_size=batch_size, 
                             validation_data=None)

        # Get the best hyperparameters determined by the objective function.
        #   objective = 'accuracy/val_accuracy'
        best_hps = optimizer.get_best_hyperparameters(num_trials=1)[0]

        # Save the results of the best parameters along with the score.
        if save_results == True:
            self.__Save(best_hps.values)

        # Release GPU resources when the training process is already complete.
        self.__Release()