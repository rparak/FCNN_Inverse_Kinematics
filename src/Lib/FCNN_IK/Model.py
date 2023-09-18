# OS (Operating system interfaces)
import os
# Typing (Support for type hints)
import typing as tp
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf
# Scikit-Optimize, or skopt (Sequential model-based optimization) [pip3 install scikit-optimize]
import skopt
# Joblib (Lightweight pipelining) [pip3 install joblib]
import joblib
# Sklearn (Simple and efficient tools for predictive data analysis) [pip3 install scikit-learn]
import sklearn.model_selection
# ...
from scikeras.wrappers import KerasClassifier
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
CONST_PROJECT_FOLDER = os.getcwd().split('DNN_Inverse_Kinematics')[0] + 'DNN_Inverse_Kinematics'

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

    def Save(self):
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
            ....
        """

        # Set the input layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(Hyperparameters['in_layer_units'], input_shape=(self.__x_train.shape[1], ), 
                                               activation=Hyperparameters['in_layer_activation']))

        # Set the hidden layers of the FCNN model architecture.
        if Hyperparameters['hidden_layers'] > 0:
            for _, hidden_layer_i in enumerate(Hyperparameters['hidden_layer_units']):
                self.__model.add(tf.keras.layers.Dense(hidden_layer_i, activation=Hyperparameters['kernel_layer_activation'], 
                                                       use_bias=Hyperparameters['use_bias']))

        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['kernel_layer_activation'], 
                                               use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        self.__model.compile(optimizer=Hyperparameters['opt'](learning_rate=Hyperparameters['opt_learning_rate']), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])

    def __Compile_Method_1(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ....
        """

        # Set the input layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(Hyperparameters['in_layer_units'], input_shape=(self.__x_train.shape[1], ), 
                                               activation=Hyperparameters['in_layer_activation']))

        # Set the hidden layers of the FCNN model architecture.
        #   1\ Hidden layers with dropout layer.
        if Hyperparameters['hidden_layers_w_d'] > 0:
            for _, hidden_layer_i in enumerate(Hyperparameters['hidden_layer_w_d_units']):
                self.__model.add(tf.keras.layers.Dense(hidden_layer_i, activation=Hyperparameters['kernel_layer_activation'], 
                                                                use_bias=Hyperparameters['use_bias']))
                self.__model.add(tf.keras.layers.Dropout(Hyperparameters['layer_drop']))

        #   1\ Hidden layers without dropout layer.
        if Hyperparameters['hidden_layers_wo_d'] > 0:
            for _, hidden_layer_i in enumerate(Hyperparameters['hidden_layer_wo_d_units']):
                self.__model.add(tf.keras.layers.Dense(hidden_layer_i, activation=Hyperparameters['kernel_layer_activation'], 
                                                       use_bias=Hyperparameters['use_bias']))
            
        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['kernel_layer_activation'], 
                                               use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        self.__model.compile(optimizer=Hyperparameters['opt'](learning_rate=Hyperparameters['opt_learning_rate']), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])

    def Compile(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ...
        """

        try:
            assert (self.__use_validation == True and 'hidden_layers_w_d' in Hyperparameters.keys()) or \
                   (self.__use_validation == False and 'hidden_layers' in Hyperparameters.keys())
            
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
        """
 
        if self.__use_validation == True:
            self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, 
                                                 validation_data=(self.__x_validation_scaled, self.__y_validation_scaled), callbacks = [self.__callback])
        else:
            self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1,
                                                 callbacks = [self.__callback])

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

    def __Save(self, parameters: str, score: str):
        """
        Description:
            ...
        """
                
        for _, data_i in enumerate([parameters, score]):
            File_IO.Save(f'{self.__file_path}_use_val_{self.__use_validation}_History', data_i, 'txt', ',')

        print(f'[INFO] The results obtained from the optimizer were successfully saved.')
        print(f'[INFO] >> file_path = {self.__file_path}_optimizer_results.txt')

    def __Compile_Method_0(self, in_layer_units, in_layer_activation, hidden_layers, hidden_layer_units, kernel_layer_activation,
                           use_bias, opt, opt_learning_rate) -> None:
        """
        Description:
            ....
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Set the input layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(in_layer_units, input_shape=(self.__x_train.shape[1], ), 
                                        activation=in_layer_activation))
        
        if hidden_layers > 0:
            # Set the hidden layers of the FCNN model architecture.
            for _ in range(hidden_layers):
                model.add(tf.keras.layers.Dense(hidden_layer_units, activation=kernel_layer_activation, 
                                                use_bias=use_bias))
        
        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=kernel_layer_activation, 
                                        use_bias=use_bias))

        # Finally, compile the model.
        return model.compile(optimizer=opt(learning_rate=opt_learning_rate), loss='mse', 
                             metrics=['accuracy', 'mse', 'mae'])

    def __Compile_Method_1(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            ....
        """
        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()
        # Set the input layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(Hyperparameters['in_layer_units'], input_shape=(self.__x_train.shape[1], ), 
                                        activation=Hyperparameters['in_layer_activation']))

        # Set the hidden layers of the FCNN model architecture.
        #   1\ Hidden layers with dropout layer.
        if Hyperparameters['hidden_layers_w_d'] > 0:
            for _, hidden_layer_i in enumerate(Hyperparameters['hidden_layer_w_d_units']):
                model.add(tf.keras.layers.Dense(hidden_layer_i, activation=Hyperparameters['kernel_layer_activation'], 
                                                use_bias=Hyperparameters['use_bias']))
                model.add(tf.keras.layers.Dropout(Hyperparameters['layer_drop']))

        #   1\ Hidden layers without dropout layer.
        if Hyperparameters['hidden_layers_wo_d'] > 0:
            for _, hidden_layer_i in enumerate(Hyperparameters['hidden_layer_wo_d_units']):
                model.add(tf.keras.layers.Dense(hidden_layer_i, activation=Hyperparameters['kernel_layer_activation'], 
                                                use_bias=Hyperparameters['use_bias']))
            
        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation=Hyperparameters['kernel_layer_activation'], 
                                        use_bias=Hyperparameters['use_bias']))

        # Finally, compile the model.
        return model.compile(optimizer=Hyperparameters['opt'](learning_rate=Hyperparameters['opt_learning_rate']))

    def Optimize(self, Hyperparameters: tp.Dict, iterations: int, cross_validation: int, save_results: bool):
        """
        Description:
            ...
        """
        try:
            assert (self.__use_validation == True and 'hidden_layers_w_d' in Hyperparameters.keys()) or \
                   (self.__use_validation == False and 'hidden_layers' in Hyperparameters.keys())
                 
            if self.__use_validation == True:
                keras_regressor = KerasClassifier(build_fn=self.__Compile_Method_1, batch_size=64, shuffle=False, verbose=0)
            else:
                keras_regressor = KerasClassifier(build_fn=self.__Compile_Method_0, batch_size=64, optimizer=tf.keras.optimizers.Adam, lr=1e-03, shuffle=False, verbose=0,
                                                  in_layer_units=32, in_layer_activation='relu', hidden_layers=3, hidden_layer_units=32,
                                                  kernel_layer_activation='relu', use_bias=False)

            print(keras_regressor.get_params().keys())
            # Bayesian optimization over the desired hyperparameters.
            opt_bayes_search = skopt.BayesSearchCV(estimator=keras_regressor, search_spaces=Hyperparameters, n_iter=iterations, 
                                                   cv=cross_validation, scoring='neg_mean_squared_error', verbose=1)
            
            # Run fit on the estimator.
            if self.__use_validation == True:
                _ = opt_bayes_search.fit(self.__x_train_scaled, self.__y_train_scaled, validation_data=(self.__x_validation_scaled, self.__y_validation_scaled))
            else:
                _ = opt_bayes_search.fit(self.__x_train_scaled, self.__y_train_scaled)

            # Save the results of the best parameters along with the score.
            if save_results == True:
                self.__Save(str(opt_bayes_search.best_params_), str(opt_bayes_search.best_score_))

            # Release GPU resources when the training process is already complete.
            self.__Release()

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] The incorrect hyperparameter configuration has been chosen.')