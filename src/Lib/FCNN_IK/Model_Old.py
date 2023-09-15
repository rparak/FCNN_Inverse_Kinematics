# OS (Operating system interfaces)
import os
# Tensorflow
import tensorflow as tf
# Scikit-Optimize, or skopt (Sequential model-based optimization) [pip3 install scikit-optimize]
import skopt

class RNN(object):
    """
    Description:
        A class for building and training a neural network model from a data set.

        Note:
            The structure of the neural network can be found in the script (Parameters.py).

    Initialization of the Class:
        Args:
            (1) x_train [Float Array]: Input training data.
            (2) y_train [Float Array]: Target training data.
            (3) x_test [Float Array]: Input validation data.
            (4) y_test [Float Array]: Target validation data.
            (5) file_name [String]: The name of the file to save the model architecture.

    Example:
        # Initialization of the class.
        RNN_cls = RNN(x_train, y_train, x_test, y_test, file_name)

        # Construct and compile a neural network model.
        RNN_cls.Build(**hyperparameters)

        # Save a model (image) of the neural network architecture.
        RNN_cls.Save(name)

        # Train a model over a fixed number of epochs.
        nn_history = RNN_cls.Train(in_epochs, in_batch_size)
    """

    def __init__(self, x_train, y_train, x_test, y_test, file_name):
        # << PRIVATE >> #
        # Input / Target training data.
        self.__x_train = x_train 
        self.__y_train = y_train
        # Input / Target validation data.
        self.__x_test = x_test
        self.__y_test = y_test
        # The name of the file to save the model architecture.
        self.__file_name = file_name

        # Initialization of a sequential neural network model.
        self.__model = tf.keras.models.Sequential()
        # A callback to save a model with a specific frequency.
        self.__callback = tf.keras.callbacks.ModelCheckpoint(filepath=f'{self.__file_name}.h5', 
                                                             verbose=2, save_best_only=True) 

    def Save(self, name):
        """
        Description:
            Function to save a model (image) of the neural network architecture.

        Args:
            (1) name [String]: The name of the file to save the model architecture.
        """

        tf.keras.utils.plot_model(self.__model, to_file=f'Architecture_Image/{name}.png', show_shapes=True, show_layer_names=True)

    def Build(self, architecture, hidden_layers_w_d, hidden_layers_wo_d, in_layer_units, hidden_layer_w_d_units, 
              hidden_layer_wo_d_units, in_layer_activation, kernel_layer_activation, layer_drop, opt, opt_learning_rate):
        """
        Description:
            A function to construct and compile a neural network model.

        Args:
            (1 - 11) NN Hyperparameters [-]: Neural Network hyperparameters.
                Note: 
                    More information can be found in the script (Parameters.py).
        """


        # Input layer.
        self.__model.add(architecture(in_layer_units, input_shape=(self.__x_train.shape[1], 1), activation=in_layer_activation))
        self.__model.add(tf.keras.layers.Dropout(layer_drop))

        # Hidden layers with dropout layer
        for _ in range(hidden_layers_w_d):
            self.__model.add(architecture(hidden_layer_w_d_units, activation=kernel_layer_activation))
            self.__model.add(tf.keras.layers.Dropout(layer_drop))

        # Hidden layers with dropout layer.
        if hidden_layers_wo_d != 0:
            for _ in range(hidden_layers_wo_d):
                self.__model.add(architecture(hidden_layer_wo_d_units, activation=kernel_layer_activation))

        # Output layer.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1]))

        # Compile a neural network model.
        self.__model.compile(loss='mse', optimizer=opt(learning_rate=opt_learning_rate), metrics= ['accuracy', 'mse', 'mae'])

    def Train(self, in_epochs, in_batch_size):
        """
        Description:
            A function to train a model over a fixed number of epochs (iterations on a dataset).

        Args:
            (1) in_epochs [INT]: Number of epochs (iterations) to train the model.
            (2) in_batch_size [INT]: Number of samples to update the gradient.
        """

        return self.__model.fit(self.__x_train, self.__y_train, epochs=in_epochs, batch_size=in_batch_size, verbose=2, 
                                validation_data=(self.__x_test, self.__y_test), callbacks = [self.__callback])


class RNN_Tuner(object):
    """
    Description:
        A class for tuning a neural network model from a dataset using a Bayesian search method.

        Note:
            The structure of the neural network can be found in the script (Parameters.py).

    Initialization of the Class:
        Args:
            (1) x_train [Float Array]: Input training data.
            (2) y_train [Float Array]: Target training data.
            (3) file_name [String]: The name of the file to save the tuner results.

    Example:
        # Initialization of the class.
        RNN_Tuner_cls = RNN_Tuner(x_train, y_train, file_name)

        # Tune a model hyperparameters.
        RNN_Tuner_cls.Tune(hyperparameters, iterations, cross_validation, save_results)

    """

    def __init__(self, x_train, y_train, file_name):
        # << PRIVATE >> #
        # Input / Target training data.
        self.__x_train = x_train
        self.__y_train = y_train
        # The name of the file to save the tuner results.
        self.__file_name = file_name

    def __Save(self, params, score):
        """
        Description:
            Function to save tuner results (hyperparameters, score).

        Args:
            (1) params [String]: Best tuning results (hyperparameters).
            (2) score [String] Best tuning score (loss).
        """

        f = open(f'{os.getcwd()}/Result_Tuner/BayesSearchCV/{self.__file_name}.txt', 'w')
        f.write('Parameters: \n' + params + '\n')
        f.write('Score: \n' + score)
        f.close()

    def __Build(self, architecture, hidden_layers_w_d, hidden_layers_wo_d, in_layer_units, hidden_layer_w_d_units, 
                hidden_layer_wo_d_units, in_layer_activation, kernel_layer_activation, layer_drop, opt, opt_learning_rate):

        """
        Description:
            A function to construct, compile, and return a neural network model.

        Args:
            (1 - 11) NN Hyperparameters [-]: More information can be found in the script (Parameters.py).
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Input layer.
        model.add(architecture(in_layer_units, input_shape=self.__x_train.shape[1::], return_sequences=True, activation=in_layer_activation))
        model.add(tf.keras.layers.Dropout(layer_drop))

        # Hidden layers with dropout layer.
        for _ in range(hidden_layers_w_d):
            model.add(architecture(hidden_layer_w_d_units, return_sequences=True, activation=kernel_layer_activation))
            model.add(tf.keras.layers.Dropout(layer_drop))

        # Hidden layers without dropout layer.
        if hidden_layers_wo_d != 0:
            for _ in range(hidden_layers_wo_d):
                model.add(architecture(hidden_layer_wo_d_units, return_sequences=True, activation=kernel_layer_activation))

        # Output layer.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[-1]))

        # Compile a neural network model.
        model.compile(loss='mse', optimizer=opt(learning_rate=opt_learning_rate), metrics=['accuracy', 'mse', 'mae'])

        return model
    
    def Tune(self, hyperparameters, iterations, cross_validation, save_results):
        """
        Description:
            A function to tune a model hyperparameters.

        Args:
            (1) hyperparameters [-]: Neural Network hyperparameters.
                Note: 
                    More information can be found in the script (Parameters.py).

            (2) iterations [INT]: Number of parameter settings that are sampled. 
            (3) cross_validation [INT]: Determines the cross-validation splitting strategy.
            (4) save_results [BOOL]: Specifies whether to save the tuner results.
        """

        regressor = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn = self.__Build)

        # Bayesian optimization over hyperparameters.
        bayes_search = skopt.BayesSearchCV(estimator=regressor, search_spaces = hyperparameters, n_iter = iterations, cv = cross_validation)
        # Run fit on the estimator.
        bayes_search.fit(self.__x_train, self.__y_train)

        if save_results == True:
            self.__Save(str(bayes_search.best_params_), str(bayes_search.best_score_))