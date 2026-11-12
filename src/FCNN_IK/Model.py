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
# Logging (Logging facility for Python)
import logging
# Custom Lib.:
#   ../FCNN_IK/Utilities
import FCNN_IK.Utilities as Utilities
#   ../Utilities/File_IO
import Utilities.File_IO as File_IO
#   ../Kinematics/Core
import Kinematics.Core
#   ../Parameters/Robot
import Parameters.Robot

"""
Description:
    Initialization of constants.
"""
# Locate the path to the project folder.
CONST_PROJECT_FOLDER = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'
# The seed of the random number generators (Python, NumPy, TensorFlow).
CONST_SEED = 0
# The order of the columns in the training history file (*_History.txt).
CONST_HISTORY_KEYS = ['loss', 'mse', 'mae', 'val_loss', 'val_mse', 'val_mae', 'learning_rate']
# The maximum position error of the analytical IK solution for a reachable query.
#   Unit: [meter]
CONST_REACHABILITY_TOLERANCE = 1e-6
# The harmless TensorFlow deprecation message to be suppressed.
#   Note:
#       tf.keras.backend.clear_session() internally calls tf.compat.v1.reset_default_graph(),
#       which logs this message. All other TensorFlow messages are kept.
CONST_SUPPRESSED_TF_MESSAGE = 'tf.reset_default_graph is deprecated'

class Message_Filter_Cls(logging.Filter):
    """
    Description:
        A logging filter that removes the records containing a specific message.

    Initialization of the Class:
        Args:
            (1) message [string]: The (part of the) message to be removed.

        Example:
            Initialization:
                tf.get_logger().addFilter(Message_Filter_Cls(CONST_SUPPRESSED_TF_MESSAGE))
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self.__message = message

    def filter(self, record: logging.LogRecord) -> bool:
        return self.__message not in record.getMessage()

# Suppress the harmless deprecation message of TensorFlow (see CONST_SUPPRESSED_TF_MESSAGE).
tf.get_logger().addFilter(Message_Filter_Cls(CONST_SUPPRESSED_TF_MESSAGE))

@tf.keras.utils.register_keras_serializable(package='FCNN_IK')
class Position_Consistency_Loss_Cls(tf.keras.losses.Loss):
    """
    Description:
        A specific class of the position-consistency loss function for training an Inverse Kinematics (IK) task.

            loss = MSE(theta_scaled) + lambda * mean(||FK(theta_predicted) - FK(theta_desired)||^2),

        where FK is a differentiable forward kinematics of the RR robotic structure built from the
        Denavit-Hartenberg (DH) parameters. The joint positions are unscaled inside TensorFlow using
        the parameters of the min-max scaler.

        Note:
            The labels of the dataset satisfy FK(theta_desired) = p, so the second term is the squared
            position error of the prediction in meters^2.

    Initialization of the Class:
        Args:
            (1) a [Vector<float> 1xm]: Link lengths (a_i) of the DH table.
                                       Unit: [meter]
            (2) theta_zero [Vector<float> 1xm]: Joint offsets (theta_zero_i) of the DH table.
                                                Unit: [radian]
            (3) y_scale [Vector<float> 1xm]: The parameter 'scale_' of the output data scaler.
            (4) y_min [Vector<float> 1xm]: The parameter 'min_' of the output data scaler.
            (5) position_loss_weight [float]: The weight (lambda) of the position term.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = Position_Consistency_Loss_Cls(a=Robot_Str.DH.Standard[:, 1], theta_zero=Robot_Str.DH.Standard[:, 0],
                                                    y_scale=scaler_y.scale_, y_min=scaler_y.min_, position_loss_weight=1.0)
            Features:
                # Use of the loss function.
                model.compile(optimizer=.., loss=Cls)
    """

    def __init__(self, a: tp.List[float], theta_zero: tp.List[float], y_scale: tp.List[float], y_min: tp.List[float],
                 position_loss_weight: float, name: str = 'position_consistency_loss', **kwargs) -> None:
        super().__init__(name=name, **kwargs)

        # Store the parameters as lists (serializable configuration).
        self.a = [float(x) for x in a]; self.theta_zero = [float(x) for x in theta_zero]
        self.y_scale = [float(x) for x in y_scale]; self.y_min = [float(x) for x in y_min]
        self.position_loss_weight = float(position_loss_weight)

    def __Forward_Kinematics(self, y_scaled: tf.Tensor) -> tf.Tensor:
        """
        Description:
            Differentiable forward kinematics of the RR robotic structure from the scaled joint positions.

        Args:
            (1) y_scaled [tf.Tensor nxm]: Scaled absolute joint positions.

        Returns:
            (1) parameter [tf.Tensor nx2]: The TCP (tool center point) position in the x, y axes.
                                           Unit: [meter]
        """

        # Inverse transformation (unscaling) of the joint positions.
        #   Note:
        #       sklearn.preprocessing.MinMaxScaler: y_scaled = y * scale_ + min_
        theta = (y_scaled - tf.constant(self.y_min, dtype=y_scaled.dtype)) / tf.constant(self.y_scale, dtype=y_scaled.dtype)

        th_0  = self.theta_zero[0] + theta[:, 0]
        th_01 = th_0 + self.theta_zero[1] + theta[:, 1]

        return tf.stack([self.a[0]*tf.cos(th_0) + self.a[1]*tf.cos(th_01),
                         self.a[0]*tf.sin(th_0) + self.a[1]*tf.sin(th_01)], axis=-1)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, y_pred.dtype)

        # Mean squared error of the scaled joint positions.
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

        # Squared position error.
        e_p = tf.reduce_sum(tf.square(self.__Forward_Kinematics(y_pred) - self.__Forward_Kinematics(y_true)), axis=-1)

        return mse + self.position_loss_weight * e_p

    def get_config(self) -> tp.Dict:
        config = super().get_config()
        config.update({'a': self.a, 'theta_zero': self.theta_zero, 'y_scale': self.y_scale, 'y_min': self.y_min,
                       'position_loss_weight': self.position_loss_weight})

        return config

def Is_Relative_Improvement(monitor_value: float, reference_value: tp.Optional[float], min_delta: float) -> bool:
    """
    Description:
        Check whether the monitored quantity (e.g. 'val_loss') has improved by at least a relative fraction
        of the best value so far, i.e. monitor_value < (1 - min_delta) * reference_value.

        Note:
            Unlike the absolute 'min_delta' of the Keras callbacks, the relative threshold does not depend on
            the scale of the loss (the loss decreases by several orders of magnitude during training). The
            monitored quantity is assumed to be minimized and positive.

    Args:
        (1) monitor_value [float]: The current value of the monitored quantity.
        (2) reference_value [float or None]: The best value of the monitored quantity so far (None -> first epoch).
        (3) min_delta [float]: The minimum relative improvement, e.g. 0.01 -> 1 %.

    Returns:
        (1) parameter [bool]: The information whether the monitored quantity has improved (True) or not (False).
    """

    if reference_value is None:
        return True

    return bool(monitor_value < (1.0 - min_delta) * reference_value)

class Relative_Early_Stopping_Cls(tf.keras.callbacks.EarlyStopping):
    """
    Description:
        Early stopping with a relative improvement threshold. 'min_delta' is a relative fraction of the best
        value of the monitored quantity (see Is_Relative_Improvement()); the remaining arguments are the same
        as for tf.keras.callbacks.EarlyStopping.
    """

    def _is_improvement(self, monitor_value, reference_value) -> bool:
        # Note:
        #   Keras stores 'min_delta' with a negative sign for a minimized quantity.
        return Is_Relative_Improvement(monitor_value, reference_value, abs(self.min_delta))

class Relative_Reduce_LR_On_Plateau_Cls(tf.keras.callbacks.ReduceLROnPlateau):
    """
    Description:
        Reduction of the learning rate on a plateau with a relative improvement threshold. 'min_delta' is
        a relative fraction of the best value of the monitored quantity (see Is_Relative_Improvement()); the
        remaining arguments are the same as for tf.keras.callbacks.ReduceLROnPlateau.
    """

    def _is_improvement(self, monitor_value, reference_value) -> bool:
        # Note:
        #   Keras stores 'min_delta' with a negative sign for a minimized quantity.
        return Is_Relative_Improvement(monitor_value, reference_value, abs(self.min_delta))

class FCNN_Trainer_Cls(object):
    """
    Description:
        A specific class for training an Inverse Kinematics (IK) task using a Fully-Connected Neural Network (FCNN).

    Initialization of the Class:
        Args:
            (1) x [Vector<[float, int]> nxm]: Input data.
                                                Note:
                                                    Where n is the number of data and m is the number of input parameters.
            (2) y [Vector<float> nxk]: Output (target) data.
                                        Note:
                                            Where n is the number of data and k is the number of output parameters.
            (3) train_size [float]: The size of the training partition.
            (4) test_size [float]: The size of the validation partition.
            (5) file_path [string]: The specified path of the file without extension (format).
            (6) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.
                                                                     Note:
                                                                        Only required for the position-consistency
                                                                        loss (position_loss_weight > 0.0).

        Example:
            Initialization:
                # Assignment of the variables.
                #   Hyperparameters (the same for all dataset sizes)
                #   obtained from optimization.
                Hyperparameters_Str = Hyperparameters.Utilities.Get_Hyperparameter_Structure()
                #   In/Out data.
                x_in  = Dataset[:, ..]
                y_out = Dataset[:, ..]

                # Initialization of the class.
                Cls = FCNN_Trainer_Cls(x=x_in, y=y_out, train_size=0.8, test_size=0.2,
                                       '../..', Robot_Str)
            Features:
                # Functions of the class.
                Cls.Compile(Hyperparameters_Str); Cls.Train(epochs=Hyperparameters_Str['epochs'],
                                                            batch_size=Hyperparameters_Str['batch_size'])
                Cls.Save()
    """

    def __init__(self, x: tp.List[tp.Union[float, int]], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str, Robot_Parameters_Str: Parameters.Robot = None) -> None:

        try:
            assert (train_size + test_size) == 1.0 and test_size > 0.0

            # Set the seed of the random number generators.
            tf.keras.utils.set_random_seed(CONST_SEED)

            # The data (History: <loss, mean square error, mean absolute error>, etc.) from the training.
            self.__train_data = None

            # The structure of the main parameters of the robot.
            self.__Robot_Parameters_Str = Robot_Parameters_Str

            # The structure of the hyperparameters.
            self.__Hyperparameters = None

            # Split the data from the dataset (x, y) into random train and validation subsets.
            self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y,
                                                                                                                                train_size=train_size, test_size=test_size,
                                                                                                                                shuffle=True, random_state=0)

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
                The files can be found in the file path specified in the input parameters of the class. The model
                itself ({file_path}.keras) is saved during the training by the checkpoint callback.
        """

        # Save the scaler parameter for input/output data.
        joblib.dump(self.__scaler_x, f'{self.__file_path}_Scaler_x.pkl')
        joblib.dump(self.__scaler_y, f'{self.__file_path}_Scaler_y.pkl')
        print(f'[INFO] The input/output scalers have been successfully saved..')
        print(f'[INFO] >> file_path = {self.__file_path}_Scaler_x.pkl')
        print(f'[INFO] >> file_path = {self.__file_path}_Scaler_y.pkl')

        # Save a summary (text) of the neural network architecture.
        #   Note:
        #       A diagram of the architecture can be created from the saved model ({file_path}.keras),
        #       see ../Evaluation/Model/visu_architecture.py
        summary = []
        self.__model.summary(print_fn=lambda line, *args, **kwargs: summary.append(line))
        with open(f'{self.__file_path}_Architecture.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary) + '\n')
        print(f'[INFO] The summary of the neural network architecture has been successfully saved.')
        print(f'[INFO] >> file_path = {self.__file_path}_Architecture.txt')

        # Save the data from the training.
        #   Note:
        #       The order of the columns is defined by the constant CONST_HISTORY_KEYS.
        if self.__train_data != None:
            # Remove the history file if it already exists.
            if os.path.isfile(f'{self.__file_path}_History.txt'):
                os.remove(f'{self.__file_path}_History.txt')

            num_of_epochs = len(self.__train_data.history['loss'])
            history = np.array([self.__train_data.history.get(key, [np.nan] * num_of_epochs) for key in CONST_HISTORY_KEYS],
                               dtype=np.float32)
            for _, data_i in enumerate(history.T):
                File_IO.Save(f'{self.__file_path}_History', data_i, 'txt', ',')
            print(f'[INFO] The training data history has been successfully saved.')
            print(f'[INFO] >> file_path = {self.__file_path}_History.txt')
            print(f'[INFO] >> columns = {CONST_HISTORY_KEYS}')

    def Compile(self, Hyperparameters: tp.Dict) -> None:
        """
        Description:
            A function to compile the model.

            Note:
                The model will be configured with the losses and metrics.

        Args:
            (1) Hyperparameters [Dictionary {..}]: The structure of the hyperparameters.
                                                    For more information about hyperparameters, see the script below:
                                                        ../Training/Hyperparameters/Utilities.py
        """

        # Store the structure of the hyperparameters.
        self.__Hyperparameters = Hyperparameters

        # Optional hyperparameters (default values).
        layer_dropout = Hyperparameters.get('layer_dropout', 0.0); activation = Hyperparameters.get('activation', 'tanh')
        position_loss_weight = Hyperparameters.get('position_loss_weight', 0.0)

        # Set the input of the FCNN model architecture.
        #   Note:
        #       The input goes straight into the first hidden layer (no additional linear layer).
        self.__model.add(tf.keras.Input(shape=(self.__x_train.shape[1], )))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, Hyperparameters['num_of_hidden_layers']):
            self.__model.add(tf.keras.layers.Dense(Hyperparameters[f'hidden_layer_{i + 1}_units'], activation=activation,
                                                   use_bias=Hyperparameters['use_bias']))
            if layer_dropout > 0.0:
                self.__model.add(tf.keras.layers.Dropout(layer_dropout))

        # Set the output layer of the FCNN model architecture.
        self.__model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation='linear',
                                               use_bias=Hyperparameters['use_bias']))

        # Set the loss function.
        #   Note:
        #       position_loss_weight = 0.0 -> Mean Squared Error (MSE)
        #       position_loss_weight > 0.0 -> Position-consistency loss
        if position_loss_weight > 0.0:
            try:
                assert self.__Robot_Parameters_Str != None
            except AssertionError as error:
                print(f'[ERROR] Information: {error}')
                print('[ERROR] The position-consistency loss requires the structure of the main parameters of the robot (Robot_Parameters_Str).')
                raise
            loss = Position_Consistency_Loss_Cls(a=self.__Robot_Parameters_Str.DH.Standard[:, 1], theta_zero=self.__Robot_Parameters_Str.DH.Standard[:, 0],
                                                 y_scale=self.__scaler_y.scale_, y_min=self.__scaler_y.min_, position_loss_weight=position_loss_weight)
            print(f'[INFO] Loss function: MSE + {position_loss_weight} * position-consistency term.')
        else:
            loss = 'mse'

        # Finally, compile the model.
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters['learning_rate']), loss=loss,
                             metrics=['mse', 'mae'])

    def Train(self, epochs: int, batch_size: int) -> None:
        """
        Description:
            A function to train the Fully-Connected Neural Network (FCNN) model.

            Note:
                The training is terminated by early stopping and the learning rate is reduced on a plateau
                of the validation loss. The patience values and the relative improvement thresholds are defined
                in the structure of the hyperparameters ('early_stopping_patience', 'early_stopping_min_delta',
                'lr_patience', 'lr_min_delta').

                The model with the lowest validation loss is saved by the checkpoint ('{file_path}.keras'),
                independent of the thresholds.

        Args:
            (1) epochs [int]: The maximum number of epochs (iterations) to train the model.
            (2) batch_size [int]: The number of samples processed before the model is updated.
        """

        # The callbacks of the training.
        #   1\ Save the best model (native Keras format).
        #   2\ Stop the training when the validation loss has stopped improving (relative threshold) and restore the best weights.
        #   3\ Reduce the learning rate when the validation loss has stopped improving (relative threshold).
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=f'{self.__file_path}.keras', monitor='val_loss',
                                                        save_best_only=True, verbose=1),
                     Relative_Early_Stopping_Cls(monitor='val_loss', patience=self.__Hyperparameters['early_stopping_patience'],
                                                 min_delta=self.__Hyperparameters['early_stopping_min_delta'], restore_best_weights=True,
                                                 verbose=1),
                     Relative_Reduce_LR_On_Plateau_Cls(monitor='val_loss', factor=0.5, patience=self.__Hyperparameters['lr_patience'],
                                                       min_delta=self.__Hyperparameters['lr_min_delta'], min_lr=1e-6, verbose=1)]

        self.__train_data = self.__model.fit(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1,
                                             validation_data=(self.__x_validation_scaled, self.__y_validation_scaled), callbacks=callbacks)

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
            (3) model_file_path [string]: The specified path to the file of the trained model (*.keras).
            (4) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

        Example:
            Initialization:
                # Initialization of the class.
                Cls = FCNN_Predictor_Cls('../..', '../..',
                                         '../..', Robot_Str)

            Features:
                # Functions of the class.
                (info, theta) = Cls.Predict([p(x, y), cfg])
                (info, theta) = Cls.Predict([p(x, y), cfg], refine_iterations=10)
    """

    def __init__(self, scaler_x_file_path: str, scaler_y_file_path: str, model_file_path: str,
                 Robot_Parameters_Str: Parameters.Robot) -> None:
        # Load the scaler parameter for input/output data.
        self.__scaler_x = joblib.load(scaler_x_file_path)
        self.__scaler_y = joblib.load(scaler_y_file_path)

        # Load the trained model from the folder.
        #   Note:
        #       The model is only used for inference, so the optimizer and loss are not needed.
        self.__model = tf.keras.models.load_model(model_file_path, compile=False)

        # The structure of the main parameters of the robot.
        self.__Robot_Parameters_Str = Robot_Parameters_Str

    def Predict(self, x: tp.List[tp.Union[float, int]], refine_iterations: int = 0, verbose: bool = True) -> tp.Tuple[tp.Dict, tp.List[float]]:
        """
        Description:
            A function to predict the absolute joint position of the robotic arm from the input
            position of the end-effector and configuration of the solution.

            Note 1:
                If refine_iterations > 0, the output of the neural network is refined using the damped
                least-squares (DLS) Jacobian method on the analytical forward kinematics and wrapped into
                the joint limits (hybrid method: NN + refinement).

            Note 2:
                A warning is displayed if any prediction is outside the joint limits, or if any queried
                position/configuration pair is not reachable within the joint limits.

        Args:
            (1) x [Vector<[float, int]> 1x3 or nx3]: Input data defined as coordinates of the x-axis, y-axis (in meters)
                                                     and the configuration of the solution.
            (2) refine_iterations [int]: The number of iterations of the refinement (0 -> pure NN prediction).
            (3) verbose [bool]: Display the warnings.

        Returns:
            (1) parameter [Dictionary {'within_limits': bool or Vector<bool> 1xn,
                                       'reachable': bool or Vector<bool> 1xn}]: Information on the results found.
                                                                                Note:
                                                                                    'within_limits': The prediction is within the joint limits.
                                                                                    'reachable': The queried position/configuration pair has an
                                                                                                 analytical IK solution within the joint limits.
            (2) parameter [Vector<float> 1xm or nxm]: Output data defined as absolute joint position of the robotic arm.
                                                      Note:
                                                        Where m is the number of joints.
        """

        x = np.asarray(x, dtype=np.float64); x_2d = x.reshape(-1, x.shape[-1])

        # Transform of data using the scale parameter.
        x_transfored = Utilities.Transform_Data_With_Scaler(self.__scaler_x, x_2d).astype(np.float32)

        # Generates output predictions from input transformed data.
        y = self.__model.predict(x_transfored, batch_size=max(1, min(x_2d.shape[0], 4096)), verbose=0)
        theta = Utilities.Inverse_Data_With_Scaler(self.__scaler_y, y).astype(np.float64)

        # Optional refinement of the prediction (hybrid method).
        if refine_iterations > 0:
            theta = Kinematics.Core.Refine_Inverse_Kinematics(theta, x_2d[:, 0:2], self.__Robot_Parameters_Str, refine_iterations)

        # Check whether the predictions are within the joint limits.
        within_limits = Kinematics.Core.Is_Within_Joint_Limits(theta, self.__Robot_Parameters_Str)
        if verbose == True and not np.all(within_limits):
            print(f'[WARNING] {np.sum(~within_limits)} of {within_limits.size} predictions are outside the joint limits.')

        # Check whether the queried position/configuration pair is reachable, i.e. whether the analytical
        # IK solution of the desired configuration is within the joint limits.
        #   Note:
        #       For an unreachable pair, the neural network still returns a prediction, but the prediction
        #       cannot reach the desired position with the desired configuration.
        (ik_info, th_ik) = Kinematics.Core.Inverse_Kinematics(x_2d[:, 0:2], self.__Robot_Parameters_Str)
        idx = np.arange(x_2d.shape[0]); cfg = np.clip(np.rint(x_2d[:, 2]).astype(np.int64), 0, th_ik.shape[1] - 1)
        reachable = Kinematics.Core.Is_Within_Joint_Limits(Kinematics.Core.Wrap_To_Joint_Limits(th_ik[idx, cfg], self.__Robot_Parameters_Str),
                                                           self.__Robot_Parameters_Str) & (ik_info['error'][idx, cfg] < CONST_REACHABILITY_TOLERANCE)
        if verbose == True and not np.all(reachable):
            print(f'[WARNING] {np.sum(~reachable)} of {reachable.size} queried position/configuration pairs are not reachable within the joint limits.')

        if x.ndim == 1:
            return ({'within_limits': bool(within_limits[0]), 'reachable': bool(reachable[0])}, theta[0])
        else:
            return ({'within_limits': within_limits, 'reachable': reachable}, theta)

class FCNN_Optimizer_Cls(object):
    """
    Description:
        A specific class for hyperparameter optimization of the Fully-Connected Neural Network (FCNN).

        The objective of the optimization is the validation loss (Mean Squared Error, MSE), which is minimized.

        Search space:
            num_of_hidden_layers: 2 .. 5
            hidden_layer_{i}_units: 64 .. 512 (step 64)
            layer_dropout: {0.0, 0.05}
            learning_rate: {1e-2, 1e-3, 5e-4}
            activation: {tanh, gelu, swish}
            use_bias: True (fixed)

        Reference:
            On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice, Li Yang and Abdallah Shami
                https://arxiv.org/abs/2007.15745

    Initialization of the Class:
        Args:
            (1) x [Vector<[float, int[]> nxm]: Input data.
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
                Cls.Optimize(num_of_trials=30, epochs_per_trial=400, batch_size=128,
                             early_stopping_patience=100, early_stopping_min_delta=0.01,
                             lr_patience=20, lr_min_delta=0.001, num_of_initial_points=10,
                             save_results=True)
    """

    def __init__(self, x: tp.List[tp.Union[float, int]], y: tp.List[float], train_size: float, test_size: float,
                 file_path: str) -> None:
        try:
            assert (train_size + test_size) == 1.0 and test_size > 0.0

            # Set the seed of the random number generators.
            tf.keras.utils.set_random_seed(CONST_SEED)

            # Split the data from the dataset (x, y) into random train and validation subsets.
            self.__x_train, self.__x_validation, self.__y_train, self.__y_validation = sklearn.model_selection.train_test_split(x, y,
                                                                                                                                train_size=train_size, test_size=test_size,
                                                                                                                                shuffle=True, random_state=0)

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

    def __Compile(self, Hyperparameters: kt.HyperParameters) -> tf.keras.Sequential:
        """
        Description:
            A function to compile the model.

            Note:
                The function will be called for each optimization trial.

        Args:
            (1) Hyperparameters [kt.HyperParameters(object)]: The structure of the hyperparameters to be optimized.

        Returns:
            (1) parameter [tf.keras.Sequential]: A model grouping layers into an object with training and inference features.
        """

        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Defined general hyperparameters to be changed.
        #   Note:
        #       Other parameters are defined within each layer.
        use_bias = Hyperparameters.Fixed('use_bias', value=True)
        layer_dropout = Hyperparameters.Choice('layer_dropout', values=[0.0, 0.05])
        activation = Hyperparameters.Choice('activation', values=['tanh', 'gelu', 'swish'])

        # Set the input of the FCNN model architecture.
        #   Note:
        #       The input goes straight into the first hidden layer (no additional linear layer).
        model.add(tf.keras.Input(shape=(self.__x_train.shape[1], )))

        # Set the hidden layers of the FCNN model architecture.
        for i in range(0, Hyperparameters.Int('num_of_hidden_layers', min_value=2, max_value=5, step=1)):
            model.add(tf.keras.layers.Dense(Hyperparameters.Int(f'hidden_layer_{i + 1}_units', min_value=64, max_value=512, step=64),
                                            activation=activation, use_bias=use_bias))
            if layer_dropout > 0.0:
                model.add(tf.keras.layers.Dropout(layer_dropout))

        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(self.__y_train.shape[1], activation='linear',
                                        use_bias=use_bias))

        # Finally, compile the model.
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 5e-4])),
                      loss='mse', metrics=['mse', 'mae'])

        return model

    def __Print_Top_Trials(self, optimizer: kt.BayesianOptimization, num_of_top_trials: int) -> None:
        """
        Description:
            A function to display the best trials of the optimization, ranked by the objective (val_loss),
            together with the number of trainable parameters of each configuration.

        Args:
            (1) optimizer [kt.BayesianOptimization(object)]: The tuner after the search.
            (2) num_of_top_trials [int]: The number of the best trials to be displayed.
        """

        print(f'[INFO] The top {num_of_top_trials} configurations (objective = val_loss, minimized):')
        for i, trial_i in enumerate(optimizer.oracle.get_best_trials(num_trials=num_of_top_trials)):
            hps_i = trial_i.hyperparameters.values
            units_i = [hps_i[f'hidden_layer_{j + 1}_units'] for j in range(hps_i['num_of_hidden_layers'])]
            # Build the model of the trial to count its parameters.
            num_of_parameters = optimizer.hypermodel.build(trial_i.hyperparameters).count_params()
            print(f'[INFO]  {i + 1}. val_loss = {trial_i.score:.6e} | parameters = {num_of_parameters:7d} | '
                  f'units = {units_i} | activation = {hps_i["activation"]} | dropout = {hps_i["layer_dropout"]} | '
                  f'learning_rate = {hps_i["learning_rate"]} | best epoch = {trial_i.best_step + 1}')

    def Optimize(self, num_of_trials: int, epochs_per_trial: int, batch_size: int, early_stopping_patience: int,
                 early_stopping_min_delta: float, lr_patience: int, lr_min_delta: float, num_of_initial_points: int,
                 save_results: bool, num_of_top_trials: int = 5) -> None:
        """
        Description:
            A function to optimize the Fully-Connected Neural Network (FCNN) model.

            Note:
                The trials use the same callbacks as the training (FCNN_Trainer_Cls.Train()), i.e. early stopping
                and the reduction of the learning rate (factor 0.5, min. 1e-6) with relative improvement thresholds,
                only with shorter patience values. The tuner always starts a new search (overwrite=True).

        Args:
            (1) num_of_trials [int]: The number of trials to be used in hyperparameter optimization.
            (2) epochs_per_trial [int]: The maximum number of epochs (iterations) to train the model per trial.
            (3) batch_size [int]: The number of samples processed before the model is updated.
            (4) early_stopping_patience [int]: The number of epochs without improvement of 'val_loss' after
                                               which the trial is stopped.
            (5) early_stopping_min_delta [float]: The minimum relative improvement of 'val_loss' for early stopping.
            (6) lr_patience [int]: The number of epochs without improvement of 'val_loss' after which the learning
                                   rate is reduced by a factor of 0.5.
            (7) lr_min_delta [float]: The minimum relative improvement of 'val_loss' for the learning-rate reduction.
            (8) num_of_initial_points [int]: The number of randomly sampled trials before the Bayesian optimization
                                             (Gaussian process) starts to select the configurations.
                                                Note:
                                                    The KerasTuner default is 3 * (number of hyperparameters).
            (9) save_results [bool]: Information about whether the optimization results will be saved.
            (10) num_of_top_trials [int]: The number of the best configurations to be displayed.
        """

        # The directory of the tuner (removed and re-created by overwrite=True).
        directory_path = f'{CONST_PROJECT_FOLDER}/Training/FCNN_Inverse_Kinematics_Optimizer'

        # Bayesian optimization with Gaussian process over the desired hyperparameters.
        optimizer = kt.BayesianOptimization(hypermodel=self.__Compile, objective=kt.Objective(name='val_loss', direction='min'),
                                            max_trials=num_of_trials, num_initial_points=num_of_initial_points, seed=CONST_SEED,
                                            directory=directory_path, project_name='FCNN_Inverse_Kinematics', overwrite=True)

        # The callbacks of each trial (the same as in the training, see FCNN_Trainer_Cls.Train()).
        #   Note:
        #       KerasTuner makes a deep copy of the callbacks for each trial, so their state is reset.
        callbacks = [Relative_Early_Stopping_Cls(monitor='val_loss', patience=early_stopping_patience,
                                                 min_delta=early_stopping_min_delta, restore_best_weights=True),
                     Relative_Reduce_LR_On_Plateau_Cls(monitor='val_loss', factor=0.5, patience=lr_patience,
                                                       min_delta=lr_min_delta, min_lr=1e-6)]

        # Start the search for the most suitable model.
        optimizer.search(self.__x_train_scaled, self.__y_train_scaled, epochs=epochs_per_trial, batch_size=batch_size,
                         validation_data=(self.__x_validation_scaled, self.__y_validation_scaled),
                         callbacks=callbacks)

        # Display the best configurations.
        self.__Print_Top_Trials(optimizer, num_of_top_trials)

        # Get the best hyperparameters determined by the objective function.
        #   objective = 'val_loss' (minimized)
        best_hps = optimizer.get_best_hyperparameters(num_trials=1)[0]

        # Save the results of the best parameters along with the score.
        if save_results == True:
            best_values = dict(best_hps.values)
            #   Note:
            #       Remove the units of the hidden layers which are not used by the best model.
            for i in range(best_values['num_of_hidden_layers'], 5):
                best_values.pop(f'hidden_layer_{i + 1}_units', None)
            best_values['val_loss'] = optimizer.oracle.get_best_trials(num_trials=1)[0].score
            #   Note:
            #       The amount of data used for the optimization. The results apply to the single hyperparameter
            #       structure used for all dataset sizes (../Training/Hyperparameters/Utilities.py).
            best_values['num_of_data'] = self.__x_train.shape[0] + self.__x_validation.shape[0]
            self.__Save(best_values)

        # Release GPU resources when the optimization process is already complete.
        self.__Release()
