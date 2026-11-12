
# Typing (Support for type hints)
import typing as tp

"""
Desription:
    Hyperparameter structure for training an Inverse Kinematics (IK) task
    using a Fully-Connected Neural Network (FCNN).
"""

def Get_Hyperparameter_Structure(N: int) -> tp.Dict:
    """
    Description:
        Get the hyperparameter structure to train the inverse kinematics (IK) task
        using a fully connected neural network (FCNN).

        To optimize the hyperparameter structure of a Fully-Connected
        Neural Network (FCNN), see the following program:
            ../Training/optimize.py

        Note:
            The values below are DEFAULTS, to be replaced by the results of the hyperparameter
            optimization (../Training/optimize.py).

            The structure of the hyperparameters:
                'use_bias': Use the bias vector in the layers.
                'layer_dropout': The dropout rate (0.0 -> no dropout layer).
                'activation': The activation function of the hidden layers ('tanh', 'gelu', 'swish').
                'num_of_hidden_layers': The number of hidden layers.
                'hidden_layer_{i}_units': The number of units of the i-th hidden layer.
                'learning_rate': The initial learning rate of the Adam optimizer.
                'batch_size': The number of samples processed before the model is updated.
                'epochs': The maximum number of epochs (training is terminated by early stopping).
                'early_stopping_patience': The number of epochs without improvement of 'val_loss'
                                           after which the training is stopped.
                'early_stopping_min_delta': The minimum relative improvement of 'val_loss' (fraction of
                                            the best value) for early stopping, e.g. 0.01 -> 1 %.
                'lr_patience': The number of epochs without improvement of 'val_loss' after which
                               the learning rate is reduced by a factor of 0.5.
                'lr_min_delta': The minimum relative improvement of 'val_loss' (fraction of the best
                                value) for the learning-rate reduction, e.g. 0.001 -> 0.1 %.
                'position_loss_weight': The weight (lambda) of the position-consistency loss
                                        (0.0 -> plain MSE loss).

    Args:
        (1) N [int]: The amount of data generated to train the model.

    Returns:
        (1) parameter [Dictionary {..}]: The structure of the hyperparameters containing the amount of
                                         data generated to train the model.
    """

    return {
        1000: {'use_bias': True,
               'layer_dropout': 0.0,
               'activation': 'tanh',
               'num_of_hidden_layers': 3,
               'hidden_layer_1_units': 64,
               'hidden_layer_2_units': 128,
               'hidden_layer_3_units': 64,
               'learning_rate': 0.001,
               'batch_size': 64,
               'epochs': 10000,
               'early_stopping_patience': 1000,
               'early_stopping_min_delta': 0.01,
               'lr_patience': 200,
               'lr_min_delta': 0.001,
               'position_loss_weight': 0.0
            },
        10000: {'use_bias': True,
                'layer_dropout': 0.0,
                'activation': 'tanh',
                'num_of_hidden_layers': 4,
                'hidden_layer_1_units': 128,
                'hidden_layer_2_units': 256,
                'hidden_layer_3_units': 256,
                'hidden_layer_4_units': 128,
                'learning_rate': 0.001,
                'batch_size': 128,
                'epochs': 10000,
                'early_stopping_patience': 500,
                'early_stopping_min_delta': 0.01,
                'lr_patience': 100,
                'lr_min_delta': 0.001,
                'position_loss_weight': 0.0
            },
        100000: {'use_bias': True,
                 'layer_dropout': 0.0,
                 'activation': 'tanh',
                 'num_of_hidden_layers': 4,
                 'hidden_layer_1_units': 256,
                 'hidden_layer_2_units': 512,
                 'hidden_layer_3_units': 512,
                 'hidden_layer_4_units': 256,
                 'learning_rate': 0.001,
                 'batch_size': 256,
                 'epochs': 10000,
                 'early_stopping_patience': 300,
                 'early_stopping_min_delta': 0.01,
                 'lr_patience': 50,
                 'lr_min_delta': 0.001,
                 'position_loss_weight': 0.0
            }
    }[N]
