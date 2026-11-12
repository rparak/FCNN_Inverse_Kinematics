
# Typing (Support for type hints)
import typing as tp

"""
Desription:
    Hyperparameter structure for training an Inverse Kinematics (IK) task
    using a Fully-Connected Neural Network (FCNN).
"""

def Get_Hyperparameter_Structure() -> tp.Dict:
    """
    Description:
        Get the hyperparameter structure to train the inverse kinematics (IK) task
        using a fully connected neural network (FCNN).

        The same architecture and the same training hyperparameters are used for all dataset
        sizes (N = 1000, 10000, 100000), so the amount of data is the only variable.

        To optimize the hyperparameter structure of a Fully-Connected
        Neural Network (FCNN), see the following program:
            ../Training/optimize.py

        Note:
            The architecture, activation, dropout and learning rate come from a 30-trial Bayesian
            search (KerasTuner, objective val_loss) on N = 10000 with 400 epochs per trial
            (../Training/optimize.py), see ../Training/Hyperparameters/FCNN_IK_Optimizer_Best_Results.txt
            (best val_loss = 2.098e-06). The batch size, epochs, patience values and relative thresholds
            are not part of the search.

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

    Returns:
        (1) parameter [Dictionary {..}]: The structure of the hyperparameters.
    """

    # Results of the hyperparameter optimization (30-trial Bayesian search on N = 10000, 400 epochs per trial).
    return {'use_bias': True,
            'layer_dropout': 0.0,
            'activation': 'gelu',
            'num_of_hidden_layers': 5,
            'hidden_layer_1_units': 384,
            'hidden_layer_2_units': 448,
            'hidden_layer_3_units': 512,
            'hidden_layer_4_units': 384,
            'hidden_layer_5_units': 128,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 10000,
            'early_stopping_patience': 500,
            'early_stopping_min_delta': 0.01,
            'lr_patience': 100,
            'lr_min_delta': 0.001,
            'position_loss_weight': 0.0
        }
