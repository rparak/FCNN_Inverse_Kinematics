
# Typing (Support for type hints)
import typing as tp

"""
Desription:
    Hyperparameter structure for training an Inverse Kinematics (IK) task 
    using a Fully-Connected Neural Network (FCNN).
"""

FCNN_HPS = {
    'use_bias': False,
    'layer_dropout': 0.10,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 4,
    'hidden_layer_1_units': 128,
    'hidden_layer_2_units': 64,
    'hidden_layer_3_units': 32,
    'hidden_layer_4_units': 16,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

def Get_Hyperparameter_Structure(N: int) -> tp.Dict:
    """
    Description:
        Get the hyperparameter structure to train the inverse kinematics (IK) task 
        using a fully connected neural network (FCNN).

        To optimize the hyperparameter structure of a Fully-Connected 
        Neural Network (FCNN), see the following program:
            ../Training/optimize.py

    Args:
        (1) N [int]: The amount of data generated to train the model.

    Returns:
        (1) parameter [int]: The structure of the hyperparameters containing the amount of 
                             data generated to train the model.
    """

    return {
        1000: {'use_bias': False,
               'layer_dropout': 0.10,
               'in_layer_activation': 'tanh',
               'num_of_hidden_layers': 4,
               'hidden_layer_1_units': 128,
               'hidden_layer_2_units': 64,
               'hidden_layer_3_units': 32,
               'hidden_layer_4_units': 16,
               'hidden_layer_activation': 'tanh',
               'out_layer_activation': 'tanh',
               'learning_rate': 0.001
            },
        10000: {'use_bias': False,
                'layer_dropout': 0.10,
                'in_layer_activation': 'tanh',
                'num_of_hidden_layers': 4,
                'hidden_layer_1_units': 128,
                'hidden_layer_2_units': 64,
                'hidden_layer_3_units': 32,
                'hidden_layer_4_units': 16,
                'hidden_layer_activation': 'tanh',
                'out_layer_activation': 'tanh',
                'learning_rate': 0.001
            },  
        100000: {'use_bias': False,
                 'layer_dropout': 0.10,
                 'in_layer_activation': 'tanh',
                 'num_of_hidden_layers': 4,
                 'hidden_layer_1_units': 128,
                 'hidden_layer_2_units': 64,
                 'hidden_layer_3_units': 32,
                 'hidden_layer_4_units': 16,
                 'hidden_layer_activation': 'tanh',
                 'out_layer_activation': 'tanh',
                 'learning_rate': 0.001
            }
    }[N]