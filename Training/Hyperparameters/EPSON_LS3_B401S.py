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