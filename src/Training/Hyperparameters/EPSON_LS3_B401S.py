"""
Desription:
    Hyperparameter structure for training an Inverse Kinematics (IK) task 
    using a Fully-Connected Neural Network (FCNN).
"""

# Method 0: 
#   No test (validation) partition.
FCNN_HPS_METHOD_0 = {
    'use_bias': False,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 5,
    'hidden_layer_1_units': 160,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 224,
    'hidden_layer_5_units': 224,
    'hidden_layer_activation': 'relu',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 1: 
#   With test (validation) partition.
FCNN_HPS_METHOD_1 = {
    'use_bias': False,
    'layer_dropout': 0.03,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 3,
    'hidden_layer_1_units': 160,
    'hidden_layer_2_units': 224,
    'hidden_layer_3_units': 256,
    'hidden_layer_activation': 'relu',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}