"""
Desription:
    Hyperparameter structure for training an Inverse Kinematics (IK) task 
    using a Fully-Connected Neural Network (FCNN).
"""

# Method 0: 
#   No test (validation) partition.
# Number of generated data:
#   N = 1000
FCNN_HPS_METHOD_0_N_1000 = {
    'use_bias': False,
    'in_layer_units': 64,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 6,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 0: 
#   No test (validation) partition.
# Number of generated data:
#   N = 10000
FCNN_HPS_METHOD_0_N_10000 = {
    'use_bias': False,
    'in_layer_units': 64,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 6,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 0: 
#   No test (validation) partition.
# Number of generated data:
#   N = 100000
FCNN_HPS_METHOD_0_N_100000 = {
    'use_bias': False,
    'in_layer_units': 64,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 6,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 1: 
#   With test (validation) partition.
# Number of generated data:
#   N = 1000
FCNN_HPS_METHOD_1_N_1000 = {
    'use_bias': False,
    'layer_dropout': 0.025,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 4,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 1: 
#   With test (validation) partition.
# Number of generated data:
#   N = 10000
FCNN_HPS_METHOD_1_N_10000 = {
    'use_bias': False,
    'layer_dropout': 0.025,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 4,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}

# Method 1: 
#   With test (validation) partition.
# Number of generated data:
#   N = 100000
FCNN_HPS_METHOD_1_N_100000 = {
    'use_bias': False,
    'layer_dropout': 0.025,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'num_of_hidden_layers': 4,
    'hidden_layer_1_units': 256,
    'hidden_layer_2_units': 256,
    'hidden_layer_3_units': 256,
    'hidden_layer_4_units': 256,
    'hidden_layer_5_units': 256,
    'hidden_layer_6_units': 256,
    'hidden_layer_activation': 'tanh',
    'out_layer_activation': 'tanh',
    'learning_rate': 0.001
}