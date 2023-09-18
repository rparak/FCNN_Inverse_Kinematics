# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf

"""
Desription:
    ...
"""
FCNN_HYPERPARAMETERS_METHOD_0 = {
    'in_layer_units': [32, 64, 128, 256],
    'in_layer_activation': ['relu', 'tanh'],
    'hidden_layers': [3, 4, 5, 6],
    'hidden_layer_units': [32, 64, 128, 256],
    'kernel_layer_activation': ['relu', 'tanh'],
    'use_bias': [False, True]
}
FCNN_HYPERPARAMETERS_METHOD_1 = {
    'in_layer_units': [32, 64, 128, 256],
    'in_layer_activation': ['relu', 'tanh'],
    'hidden_layers_w_d': [0, 1, 2, 3],
    'hidden_layers_wo_d': [0, 1, 2, 3],
    'hidden_layer_w_d_units': [32, 64, 128, 256],
    'hidden_layer_wo_d_units': [32, 64, 128, 256],
    'kernel_layer_activation': ['relu', 'tanh'],
    'layer_drop': [0.01, 0.025, 0.05],
    'use_bias': [False, True],
    'opt': [tf.keras.optimizers.Adam],
    'opt_learning_rate': [1e-03]
}