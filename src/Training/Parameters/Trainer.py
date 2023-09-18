# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf

"""
Desription:
    ...
"""
FCNN_HYPERPARAMETERS_METHOD_0 = {
    'architecture': tf.keras.layers.Dense,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'hidden_layers': 5,
    'hidden_layer_units': [64, 128, 256, 128, 64],
    'kernel_layer_activation': 'tanh',
    'use_bias': False,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-03
}
FCNN_HYPERPARAMETERS_METHOD_1 = {
    'architecture': tf.keras.layers.Dense,
    'in_layer_units': 32,
    'in_layer_activation': 'tanh',
    'hidden_layers_w_d': 5,
    'hidden_layers_wo_d': 0,
    'hidden_layer_w_d_units': [64, 128, 256, 128, 64],
    'hidden_layer_wo_d_units': [],
    'kernel_layer_activation': 'tanh',
    'layer_drop': 0.01,
    'use_bias': False,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-03
}