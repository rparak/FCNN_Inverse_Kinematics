# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf


"""
Desription:
    ...
"""
DCNN_HYPERPARAMETERS_TUNER_METHOD_0 = {
    'epochs': [1000],
    'batch_size': [32, 64],
    'architecture': [tf.keras.layers.Dense],
    'hidden_layers': [2, 3, 4, 5],
    'in_layer_units': [32, 64, 128, 256],
    'hidden_layer_units': [32, 64, 128, 256],
    'in_layer_activation': ['relu', 'tanh'],
    'kernel_layer_activation': ['relu', 'tanh'],
    'opt': [tf.keras.optimizers.Adam],
    'opt_learning_rate': [1e-03]
}
DCNN_HYPERPARAMETERS_TUNER_METHOD_1 = {
    'epochs': [1000],
    'batch_size': [32, 64],
    'architecture': [tf.keras.layers.Dense],
    'hidden_layers_w_d': [0, 1, 2, 3],
    'hidden_layers_wo_d': [0, 1, 2, 3],
    'in_layer_units': [32, 64, 128, 256],
    'hidden_layer_w_d_units': [32, 64, 128, 256],
    'hidden_layer_wo_d_units': [32, 64, 128, 256],
    'in_layer_activation': ['relu', 'tanh'],
    'kernel_layer_activation': ['relu', 'tanh'],
    'layer_drop': [0.01, 0.025, 0.05],
    'opt': [tf.keras.optimizers.Adam],
    'opt_learning_rate': [1e-03]
}

"""
Desription:
    ...
"""
DCNN_HYPERPARAMETERS_TRAINER_METHOD_0 = {
    'architecture': tf.keras.layers.Dense,
    'hidden_layers': 5,
    'in_layer_units': 32,
    'hidden_layer_units': [64, 128, 256, 128, 64],
    'in_layer_activation': 'tanh',
    'kernel_layer_activation': 'tanh',
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-03
}
DCNN_HYPERPARAMETERS_TRAINER_METHOD_1 = {
    'architecture': tf.keras.layers.Dense,
    'hidden_layers_w_d': 5,
    'hidden_layers_wo_d': 0,
    'in_layer_units': 32,
    'hidden_layer_w_d_units': [64, 128, 256, 128, 64],
    'hidden_layer_wo_d_units': [],
    'in_layer_activation': 'tanh',
    'kernel_layer_activation': 'tanh',
    'layer_drop': 0.01,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-03
}