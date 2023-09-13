# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf

"""
Desription:
    Neural network input hyperparameters for model tuning.

    Note:
        More information can be found in the script (tune.py or Model.py).
"""
HYPERPARAMETERS_TUNER = {
    'epochs':[100],
    'batch_size': [32, 64],
    'architecture': [tf.keras.layers.SimpleRNN],
    'hidden_layers_w_d': [1, 2],
    'hidden_layers_wo_d': [0, 1, 2],
    'in_layer_units': [32, 64, 128],
    'hidden_layer_w_d_units': [32, 64, 128],
    'hidden_layer_wo_d_units': [32, 64, 128],
    'in_layer_activation': ['tanh'],
    'kernel_layer_activation': ['sigmoid'],
    'layer_drop': [0.05, 0.10],
    'opt': [tf.keras.optimizers.Adam],
    'opt_learning_rate': [1e-3, 1e-2]
}

"""
Desription:
    Neural network input hyperparameters for model training.

    Note:
        More information can be found in the script (train.py or Model.py).
"""
HYPERPARAMETERS_SIMPLE_RNN = {
    'architecture': tf.keras.layers.Dense,
    'hidden_layers_w_d': 1,
    'hidden_layers_wo_d': 1,
    'in_layer_units': 128,
    'hidden_layer_w_d_units': 128,
    'hidden_layer_wo_d_units': 128,
    'in_layer_activation': 'tanh',
    'kernel_layer_activation': 'sigmoid',
    'layer_drop': 0.05,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-2
}

HYPERPARAMETERS_LSTM = {
    'architecture': tf.keras.layers.LSTM,
    'hidden_layers_w_d': 1,
    'hidden_layers_wo_d': 1,
    'in_layer_units': 128,
    'hidden_layer_w_d_units': 128,
    'hidden_layer_wo_d_units': 128,
    'in_layer_activation': 'tanh',
    'kernel_layer_activation': 'sigmoid',
    'layer_drop': 0.05,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-2
}

HYPERPARAMETERS_GRU = {
    'architecture': tf.keras.layers.GRU,
    'hidden_layers_w_d': 1,
    'hidden_layers_wo_d': 1,
    'in_layer_units': 128,
    'hidden_layer_w_d_units': 128,
    'hidden_layer_wo_d_units': 128,
    'in_layer_activation': 'tanh',
    'kernel_layer_activation': 'sigmoid',
    'layer_drop': 0.05,
    'opt': tf.keras.optimizers.Adam,
    'opt_learning_rate': 1e-2
}