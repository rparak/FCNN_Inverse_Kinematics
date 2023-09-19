from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import keras_tuner
import numpy as np
# https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f
# for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
gpu_arr = tf.config.experimental.list_physical_devices('GPU')
if gpu_arr:
    for _, gpu_i in enumerate(gpu_arr):
        tf.config.experimental.set_memory_growth(gpu_i, True)
else:
    print('[INFO] No GPU device was found.')

def build_model(hp):
        # Initialization of a sequential neural network model.
        model = tf.keras.models.Sequential()

        # Defined general hyperparameters to be changed.
        #   Note:
        #       Other parameters are defined within each layer.
        use_bias = hp.Choice('use_bias', values=[False, True])

        # Set the input layer of the FCNN model architecture.
        model.add(layers.Flatten())

        # Set the output layer of the FCNN model architecture.
        model.add(tf.keras.layers.Dense(10, activation=hp.Choice('out_layer_activation', values=['relu', 'tanh']), 
                                        use_bias=use_bias))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])), 
                      loss="categorical_crossentropy", metrics=["accuracy"],)

        return model

#build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.BayesianOptimization(
    hypermodel=build_model,
    objective=keras_tuner.Objective(name="accuracy", direction="max"),
    max_trials=2,
    directory="my_dir",
    project_name="helloworld",
)

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#tuner.search(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val))
tuner.search(x_train, y_train, epochs=1, batch_size=64, validation_data=None)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#print(type(best_hps.values))
#print(best_hps.get('units'))
val = best_hps.values

tuner.results_summary(num_trials=1)

with open('myfile', 'w') as f:
    for _, (key, value) in enumerate(val.items()):
        f.write(f'{key}: {value}\n')

keras.backend.clear_session()