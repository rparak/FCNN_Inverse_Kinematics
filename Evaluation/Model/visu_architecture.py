# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# Tensorflow (Machine learning) [pip3 install tensorflow]
import tensorflow as tf
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import matplotlib.patches
# Custom Lib.:
#   ../FCNN_IK/Model
#       Note:
#           The import registers the custom loss function required to load the model.
import FCNN_IK.Model

"""
Description:
    Initialization of constants.
"""
# The models (dataset configurations) to be visualized.
CONST_NUM_OF_DATA = [1000, 10000, 100000]
# The names of the input and output quantities of the neural network.
CONST_INPUT_NAMES = 'x, y, cfg'
CONST_OUTPUT_NAMES = 'θ₁, θ₂'
# Colors of the diagram.
#   Input/output layers, hidden layers, text and arrows.
CONST_COLOR_IO = '#8d8d8d'; CONST_COLOR_HIDDEN = '#3d85c6'; CONST_COLOR_TEXT = '#262626'
# Save the figures to the files (*.svg) for the README.
#   Note:
#       f'{project_folder}/images/Architecture_N_{N}.svg'
CONST_SAVE_FIGURE = False

def Get_Layers(model: tf.keras.Model) -> list:
    """
    Description:
        Get the description of the individual layers of the model used to draw the diagram.

        Note:
            The dropout rate is assigned to the previous dense layer.

    Args:
        (1) model [tf.keras.Model]: The loaded model of the neural network.

    Returns:
        (1) parameter [List<Dictionary {'units': int, 'activation': string, 'dropout': float,
                                        'num_of_parameters': int}>]: The description of the dense layers.
    """

    layers = []
    for _, layer_i in enumerate(model.layers):
        if isinstance(layer_i, tf.keras.layers.Dense):
            layers.append({'units': layer_i.units, 'activation': layer_i.get_config()['activation'], 'dropout': 0.0,
                           'num_of_parameters': int(np.sum([np.prod(w.shape) for w in layer_i.trainable_weights]))})
        elif isinstance(layer_i, tf.keras.layers.Dropout) and len(layers) > 0:
            layers[-1]['dropout'] = layer_i.rate

    return layers

def main():
    """
    Description:
        A program to visualize the architecture of the trained neural networks as a layer diagram.

        The diagram shows the input (x, y, cfg), the dense layers with their number of units and
        activation function, the output (θ₁, θ₂) and the number of trainable parameters.

        Note:
            The figure is saved as a vector graphic (*.svg) with the text kept as text, so it remains
            small and sharp in the README.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Set the parameters for the scientific style.
    #   Note:
    #       LaTeX is disabled and the text is stored as text (not as paths) to keep the vector graphic small.
    plt.style.use('science')
    plt.rcParams.update({'text.usetex': False, 'svg.fonttype': 'none', 'font.family': 'sans-serif'})

    for _, N_i in enumerate(CONST_NUM_OF_DATA):
        file_path = f'{project_folder}/Data/Model/Config_N_{N_i}'
        if not os.path.isfile(f'{file_path}.keras'):
            print(f'[WARNING] The model {file_path}.keras does not exist. The model is skipped.')
            continue

        # Load the model and get the description of the layers.
        model = tf.keras.models.load_model(f'{file_path}.keras', compile=False)
        layers = Get_Layers(model)
        num_of_inputs = model.inputs[0].shape[-1]
        num_of_parameters = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))

        # The blocks of the diagram: input, dense layers (the hidden layers and the last one is the output layer).
        blocks = [{'title': 'Input', 'lines': [f'{num_of_inputs} values', f'({CONST_INPUT_NAMES})'], 'color': CONST_COLOR_IO, 'units': num_of_inputs}]
        for i, layer_i in enumerate(layers):
            if i == len(layers) - 1:
                title = 'Output'; color = CONST_COLOR_IO
            else:
                title = f'Hidden {i + 1}'; color = CONST_COLOR_HIDDEN
            lines = [f'{layer_i["units"]} units', layer_i['activation']]
            if i == len(layers) - 1:
                lines.append(f'({CONST_OUTPUT_NAMES})')
            if layer_i['dropout'] > 0.0:
                lines.append(f'dropout {layer_i["dropout"]:g}')
            blocks.append({'title': title, 'lines': lines, 'color': color, 'units': layer_i['units'],
                           'params': layer_i['num_of_parameters']})

        # Create a figure.
        #   Note:
        #       The height of a block is proportional to the logarithm of the number of units.
        (width, gap) = (1.6, 0.55)
        max_height = 3.0; min_height = 1.9
        log_units = np.log2([b['units'] for b in blocks]); log_max = max(np.max(log_units), 1.0)
        figure, ax = plt.subplots(figsize=(1.35 * len(blocks) + 0.6, 3.4))

        for i, block_i in enumerate(blocks):
            height = min_height + (max_height - min_height) * log_units[i] / log_max
            x_i = i * (width + gap)
            # Draw the block of the layer.
            ax.add_patch(matplotlib.patches.FancyBboxPatch((x_i, -height / 2.0), width, height, boxstyle='round,pad=0.0,rounding_size=0.12',
                                                           facecolor=block_i['color'], edgecolor='none', alpha=0.14))
            ax.add_patch(matplotlib.patches.FancyBboxPatch((x_i, -height / 2.0), width, height, boxstyle='round,pad=0.0,rounding_size=0.12',
                                                           facecolor='none', edgecolor=block_i['color'], linewidth=1.2))
            # Text of the block.
            ax.text(x_i + width / 2.0, 0.32, block_i['title'], ha='center', va='center', fontsize=9.5, fontweight='bold', color=CONST_COLOR_TEXT)
            ax.text(x_i + width / 2.0, -0.05, '\n'.join(block_i['lines']), ha='center', va='top', fontsize=8.5, linespacing=1.35,
                    color=CONST_COLOR_TEXT)
            # The number of trainable parameters of the layer.
            if 'params' in block_i:
                ax.text(x_i + width / 2.0, -max_height / 2.0 - 0.25, f'{block_i["params"]:,} param.'.replace(',', ' '), ha='center', va='top',
                        fontsize=8.0, color='#595959')
            # Arrow to the next block.
            if i < len(blocks) - 1:
                ax.annotate('', xy=(x_i + width + gap - 0.06, 0.0), xytext=(x_i + width + 0.06, 0.0),
                            arrowprops={'arrowstyle': '-|>', 'color': '#595959', 'linewidth': 1.0, 'shrinkA': 0, 'shrinkB': 0})

        # Set parameters of the graph (plot).
        ax.set_title(f'Neural-Network Architecture (N = {N_i}): {num_of_parameters:,} trainable parameters'.replace(',', ' '),
                     fontsize=11, pad=8.0, color=CONST_COLOR_TEXT)
        ax.set_xlim(-0.15, len(blocks) * (width + gap) - gap + 0.15)
        ax.set_ylim(-max_height / 2.0 - 0.7, max_height / 2.0 + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Display information.
        print(f'[INFO] Model: Config_N_{N_i}')
        for _, block_i in enumerate(blocks):
            print(f'[INFO]  {block_i["title"]:<14}: {", ".join(block_i["lines"])}' + (f', {block_i["params"]} param.' if 'params' in block_i else ''))
        print(f'[INFO]  Trainable parameters: {num_of_parameters}')

        # Save the figure.
        if CONST_SAVE_FIGURE == True:
            figure.savefig(f'{project_folder}/images/Architecture_N_{N_i}.svg', format='svg', bbox_inches='tight')
            print(f'[INFO] >> file_path = {project_folder}/images/Architecture_N_{N_i}.svg')

        # Show the result.
        plt.show()

        # Release class object.
        del model

if __name__ == "__main__":
    sys.exit(main())
