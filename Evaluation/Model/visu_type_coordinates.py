# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Parameters/Robot
import Parameters.Robot
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics
#   ../Kinematics/Core
import Kinematics.Core
#   ../FCNN_IK/Model
import FCNN_IK.Model


"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# The configuration ID of the inverse kinematics (IK) solution.
CONST_IK_CONFIGURATION = 0
# Number of data (x, y coordinates) to be generated.
CONST_NUM_OF_DATA = 10

def main():
    """
    Description:
        The program to visualize both the desired and predicted coordinates of the robot's end-effector 
        using an inverse kinematics neural-network predictor.
        
        The comparison is tested on multiple randomly generated coordinates calculated from reachable 
        points using forward kinematics.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE
    
    # The tolerance of the data.
    tolerance = 4
    
    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0; data = []; tolerance = 4
    while CONST_NUM_OF_DATA > i:
        # Random generation of absolute joint orientations.
        #   Note:
        #       The boundaries of the random generation are defined in the object structure.
        theta_rand = np.random.uniform(Robot_Str.Theta.Limit[:, 0], Robot_Str.Theta.Limit[:, 1])

        # Obtain the x, y coordinates using forward kinematics.
        p_tmp = np.round(Kinematics.Core.Forward_Kinematics(theta_rand, Robot_Str)[1], tolerance).astype('float32')

        # If there is a duplicate of the input data, skip to the next step.
        if data != []:
            for _, x_i in enumerate(data):
                if p_tmp in x_i[0:2]:
                    continue

        # Store the acquired data.
        data.append(p_tmp)

        i += 1

    data_predicted = []; e_p = []
    for _, N_i in enumerate([1000, 10000, 100000]):
        # Prediction of the absolute joint position of the robotic arm.
        #   1\ Initialization.
        FCNN_IK_Predictor_Cls = FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/Data/Model/Config_N_{N_i}_Scaler_x.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}_Scaler_y.pkl', 
                                                                 f'{project_folder}/Data/Model/Config_N_{N_i}.h5')
        
        # Obtain the predicted end-effector coordinates using neural-netowrk inverse kinematics from the generated 
        # radom coordinates.
        data_predicted_i = []; e_p_i = []
        for _, data_i in enumerate(np.array(data, dtype=np.float32)):
            # Predict the absolute joint position of the robotic arm from the input position of the end-effector 
            # and configuration of the solution.
            theta_predicted = FCNN_IK_Predictor_Cls.Predict(np.array([data_i[0], data_i[1], CONST_IK_CONFIGURATION], dtype=np.float32))[0]

            # Obtain the end-effector coordinates.
            data_predicted_i.append(np.round(Kinematics.Core.Forward_Kinematics(theta_predicted, Robot_Str)[1], tolerance).astype('float32'))

            # Obtain the absolute orientation error.
            e_p_i.append(Mathematics.Euclidean_Norm(np.round(Kinematics.Core.Forward_Kinematics(theta_predicted, Robot_Str)[1], tolerance).astype('float32') - data_i))

        # Store the data.
        data_predicted.append(data_predicted_i); e_p.append(e_p_i)

        # Release class object.
        del FCNN_IK_Predictor_Cls

    # Set the parameters for the scientific style.
    plt.style.use('science')

    aux_label = r'$IK_{Cfg}$'; data = np.array(data, dtype=np.float32); 
    for _, (predicted_i, e_p_i, color_i) in enumerate(zip(data_predicted, e_p,
                                                                     ['#3d85c6', '#6aa84f', '#674ea7'])):
        # Create a figure.
        figure = plt.figure()
        figure.tight_layout()
        ax = figure.add_subplot()

        predicted_i = np.array(predicted_i, dtype=np.float32)
        # Visualization of relevant structures.
        ax.plot(data[:, 0], data[:, 1], '.', color='#8d8d8d', alpha=1.0, markersize=8.0, markeredgewidth=2.0, markerfacecolor='#ffffff', 
                label=f'Desired Coordinates: N = {CONST_NUM_OF_DATA}')
        ax.plot(predicted_i[:, 0], predicted_i[:, 1], '.', color=color_i, alpha=1.0, markersize=8.0, markeredgewidth=2.0, markerfacecolor='#ffffff', 
                label=f'Desired Predicted: {aux_label} = {CONST_IK_CONFIGURATION}, MAE = {np.round(np.mean(e_p_i), 3)}')

        # Set parameters of the graph (plot).
        ax.set_title(f'Visualization of Robot End-Effector Coordinates using a Neural-Network Predictor for Inverse Kinematics Calculation\nNeural-Network Type {i}', fontsize=25, pad=25.0)
        # Set parameters of the graph (plot).
        #   Set the x ticks.
        ax.set_xticks(np.arange(np.min(data[:, 0]) - 0.1, np.max(data[:, 0]) + 0.1, 0.1))
        #   Set the y ticks.
        ax.set_yticks(np.arange(np.min(data[:, 1]) - 0.1, np.max(data[:, 1]) + 0.1, 0.1))
        #   Label.
        ax.set_xlabel(r'x-axis in meters', fontsize=15, labelpad=10)
        ax.set_ylabel(r'y-axis in meters', fontsize=15, labelpad=10) 
        #   Set parameters of the visualization.
        ax.grid(which='major', linewidth = 0.15, linestyle = '--')
        # Get handles and labels for the legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        # Remove duplicate labels.
        legend = dict(zip(labels, handles))
        # Show the labels (legends) of the graph.
        ax.legend(legend.values(), legend.keys(), fontsize=10.0)

        # Show the result.
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    sys.exit(main())
