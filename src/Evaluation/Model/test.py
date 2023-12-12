# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Lib.:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Trajectory/Utilities
import Lib.Trajectory.Utilities
#   ../Configuration/Parameters
import Configuration.Parameters
#   ../Lib/FCNN_IK/Model
import Lib.FCNN_IK.Model


"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# Save the data to a file.
CONST_SAVE_DATA = False
# Dataset configuration.
#   Number of data to be generated.
CONST_NUM_OF_DATA = 100000
#   Method to be used for training.
#       Method 0: No test (validation) partition.
#       Method 1: With test (validation) partition.
CONST_DATASET_METHOD = 1

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Prediction of the absolute joint position of the robotic arm.
    #   1\ Initialization.
    if CONST_DATASET_METHOD == 0:
        FCNN_IK_Predictor_Cls = Lib.FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False_Scaler_y.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_False.h5')
    elif CONST_DATASET_METHOD == 1:
        FCNN_IK_Predictor_Cls = Lib.FCNN_IK.Model.FCNN_Predictor_Cls(f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_x.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True_Scaler_y.pkl', 
                                                                     f'{project_folder}/src/Data/Model/{Robot_Str.Name}/Config_N_{CONST_NUM_OF_DATA}_use_val_True.h5')
    

    # Initialization of the class to generate trajectory.
    Polynomial_Cls = Lib.Trajectory.Utilities.Polynomial_Profile_Cls(delta_time=0.1)
    
    # Obtain the constraints for absolute joint positions in order to generate multi-axis position trajectories.
    (abs_j_pos_0, abs_j_pos_1) = Configuration.Parameters.Get_Absolute_Joint_Positions(Robot_Str.Name)

    # Generation of multi-axis position trajectories from input parameters.
    theta_arr = []
    for _, (th_actual, th_desired) in enumerate(zip(abs_j_pos_0, abs_j_pos_1)):
        (theta_arr_i, _, _) = Polynomial_Cls.Generate(th_actual, th_desired, 0.0, 0.0, 0.0, 0.0,
                                                      Configuration.Parameters.CONST_T_0, Configuration.Parameters.CONST_T_1)
        theta_arr.append(theta_arr_i)

    # The tolerance of the data.
    tolerance = 4

    # Obtain the homogeneous transformation matrix using forward kinematics from 
    # the generated multi-axis position trajectories.
    e_p = []; e_q = []
    for _, theta_arr_i in enumerate(np.array(theta_arr, dtype=np.float64).T):
        # Obtain the desired homogeneous transformation matrix.
        T = Kinematics.Forward_Kinematics(theta_arr_i, 'Fast', Robot_Str)[1]
        #   Get the translational and rotational part from the desired transformation matrix.
        p = np.round(T.p.all(), tolerance); q = np.round(T.Get_Rotation('QUATERNION').all(), tolerance)

        # Predict the absolute joint position of the robotic arm from the input 
        # position and orientation of the end-effector.
        data = FCNN_IK_Predictor_Cls.Predict(np.concatenate((p, q)).astype('float32'))[0]

        # Obtain the predicted homogeneous transformation matrix.
        T_1 = Kinematics.Forward_Kinematics(data, 'Fast', Robot_Str)[1]
        
        # Obtain the absolute error of position and orientation.
        e_p.append(Mathematics.Euclidean_Norm((T_1.p - T.p).all())); e_q.append(T_1.Get_Rotation('QUATERNION').Distance('Euclidean', T.Get_Rotation('QUATERNION')))

    # Set the parameters for the scientific style.
    plt.style.use('science')

    label = [r'$e_{p}(\hat{t})$', r'$e_{q}(\hat{t})$']; title = ['Absolute Position Error (APE)', 
                                                                 'Absolute Orientation Error (AOE)']
    for i, e_i in enumerate([e_p, e_q]):
        # Create a figure.
        _, ax = plt.subplots()

        # Visualization of relevant structures.
        ax.plot(Polynomial_Cls.t, e_i, 'x', color='#8d8d8d', linewidth=3.0, markersize=8.0, markeredgewidth=3.0, markerfacecolor='#8d8d8d', label=label[i])
        ax.plot(Polynomial_Cls.t, [np.mean(e_i)] * Polynomial_Cls.t.size, '--', color='#8d8d8d', linewidth=1.5, label=f'Mean Absolute Error (MAE)')

        # Set parameters of the graph (plot).
        ax.set_title(f'{title[i]}', fontsize=25, pad=25.0)
        #   Set the x ticks.
        ax.set_xticks(np.arange(np.min(Polynomial_Cls.t) - 0.1, np.max(Polynomial_Cls.t) + 0.1, 0.1))
        #   Set the y ticks.
        tick_y_tmp = (np.max(e_i) - np.min(e_i))/10.0
        tick_y = tick_y_tmp if tick_y_tmp != 0.0 else 0.1
        ax.set_yticks(np.arange(np.min(e_i) - tick_y, np.max(e_i) + tick_y, tick_y))
        #   Label
        ax.set_xlabel(r'Normalized time $\hat{t}$ in the range of [0.0, 1.0]', fontsize=15, labelpad=10)
        ax.set_ylabel(f'Absolute error {label[i]} in millimeters', fontsize=15, labelpad=10) 
        #   Set parameters of the visualization.
        ax.grid(which='major', linewidth = 0.15, linestyle = '--')
        # Get handles and labels for the legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        # Remove duplicate labels.
        legend = dict(zip(labels, handles))
        # Show the labels (legends) of the graph.
        ax.legend(legend.values(), legend.keys(), fontsize=10.0)

        # Display the results as the values shown in the console.
        print(f'[INFO] Iteration: {i}')
        print(f'[INFO] max(label{i}) = {np.max(e_i)} in mm')
        print(f'[INFO] min(label{i}) = {np.min(e_i)} in mm')
        print(f'[INFO] MAE = {np.mean(e_i)} in mm')

        if CONST_SAVE_DATA == True:
            # Set the full scree mode.
            plt.get_current_fig_manager().full_screen_toggle()

            # Save the results.
            plt.savefig(f'{project_folder}/images/IK/{Robot_Str.Name}/Method_Analyrtical_IK_Error_{label[i]}.png', 
                        format='png', dpi=300)
        else:
            # Show the result.
            plt.show()

if __name__ == "__main__":
    sys.exit(main())
