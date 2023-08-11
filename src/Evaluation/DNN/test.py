# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Core as Kinematics
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.EPSON_LS3_B401S_Str
# Interpolation stop(t_0), start(t_1) time in seconds.
CONST_T_0 = 0.0
CONST_T_1 = 1.0
# FPS (Frames Per Seconds) value.
CONST_FPS = 30
# Absolute joint position start(id: 0) and stop(id: 1).
CONST_ABS_J_POS_0 = np.array([0.0, 0.0, 0.0, 0.0], dtype = np.float32)
CONST_ABS_J_POS_1 = np.array([Mathematics.Degree_To_Radian(90.0), Mathematics.Degree_To_Radian(0.0), 0.10, Mathematics.Degree_To_Radian(45.0)],
                              dtype = np.float32)

def smoothstep(x):
    return x * x * (3 - 2 * x)

def smooth_interpolate_angle(start_angle, end_angle, t):
    # Normalize the angles to be between 0 and 360 degrees
    start_angle = start_angle % 360
    end_angle = end_angle % 360

    # Calculate the shortest angular distance between the two angles
    angular_distance = (end_angle - start_angle + 180) % 360 - 180

    # Calculate the smoothed t value
    smoothed_t = smoothstep(t)

    # Calculate the interpolated angle
    interpolated_angle = start_angle + angular_distance * smoothed_t

    return interpolated_angle

def main():
    """
    Description:
        ...
    """

    # Time in seconds.
    t = np.linspace(CONST_T_0, CONST_T_1, 100)

    y = smooth_interpolate_angle(0.0, Mathematics.Degree_To_Radian(90.0), t)

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure.
    _, ax = plt.subplots()

    # ...
    ax.plot(t, y, '--o', color='#d0d0d0', linewidth=1.0, markersize = 3.0, 
            markeredgewidth = 1.5, markerfacecolor = '#ffffff', label='...')

    # Show the result.
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
