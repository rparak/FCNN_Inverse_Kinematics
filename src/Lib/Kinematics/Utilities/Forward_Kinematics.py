# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters

def __FKF_EPSON_LS3_B401S(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot_Parameters_Str) -> tp.Tuple[tp.List[float], 
                                                                                                                    tp.List[tp.List[float]]]:
    """
    Description:
        Calculation of forward kinematics using a fast method for the Epson LS3 B401S robotic arm.

    Args:
        (1) theta [Vector<float>]: Desired absolute joint position in radians / meters.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Matrix<float> 4x4]: Homogeneous end-effector transformation matrix.
    """
    
    """
    Description:
        Abbreviations for individual functions. Used to speed up calculations.
    """
    # Angles:
    th_01 = theta[0] + theta[1]; th_013 = th_01 - theta[3]
    # Sine / Cosine functions:
    c_th_013 = np.cos(th_013); s_th_013 = np.sin(th_013)

    # Computation of the homogeneous end-effector transformation matrix {T}
    T = np.array(np.identity(4), dtype=np.float64)
    T[0,0] = c_th_013
    T[0,1] = s_th_013
    T[0,2] = 0.0
    T[0,3] = 0.225*np.cos(theta[0]) + 0.175*np.cos(th_01)
    T[1,0] = s_th_013
    T[1,1] = -c_th_013
    T[1,2] = 0.0
    T[1,3] = 0.225*np.sin(theta[0]) + 0.175*np.sin(th_01)
    T[2,0] = 0.0
    T[2,1] = 0.0
    T[2,2] = -1.0
    T[2,3] = theta[2] + 0.144499991167482
    T[3,0] = 0.0
    T[3,1] = 0.0
    T[3,2] = 0.0
    T[3,3] = 1.0

    # T_Base @ T_n @ T_EE
    return Robot_Parameters_Str.T.Base @ T @ Robot_Parameters_Str.T.End_Effector

def FKFast_Solution(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot_Parameters_Str) -> tp.Tuple[tp.List[float], 
                                                                                                              tp.List[tp.List[float]]]:
    """
    Description:
        Calculation of forward kinematics using the fast method. The function was created by simplifying 
        the modified Denavit-Hartenberg (DH) method.

        Note 1:
            The resulting function shape for each robot was created by manually removing duplicates.

        Note 2:
            The Forward Kinematics fast calculation method works only for defined robot types. But the function can be easily 
            extended to another type of robot.

            Manipulators:
                Epson: LS3 B401S

    Args:
        (1) theta [Vector<float>]: Desired absolute joint position in radians / meters.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<bool>]: The result is a vector of values with a warning if the limit 
                                      is exceeded. 
                                      Note:
                                        The value in the vector is "True" if the desired absolute 
                                        joint positions are within the limits, and "False" if they 
                                        are not.
        (2) parameter [Matrix<float> 4x4]: Homogeneous end-effector transformation matrix.
    """
        
    return {
        'EPSON_LS3_B401S': lambda th, r_param_str: __FKF_EPSON_LS3_B401S(th, r_param_str)
    }[Robot_Parameters_Str.Name](theta, Robot_Parameters_Str)