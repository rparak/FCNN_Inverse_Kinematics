# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Script:
#   ../Lib/Parameters/Robot
import Lib.Parameters.Robot as Parameters
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

def Check_Theta_Limit(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot_Parameters_Str) -> tp.List[float]:
    """
    Description:
        Function to check that the desired absolute joint positions are not out of limit.

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
    """

    th_limit_err = [False] * theta.size
    for i, (th_i, th_i_limit) in enumerate(zip(theta, Robot_Parameters_Str.Theta.Limit)):
        th_limit_err[i] = False if th_i_limit[0] <= th_i <= th_i_limit[1] else True

    return th_limit_err

def Get_Angle_Axis_Error(T_desired: tp.List[tp.List[float]], T_current: tp.List[tp.List[float]]) ->tp.List[float]:
    """
    Description:    
        Get an error (angle-axis) vector which represents the translation and rotation from the end-effector's current 
        position (T_current) to the desired position (T_desired).

        The position error is defined as a:
            e_i_p(theta) = d_p_i - p_i(theta),

        where d_p_i is the desired position and p_i is the current position.

        The rotation error is defined as a:

            e_i_R(theta) = alpha(d_R_i * R_i.T(theta)),

        where d_R_i is the desired rotation and R_i is the current rotation. And alpha(R) is angle-axis equivalent of rotation matrix R.

        Reference:
            T. Sugihara, "Solvability-Unconcerned Inverse Kinematics by the Levenberg-Marquardt Method."

    Args:
        (1) T_desired [Matrix<float> 4x4]: Homogeneous transformation matrix of the desired position/rotation.
        (2) T_current [Matrix<float> 4x4]: Homogeneous transformation matrix of the current position/rotation.

    Returns:
        (1) parameter [Vector<float> 1x6]: Vector of an error (angle-axis) from current to the desired position/rotation.
    """
    
    # Initialization of the output vector, which consists of a translational 
    # and a rotational part.
    e_i = np.zeros(6, dtype=np.float32)

    # 1\ Calculation of position error (e_i_p).
    e_i[:3] = (T_desired.p - T_current.p).all()

    # 2\ Calculation of rotation error (e_i_R).
    R = T_desired.R @ T_current.Transpose().R

    # Trace of an 3x3 square matrix R (tr(R)).
    Tr_R = Transformation.Get_Matrix_Trace(R)
    # Not-zero vector {l}.
    l = Transformation.Vector3_Cls([R[2, 1] - R[1, 2], 
                                    R[0, 2] - R[2, 0], 
                                    R[1, 0] - R[0, 1]], T_desired.Type)
    # Length (norm) of the non-zero vector {l}.
    l_norm = l.Norm()

    if l_norm > Mathematics.CONST_EPS_64:
        e_i[3:] = (np.arctan2(l_norm, Tr_R - 1) * l.all()) / l_norm
    else:
        """
        Condition (Tr_R > 0):
            (r_{11} ,r_{22} ,r_{33}) = (+, +, +), then alpha = Null vector.

            The row e_i[3:] = [0.0, 0.0, 0.0]> is not necessary because 
            <e_i = np.zeros(6, dtype=np.float32)>.
        """
        if Tr_R <= 0:
            e_i[3:] = Mathematics.CONST_MATH_HALF_PI * (Transformation.Get_Matrix_Diagonal(R) + 1)

    return e_i