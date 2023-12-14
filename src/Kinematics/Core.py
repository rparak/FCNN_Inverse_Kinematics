# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Parameters/Robot
import Parameters.Robot
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

def __Check_Theta_Limit(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.List[float]:
    """
    Description:
        Function to check that the desired absolute joint positions are not out of limit.

    Args:
        (1) theta [Vector<float>]: Desired absolute joint position in radians.
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

def Forward_Kinematics(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Tuple[tp.List[float], 
                                                                                                  tp.List[float]]:
    """
    Description:
        Calculation of forward kinematics using the fast method. The function was created by simplifying 
        the standard Denavit-Hartenberg (DH) method.

        Note:
            The calculation only works for the RR robotic structure (called SCARA -> simplified version).

    Args:
        (1) theta [Vector<float>]: Desired absolute joint position in radians.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<bool> 1x2]: The result is a vector of values with a warning if the limit 
                                          is exceeded. 
                                            Note:
                                                The value in the vector is "True" if the desired absolute 
                                                joint positions are within the limits, and "False" if they 
                                                are not.
        (2) paramter [Vector<float> 1x2]: The obtained TCP (tool center point) in Cartesian coordinates (also called 
                                          orthogonal coordinates) defined as a vector in the x, y axes.
    """
    
    # Check that the desired absolute joint positions are not out of limit.
    th_limit_err = __Check_Theta_Limit(theta, Robot_Parameters_Str)

    # Express the absolute positions of the joints.
    th_0 = Robot_Parameters_Str.DH.Standard[0, 0] + theta[0]
    th_1 = Robot_Parameters_Str.DH.Standard[1, 0] + theta[1]

    # Calculation of forward kinematics using the fast method. The function was created by simplifying 
    # the Standard Denavit-Hartenberg (DH) method.
    x = np.zeros(Robot_Parameters_Str.Theta.Zero.size, dtype=np.float32)
    x[0] = Robot_Parameters_Str.DH.Standard[0, 1]*np.cos(th_0) + Robot_Parameters_Str.DH.Standard[1, 1]*np.cos(th_0 + th_1)
    x[1] = Robot_Parameters_Str.DH.Standard[0, 1]*np.sin(th_0) + Robot_Parameters_Str.DH.Standard[1, 1]*np.sin(th_0 + th_1)

    return (th_limit_err, x)

def Inverse_Kinematics(p: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Tuple[tp.Dict, 
                                                                                              tp.List[tp.List[float]]]:
    """
    Description:
        A function to compute the solution of the inverse kinematics (IK) of the RR robotic 
        structure (called SCARA -> simplified version) using an analytical method.

        Note:
            R - Revolute

    Args:
        (1) p [Vector<float> 1x2]: The desired TCP (tool center point) in Cartesian coordinates (also called 
                                   orthogonal coordinates) defined as a vector in the x, y axes.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Dictionary {'error': Vector<float> 1xk]: Information on the results found.
                                                                    Note 1:
                                                                         Where k is the number of solutions.
                                                                    Note 2:
                                                                        'error': Information about the absolute position error.
        (2) parameter [Vector<float> 2x2]: Obtained solutions of the absolute positions of the joints in radians.
    """
        
    # Initialization of output solutions.
    theta_solutions = np.zeros((2, Robot_Parameters_Str.Theta.Zero.size), dtype=np.float32)

    """
    Calculation angle of Theta 1, 2 (Inverse trigonometric functions):
        Rule 1: 
            The range of the argument 'x' for arccos function is limited from -1 to 1.
                -1 <= x <= 1
        Rule 2: 
            Output of arccos is limited from 0 to PI (radian).
                0 <= y <= PI

    Auxiliary Calculations.
        Pythagorean theorem:
            L = sqrt(x^2 + y^2)
        Others:
            tan(gamma) = y/x -> gamma = arctan2(y, x)
    """

    # The Law of Cosines.
    #   L_{2}^2 = L_{1}^2 + L^2 - 2*L_{1}*L*cos(beta)
    #       ...
    #   cos(beta) = (L_{1}^2 + L^2 - L_{2}^2) / (2*L_{1}*L)
    #       ...
    #   beta = arccos((L_{1}^2 + L^2 - L_{2}^2) / (2*L_{1}*L))
    beta = ((Robot_Parameters_Str.DH.Standard[0, 1]**2) + (p[0]**2 + p[1]**2) - (Robot_Parameters_Str.DH.Standard[1, 1]**2)) \
            / (2*Robot_Parameters_Str.DH.Standard[0, 1]*np.sqrt(p[0]**2 + p[1]**2))

    # Calculation of the absolute position of the Theta_{1} joint.
    if beta > 1:
        theta_solutions[0, 0] = np.arctan2(p[1], p[0]) 
    elif beta < -1:
        theta_solutions[0, 0] = (np.arctan2(p[1], p[0]) - Mathematics.CONST_MATH_PI)
    else:
        # Configuration 1:
        #   cfg_{1} = gamma - beta 
        theta_solutions[0, 0] = (np.arctan2(p[1], p[0]) - np.arccos(beta))
        # Configuration 2:
        #   cfg_{2} = gamma + beta 
        theta_solutions[1, 0] = (np.arctan2(p[1], p[0]) + np.arccos(beta))
            
    # The Law of Cosines.
    #   L^2 = L_{1}^2 + L_{2}^2 - 2*L_{1}*L{2}*cos(alpha)
    #       ...
    #   cos(alpha) = (L_{1}^2 + L_{2}^2 - L^2) / 2*L_{1}*L{2}
    #       ...
    #   alpha = arccos((L_{1}^2 + L_{2}^2 - L^2) / 2*L_{1}*L{2})
    alpha = ((Robot_Parameters_Str.DH.Standard[0, 1]**2) + (Robot_Parameters_Str.DH.Standard[1, 1]**2) - (p[0]**2 + p[1]**2)) \
            / (2*(Robot_Parameters_Str.DH.Standard[0, 1]*Robot_Parameters_Str.DH.Standard[1, 1]))

    # Calculation of the absolute position of the Theta_{2} joint.
    if alpha > 1:
        theta_solutions[0, 1] = Mathematics.CONST_MATH_PI
    elif alpha < -1:
        theta_solutions[0, 1] = 0.0
    else:
        # Configuration 1:
        #   cfg_{1} = PI - alpha
        theta_solutions[0, 1] = Mathematics.CONST_MATH_PI - np.arccos(alpha)
        # Configuration 2:
        #   cfg_{2} = alpha - PI
        theta_solutions[1, 1] = np.arccos(alpha) - Mathematics.CONST_MATH_PI

    # Obtain the absolute position error.
    info = {'error': np.zeros(theta_solutions.shape[0], dtype=np.float32)}
    for i, th_sol_i in enumerate(theta_solutions):
        info['error'][i] = Mathematics.Euclidean_Norm(Forward_Kinematics(th_sol_i, Robot_Parameters_Str)[1] - p)

    return (info, theta_solutions)



