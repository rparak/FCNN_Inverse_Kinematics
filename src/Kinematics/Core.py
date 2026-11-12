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
        (1) theta [Vector<float> 1xm or nxm]: Desired absolute joint position in radians.
                                              Note:
                                                Where n is the number of samples and m is the number of joints.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<bool> 1xm or nxm]: The result is a vector of values with a warning if the limit
                                                 is exceeded.
                                                 Note:
                                                    The value in the vector is "False" if the desired absolute
                                                    joint positions are within the limits, and "True" if they
                                                    are not.
    """

    theta = np.asarray(theta)
    th_limit_err = (theta < Robot_Parameters_Str.Theta.Limit[:, 0]) | (theta > Robot_Parameters_Str.Theta.Limit[:, 1])

    return th_limit_err.tolist() if theta.ndim == 1 else th_limit_err

def Wrap_To_Joint_Limits(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.List[float]:
    """
    Description:
        Function to wrap the absolute joint positions into the interval [limit_min, limit_min + 2*PI).

        Note:
            The wrapping does not change the position of the TCP (tool center point), because
            the angles are shifted by a multiple of 2*PI. Joint positions that still lie outside
            the limits after wrapping are not reachable by the robot.

    Args:
        (1) theta [Vector<float> 1xm or nxm]: Absolute joint position in radians.
                                              Note:
                                                Where n is the number of samples and m is the number of joints.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<float> 1xm or nxm]: Wrapped absolute joint position in radians.
                                                  Unit: [radian]
    """

    th_limit_min = Robot_Parameters_Str.Theta.Limit[:, 0].astype(np.float64)

    return th_limit_min + np.mod(np.asarray(theta, dtype=np.float64) - th_limit_min, 2.0 * Mathematics.CONST_MATH_PI)

def Is_Within_Joint_Limits(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Union[bool, tp.List[bool]]:
    """
    Description:
        Function to check whether all absolute joint positions are within the limits.

    Args:
        (1) theta [Vector<float> 1xm or nxm]: Absolute joint position in radians.
                                              Note:
                                                Where n is the number of samples and m is the number of joints.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [bool or Vector<bool> 1xn]: The value is "True" if all joint positions of the sample
                                                  are within the limits, and "False" if they are not.
    """

    theta = np.asarray(theta)
    is_within = np.all((theta >= Robot_Parameters_Str.Theta.Limit[:, 0]) & (theta <= Robot_Parameters_Str.Theta.Limit[:, 1]),
                       axis=-1)

    return bool(is_within) if theta.ndim == 1 else is_within

def Forward_Kinematics(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Tuple[tp.List[float],
                                                                                                  tp.List[float]]:
    """
    Description:
        Calculation of forward kinematics using the fast method. The function was created by simplifying
        the standard Denavit-Hartenberg (DH) method.

        Note 1:
            The calculation only works for the RR robotic structure (called SCARA -> simplified version).

        Note 2:
            The input can be a single joint configuration (1xm) or a batch of configurations (nxm).

    Args:
        (1) theta [Vector<float> 1xm or nxm]: Desired absolute joint position in radians.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<bool> 1xm or nxm]: The result is a vector of values with a warning if the limit
                                                 is exceeded.
                                                    Note:
                                                        The value in the vector is "False" if the desired absolute
                                                        joint positions are within the limits, and "True" if they
                                                        are not.
        (2) paramter [Vector<float> 1x2 or nx2]: The obtained TCP (tool center point) in Cartesian coordinates (also called
                                                 orthogonal coordinates) defined as a vector in the x, y axes.
                                                 Unit: [meter]
    """

    theta = np.asarray(theta, dtype=np.float64)

    # Check that the desired absolute joint positions are not out of limit.
    th_limit_err = __Check_Theta_Limit(theta, Robot_Parameters_Str)

    # Express the absolute positions of the joints.
    th_0 = Robot_Parameters_Str.DH.Standard[0, 0] + theta[..., 0]
    th_1 = Robot_Parameters_Str.DH.Standard[1, 0] + theta[..., 1]

    # Calculation of forward kinematics using the fast method. The function was created by simplifying
    # the Standard Denavit-Hartenberg (DH) method.
    x = np.zeros(theta.shape[:-1] + (Robot_Parameters_Str.Theta.Zero.size, ), dtype=np.float64)
    x[..., 0] = Robot_Parameters_Str.DH.Standard[0, 1]*np.cos(th_0) + Robot_Parameters_Str.DH.Standard[1, 1]*np.cos(th_0 + th_1)
    x[..., 1] = Robot_Parameters_Str.DH.Standard[0, 1]*np.sin(th_0) + Robot_Parameters_Str.DH.Standard[1, 1]*np.sin(th_0 + th_1)

    return (th_limit_err, x)

def Inverse_Kinematics(p: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Tuple[tp.Dict,
                                                                                              tp.List[tp.List[float]]]:
    """
    Description:
        A function to compute the solution of the inverse kinematics (IK) of the RR robotic
        structure (called SCARA -> simplified version) using an analytical method.

        Note 1:
            R - Revolute

        Note 2:
            The input can be a single position (1x2) or a batch of positions (nx2).

        Note 3:
            Both solutions are always computed. Solution 0 has theta_{2} >= 0 and solution 1
            has theta_{2} <= 0. The solutions are not wrapped into the joint limits, see the
            function Wrap_To_Joint_Limits().

        Note 4:
            The analytical solution assumes that the DH joint offsets (theta_zero) are zero.
            An AssertionError is raised otherwise.

    Args:
        (1) p [Vector<float> 1x2 or nx2]: The desired TCP (tool center point) in Cartesian coordinates (also called
                                          orthogonal coordinates) defined as a vector in the x, y axes.
                                          Unit: [meter]
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Dictionary {'error': Vector<float> 1xk or nxk]: Information on the results found.
                                                                        Note 1:
                                                                            Where k is the number of solutions.
                                                                        Note 2:
                                                                            'error': Information about the absolute position error.
        (2) parameter [Vector<float> 2x2 or nx2x2]: Obtained solutions of the absolute positions of the joints in radians.
                                                    Unit: [radian]
    """

    try:
        assert np.all(Robot_Parameters_Str.DH.Standard[:, 0] == 0.0)
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print(f'[ERROR] The analytical inverse kinematics assumes zero DH joint offsets (theta_zero), but the robot {Robot_Parameters_Str.Name} '
              f'has theta_zero = {Robot_Parameters_Str.DH.Standard[:, 0]}.')
        raise

    p = np.asarray(p, dtype=np.float64)

    # Initialization of output solutions.
    theta_solutions = np.zeros(p.shape[:-1] + (2, Robot_Parameters_Str.Theta.Zero.size), dtype=np.float64)

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
    #
    #   Note:
    #       The argument is clipped to [-1, 1] to handle points (slightly) outside
    #       the reachable workspace, e.g. due to rounding.
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = ((Robot_Parameters_Str.DH.Standard[0, 1]**2) + (p[..., 0]**2 + p[..., 1]**2) - (Robot_Parameters_Str.DH.Standard[1, 1]**2)) \
                / (2*Robot_Parameters_Str.DH.Standard[0, 1]*np.sqrt(p[..., 0]**2 + p[..., 1]**2))
    beta = np.clip(np.nan_to_num(beta, nan=1.0), -1.0, 1.0)

    # Calculation of the absolute position of the Theta_{1} joint.
    #   Configuration 1:
    #       cfg_{1} = gamma - beta
    theta_solutions[..., 0, 0] = (np.arctan2(p[..., 1], p[..., 0]) - np.arccos(beta))
    #   Configuration 2:
    #       cfg_{2} = gamma + beta
    theta_solutions[..., 1, 0] = (np.arctan2(p[..., 1], p[..., 0]) + np.arccos(beta))

    # The Law of Cosines.
    #   L^2 = L_{1}^2 + L_{2}^2 - 2*L_{1}*L{2}*cos(alpha)
    #       ...
    #   cos(alpha) = (L_{1}^2 + L_{2}^2 - L^2) / 2*L_{1}*L{2}
    #       ...
    #   alpha = arccos((L_{1}^2 + L_{2}^2 - L^2) / 2*L_{1}*L{2})
    alpha = ((Robot_Parameters_Str.DH.Standard[0, 1]**2) + (Robot_Parameters_Str.DH.Standard[1, 1]**2) - (p[..., 0]**2 + p[..., 1]**2)) \
            / (2*(Robot_Parameters_Str.DH.Standard[0, 1]*Robot_Parameters_Str.DH.Standard[1, 1]))
    alpha = np.clip(alpha, -1.0, 1.0)

    # Calculation of the absolute position of the Theta_{2} joint.
    #   Configuration 1:
    #       cfg_{1} = PI - alpha
    theta_solutions[..., 0, 1] = Mathematics.CONST_MATH_PI - np.arccos(alpha)
    #   Configuration 2:
    #       cfg_{2} = alpha - PI
    theta_solutions[..., 1, 1] = np.arccos(alpha) - Mathematics.CONST_MATH_PI

    # Obtain the absolute position error.
    info = {'error': np.linalg.norm(Forward_Kinematics(theta_solutions, Robot_Parameters_Str)[1] - p[..., np.newaxis, :],
                                    axis=-1)}

    return (info, theta_solutions)

def Jacobian(theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.List[tp.List[float]]:
    """
    Description:
        Calculation of the analytical Jacobian matrix of the TCP (tool center point) position with respect 
        to the absolute joint positions.

        Note:
            The calculation only works for the RR robotic structure (called SCARA -> simplified version).

    Args:
        (1) theta [Vector<float> 1xm or nxm]: Absolute joint position in radians.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Matrix<float> 2xm or nx2xm]: The Jacobian matrix d(x, y)/d(theta).
                                                    Unit: [meter/radian]
    """

    theta = np.asarray(theta, dtype=np.float64)

    # Express the absolute positions of the joints.
    th_0  = Robot_Parameters_Str.DH.Standard[0, 0] + theta[..., 0]
    th_01 = th_0 + Robot_Parameters_Str.DH.Standard[1, 0] + theta[..., 1]

    J = np.zeros(theta.shape[:-1] + (2, Robot_Parameters_Str.Theta.Zero.size), dtype=np.float64)
    J[..., 0, 1] = -Robot_Parameters_Str.DH.Standard[1, 1]*np.sin(th_01)
    J[..., 1, 1] =  Robot_Parameters_Str.DH.Standard[1, 1]*np.cos(th_01)
    J[..., 0, 0] = -Robot_Parameters_Str.DH.Standard[0, 1]*np.sin(th_0) + J[..., 0, 1]
    J[..., 1, 0] =  Robot_Parameters_Str.DH.Standard[0, 1]*np.cos(th_0) + J[..., 1, 1]

    return J

def Refine_Inverse_Kinematics(theta: tp.List[float], p: tp.List[float], Robot_Parameters_Str: Parameters.Robot, 
                              num_of_iterations: int, damping: float = 1e-3, max_step: float = 0.5) -> tp.List[float]:
    """
    Description:
        A function to refine an initial estimate of the absolute joint positions (e.g. the output of a neural
        network) using the damped least-squares (DLS) Jacobian method on the analytical forward kinematics.

            theta_{k+1} = theta_{k} + J^T (J J^T + damping^2 I)^-1 (p - FK(theta_{k}))

        The norm of each step is limited to max_step. Without this limit, a poor initial estimate can
        converge to the other configuration of the solution (elbow flip).

        After the refinement, the joint positions are wrapped into the joint limits.

    Args:
        (1) theta [Vector<float> 1xm or nxm]: Initial estimate of the absolute joint position in radians.
        (2) p [Vector<float> 1x2 or nx2]: The desired TCP (tool center point) in the x, y axes.
                                          Unit: [meter]
        (3) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.
        (4) num_of_iterations [int]: The number of iterations of the DLS method.
        (5) damping [float]: The damping factor of the DLS method.
                             Unit: [meter]
        (6) max_step [float]: The maximum norm of the joint step per iteration.
                              Unit: [radian]

    Returns:
        (1) parameter [Vector<float> 1xm or nxm]: Refined absolute joint position in radians.
                                                  Unit: [radian]
    """

    theta = np.array(theta, dtype=np.float64); p = np.asarray(p, dtype=np.float64)

    for _ in range(num_of_iterations):
        # The position error.
        e = p - Forward_Kinematics(theta, Robot_Parameters_Str)[1]

        # The damped least-squares step.
        J = Jacobian(theta, Robot_Parameters_Str); J_T = np.swapaxes(J, -1, -2)
        A = J @ J_T + (damping**2) * np.eye(J.shape[-2])
        d_theta = (J_T @ np.linalg.solve(A, e[..., np.newaxis]))[..., 0]

        # Limit the norm of the step.
        d_theta_norm = np.linalg.norm(d_theta, axis=-1, keepdims=True)
        theta += d_theta * np.minimum(1.0, max_step / np.maximum(d_theta_norm, 1e-12))

    return Wrap_To_Joint_Limits(theta, Robot_Parameters_Str)
