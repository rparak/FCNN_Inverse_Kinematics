# System (Default)
import sys
#   Add access if it is not in the system path.
sys.path.append('../..')
# Time (Time access and conversions)
import time
# Sympy (Symbolic mathematics) [pip3 install sympy]
import sympy as sp
# Custom Script:
#   ../Lib/Manipulator/Parameters
import Lib.Parameters.Robot as Parameters

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.ABB_IRB_120_Str

def __DH_Modified(theta: float, a: float, d: float, alpha: float) -> sp.Matrix:
    """
    Description:
        Modified Denavit-Hartenberg Method.
        
    Args:
        (1 - 4) theta, a, d, alpha [float]: DH (Denavit-Hartenberg) parameters in the current episode.
        
    Returns:
        (1) parameter [Matrix<sympy> 4x4]: Homogeneous transformation matrix in the current episode.
    """
        
    return sp.Matrix([[sp.cos(theta)              ,        (-1.0)*sp.sin(theta),                  0.0,                      a],
                      [sp.sin(theta)*sp.cos(alpha), sp.cos(theta)*sp.cos(alpha), (-1.0)*sp.sin(alpha), (-1.0)*sp.sin(alpha)*d],
                      [sp.sin(theta)*sp.sin(alpha), sp.cos(theta)*sp.sin(alpha),        sp.cos(alpha),        sp.cos(alpha)*d],
                      [                        0.0,                         0.0,                  0.0,                    1.0]])

def Forward_Kinematics_Modified(theta: sp.symbols, Robot_Parameters_Str: Parameters.Robot_Parameters_Str) -> sp.Matrix:
    """
    Description:
        Calculation of forward kinematics using the modified Denavit-Hartenberg (DH) method.
        
        Note:
            DH (Denavit-Hartenberg) table: 
                theta (id: 0), a (id: 1), d (id: 2), alpha (id: 3)

    Args:
        (1) theta [Vector<sympy>]: Desired absolute joint position as a symbols.
        (2) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.
        
    Returns:
        (1) parameter [Matrix<sympy> 4x4]: Homogeneous end-effector transformation matrix.
    """
        
    T_i = sp.Matrix(sp.eye(4))
    for _, (th_i, dh_i, th_i_type, th_ax_i) in enumerate(zip(theta, Robot_Parameters_Str.DH.Modified, Robot_Parameters_Str.Theta.Type, 
                                                             Robot_Parameters_Str.Theta.Axis)):
        # Forward kinematics using modified DH parameters.
        if th_i_type == 'R':
            # Identification of joint type: R - Revolute
            T_i = T_i @ __DH_Modified(dh_i[0] + th_i, dh_i[1], dh_i[2], dh_i[3])
        elif th_i_type == 'P':
            # Identification of joint type: P - Prismatic
            if th_ax_i == 'Z':
                T_i = T_i @ __DH_Modified(dh_i[0], dh_i[1], dh_i[2] - th_i, dh_i[3])
            else:
                # Translation along the X axis.
                T_i = T_i @ __DH_Modified(dh_i[0], dh_i[1] + th_i, dh_i[2], dh_i[3])

    return sp.simplify(T_i)

def main():
    """
    Description:
        A program to simplify the solution of forward kinematics (FK). The results of the simplification will be used to calculate 
        the FK faster.
    """

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # Initialize a string containing the symbol assigned with the variable.
    theta = [sp.symbols(f'theta[{i}]') for i in range(len(Robot_Str.Theta.Name))]

    print('[INFO] The calculation is in progress.')
    t_0 = time.time()

    """
    Description:
        Calculation of forward kinematics simplification using the modified Denavit-Hartenberg (DH) method.
    """
    T_simpl = Forward_Kinematics_Modified(theta, Robot_Str)

    print('[INFO] Code generation.')
    print('T = np.array(np.identity(4), dtype=np.float64)')

    for i, T_i_simpl in enumerate(T_simpl.tolist()):
        for j, T_ij_simpl in enumerate(T_i_simpl):
            # Replace (convert) the old value string to the new one.
            #   Note: Better conversion to standard form (copy + paste to function).
            T_ij_simpl_new = str(sp.nsimplify(T_ij_simpl, tolerance=1e-5, rational=True).evalf()).replace('sin', 'np.sin').replace('cos', 'np.cos')
            print(f'T[{i},{j}] = {T_ij_simpl_new}')

    print('[INFO] The simplification process is successfully completed!')
    print(f'[INFO] Total Time: {(time.time() - t_0):.3f} in seconds')

if __name__ == '__main__':
    sys.exit(main())