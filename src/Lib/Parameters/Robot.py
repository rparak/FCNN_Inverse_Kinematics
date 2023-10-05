# Numpy (Array computing) [pip3 install numpy]tp
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

@dataclass
class DH_Parameters_Str:
    """
    Description:
        The auxiliary structure of the Denavit-Hartenberg (DH) parameters.

        Note 1:
            Private structure.

        Note 2:
            DH (Denavit-Hartenberg) parameters: 
    
            (1) theta_zero [Vector<float>]: Joint angle (Theta_i). Rotation part in radians.
                                            Unit: [radian]                        
            (2) a [Vector<float>]: Link length (a_i). Translation part in meters.
                                   Unit: [meter]
            (3) d [Vector<float>]: Link offset (d_i). Translation part in meters.
                                   Unit: [meter]
            (4) alpha [Vector<float>]: Link twist (alpha_i). Rotation part in radians.
                                       Unit: [radian]
    """

    # Standard Denavit-Hartenberg (DH):
    #       DH_theta_zero = th{i} + theta_zero{i}
    #       DH_a          = a{i}
    #       DH_d          = d{i}
    #       DH_alpha      = alpha{i}
    #   Unit [Matrix<float>]
    Standard: tp.List[tp.List[float]] = field(default_factory=list)
    # Modified Denavit-Hartenberg (DH):
    #       DH_theta_zero = th{i} + theta_zero{i}
    #       DH_a          = a{i - 1}
    #       DH_d          = d{i}
    #       DH_alpha      = alpha{i - 1}
    #   Unit [Matrix<float>]
    Modified: tp.List[tp.List[float]] = field(default_factory=list)

@dataclass
class Theta_Parameters_Str(object):
    """
    Description:
        The auxiliary structure of the joint (theta) parameters.

        Note:
            Private structure.
    """

    # Zero absolute position of each joint.
    #   Unit [Vector<float>]
    Zero: tp.List[float] = field(default_factory=list)
    # Home absolute position of each joint.
    #   Unit [Vector<float>]
    Home: tp.List[float] = field(default_factory=list)
    # Limits of absolute joint position in radians and meters.
    #   Unit [Matrix<float>]
    Limit: tp.List[tp.List[float]] = field(default_factory=list)
    # Other parameters of the object structure.
    #   The name of the joints.
    #       Unit [Vector<string>]
    Name: tp.List[str] = field(default_factory=list)
    #   Identification of the type of joints.
    #       Note: R - Revolute, P - Prismatic
    #       Unit [Vector<string>]
    Type: tp.List[str] = field(default_factory=list)
    #   Identification of the axis of the absolute position of the joint. 
    #       Note: 'X', 'Z'
    #       Unit [Vector<string>]
    Axis: tp.List[str] = field(default_factory=list)
    #   Identification of the axis direction.
    #       Note: (+1) - Positive, (-1) - Negative
    #       Unit [Vector<int>]
    Direction: tp.List[int] = field(default_factory=list)

@dataclass
class T_Parameters_Str:
    """
    Description:
        The auxiliary structure of the homogeneous transformation matrix {T} parameters.

        Note:
            Private structure.
    """

    # Homogeneous transformation matrix of the base.
    #   Unit [Matrix<float>]
    Base: tp.List[tp.List[float]] = field(default_factory=list)
    # Homogeneous transformation matrix of the end-effector (tool).
    #   Unit [Matrix<float>]
    End_Effector: tp.List[tp.List[float]] = field(default_factory=list)
    # The zero configuration of the homogeneous transformation 
    # matrix of each joint (theta). The method (Standard, Modified) chosen 
    # to determine the configuration depends on the specific task.
    #   Unit [Matrix<float>]
    Zero_Cfg: tp.List[tp.List[float]] = field(default_factory=list)

@dataclass
class Robot_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the robot.

    Initialization of the Class (structure):
        Input:
            (1) name [string]: Name of the robotic structure.

    Example:
        Initialization:
            Cls = Robot_Parameters_Str(name)
            Cls.Name = ...
            ...
            Cls.T = ..
    """

    # Name of the robotic structure.
    #   Unit [string]
    Name: str = ''
    # Identification number.
    #   Unit [int]
    Id: int = 0
    # Denavit-Hartenberg (DH) parameters.
    #   Unit [DH_Parameters_Str(object)]
    DH: DH_Parameters_Str = field(default_factory=DH_Parameters_Str)
    # Absolute joint position (theta) parameters.
    #   Unit [Theta_Parameters_Str(object)]
    Theta: Theta_Parameters_Str = field(default_factory=Theta_Parameters_Str)
    # Homogeneous transformation matrix (T) parameters.
    #   Unit [T_Parameters_Str(object)]
    T: T_Parameters_Str = field(default_factory=T_Parameters_Str)

"""
Robot Type - Epson LS3-B401S:
    Absolute Joint Position:
        Joint 1: [-40, +220.0] [°]
        Joint 2: [+/- 140.0] [°]
        Joint 3: [-0.150, +0.0] [m]
        Joint 4: [+/- 180.0] [°]

    Denavit-Hartenberg (DH) Standard:
        Method 1 (th_3 - rotates counterclockwise):
            Note 1: The direction of the Z axis is upwards.
            Note 2: The Denavit-Hartenberg parameter d from 
                    the table will be positive (see Kinematics.py).
                theta_zero = [   0.0,    0.0, 0.0,     0.0]
                a          = [ 0.225,  0.175, 0.0,     0.0]
                d          = [0.1731, 0.0499, 0.0, -0.0785]
                alpha      = [   0.0,    0.0, 0.0,     0.0]
        Method 2 (th_3 - rotates clockwise):
            Note 1: The direction of the Z axis is downwards.
            Note 2: The Denavit-Hartenberg parameter d from 
                    the table will be negative (see Kinematics.py).
                theta_zero = [   0.0,    0.0, 0.0,    0.0]
                a          = [ 0.225,  0.175, 0.0,    0.0]
                d          = [0.1731, 0.0499, 0.0, 0.0785]
                alpha      = [   0.0,   3.14, 0.0,    0.0]
"""
EPSON_LS3_B401S_Str = Robot_Parameters_Str(Name='EPSON_LS3_B401S', Id=1)
# Homogeneous transformation matrix of the base.
#   1\ None: Identity Matrix
#       [[1.0, 0.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0, 0.0],
#        [0.0, 0.0, 1.0, 0.0],
#        [0.0, 0.0, 0.0, 1.0]]
EPSON_LS3_B401S_Str.T.Base = HTM_Cls(None, np.float64)
# End-effector (tool):
#   1\ None: Identity Matrix
#       [[1.0, 0.0, 0.0, 0.0],
#        [0.0, 1.0, 0.0, 0.0],
#        [0.0, 0.0, 1.0, 0.0],
#        [0.0, 0.0, 0.0, 1.0]]
EPSON_LS3_B401S_Str.T.End_Effector = HTM_Cls(None, np.float64)
# Denavit-Hartenberg (DH)
EPSON_LS3_B401S_Str.DH.Standard = np.array([[0.0, 0.225,  0.1731,               0.0],
                                            [0.0, 0.175,  0.0499, 3.141592653589793],
                                            [0.0,   0.0,     0.0,               0.0],
                                            [0.0,   0.0,  0.0785,               0.0]], dtype = np.float64) 
EPSON_LS3_B401S_Str.DH.Modified = np.array([[0.0,   0.0,  0.1731,               0.0],
                                            [0.0, 0.225,  0.0499,               0.0],
                                            [0.0, 0.175,     0.0, 3.141592653589793],
                                            [0.0,   0.0,  0.0785,               0.0]], dtype = np.float64) 
# Zero/Home absolute position of each joint.
EPSON_LS3_B401S_Str.Theta.Zero = np.array([0.0, 0.0, 0.0, 0.0], 
                                          dtype = np.float64)
EPSON_LS3_B401S_Str.Theta.Home = np.array([Mathematics.Degree_To_Radian(90.0), Mathematics.Degree_To_Radian(0.0), 0.0, Mathematics.Degree_To_Radian(0.0)],
                                          dtype = np.float64)
# Limits of absolute joint position.
EPSON_LS3_B401S_Str.Theta.Limit = np.array([[-0.6981317007977318, 3.839724354387525], 
                                            [ -2.443460952792061, 2.443460952792061], 
                                            [                0.0,             0.150], 
                                            [ -3.141592653589793, 3.141592653589793]], dtype = np.float64)
# Parameters of the object (Blender robot arm).
EPSON_LS3_B401S_Str.Theta.Name = [f'Joint_1_{EPSON_LS3_B401S_Str.Name}_ID_{EPSON_LS3_B401S_Str.Id:03}', 
                                  f'Joint_2_{EPSON_LS3_B401S_Str.Name}_ID_{EPSON_LS3_B401S_Str.Id:03}', 
                                  f'Joint_3_{EPSON_LS3_B401S_Str.Name}_ID_{EPSON_LS3_B401S_Str.Id:03}', 
                                  f'Joint_4_{EPSON_LS3_B401S_Str.Name}_ID_{EPSON_LS3_B401S_Str.Id:03}']
EPSON_LS3_B401S_Str.Theta.Type = ['R', 'R', 'P', 'R']
EPSON_LS3_B401S_Str.Theta.Axis = ['Z', 'Z', 'Z', 'Z']
EPSON_LS3_B401S_Str.Theta.Direction = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float16)