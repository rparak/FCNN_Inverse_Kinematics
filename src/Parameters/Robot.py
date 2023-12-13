# Numpy (Array computing) [pip3 install numpy]tp
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics

@dataclass
class DH_Parameters_Str:
    """
    Description:
        The auxiliary structure of the Denavit-Hartenberg (DH) parameters.

        Note:
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

@dataclass
class Theta_Parameters_Str(object):
    """
    Description:
        The auxiliary structure of the joint (theta) parameters.
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

@dataclass
class Robot_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the robot.

    Initialization of the Class (structure):
        Input:
            (1) Name [string]: Name of the robotic structure.
            (2) Id [int]: Identification number.

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

"""
Robot Type - Epson LS3-B401S:
    Absolute Joint Position:
        Joint 1: [-40, +220.0] [°]
        Joint 2: [+/- 140.0] [°]

    Denavit-Hartenberg (DH) Standard:
        theta_zero = [   0.0,    0.0]
        a          = [ 0.225,  0.175]

    Note:
        Parameters are expressed only for the first two axes of rotation. The structure is defined 
        as a robotic arm with two joints.
"""
EPSON_LS3_B401S_Str = Robot_Parameters_Str(Name='EPSON_LS3_B401S', Id=1)
# Denavit-Hartenberg (DH)
EPSON_LS3_B401S_Str.DH.Standard = np.array([[0.0, 0.225],
                                            [0.0, 0.175]], dtype = np.float32) 
# Zero/Home absolute position of each joint.
EPSON_LS3_B401S_Str.Theta.Zero = np.array([0.0, 0.0], 
                                          dtype = np.float32)
EPSON_LS3_B401S_Str.Theta.Home = np.array([Mathematics.Degree_To_Radian(90.0), Mathematics.Degree_To_Radian(0.0)],
                                          dtype = np.float32)
# Limits of absolute joint position.
EPSON_LS3_B401S_Str.Theta.Limit = np.array([[-0.6981317007977318, 3.839724354387525], 
                                            [ -2.443460952792061, 2.443460952792061]], dtype = np.float32)
