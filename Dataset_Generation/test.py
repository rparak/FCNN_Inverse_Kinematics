# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Lib.:
#   ../Kinematics/Core
import Kinematics.Core
#   ../Parameters/Robot
import Parameters.Robot

x = Kinematics.Core.Forward_Kinematics(np.array([0.0, 0.0]), Parameters.Robot.EPSON_LS3_B401S_Str)

#print(x)

y = Kinematics.Core.Inverse_Kinematics(x[1], Parameters.Robot.EPSON_LS3_B401S_Str)

print(y[0])