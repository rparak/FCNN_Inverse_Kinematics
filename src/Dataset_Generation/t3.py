# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Library:
#   ../Lib/Kinematics/Core
import Lib.Kinematics.Utilities.General as General
import Lib.Transformation.Core as Transformation
import Lib.Transformation.Utilities.Mathematics as Mathematics


T_1 = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Rotation([20.0,0.0,0.0], 'ZYX').Translation([0.0, 0.0, 0.1])
T_2 = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)

print(Mathematics.Euclidean_Norm((T_1.p - T_2.p).all()))

q_1 = T_1.Get_Rotation('QUATERNION')
q_2 = T_2.Get_Rotation('QUATERNION')
print(q_1.Distance('Euclidean', q_2))

