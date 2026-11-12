# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OS (Operating system interfaces)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# SciPy (Spatial algorithms and data structures) [pip3 install scipy]
import scipy.spatial
# Custom Lib.:
#   ../Parameters/Robot
import Parameters.Robot
#   ../Transformation/Utilities/Mathematics
import Transformation.Utilities.Mathematics as Mathematics
#   ../Utilities/Dataset
import Utilities.Dataset as Dataset

"""
Description:
    Initialization of constants.
"""
# Set the structure of the main parameters of the robot.
CONST_ROBOT_TYPE = Parameters.Robot.EPSON_LS3_B401S_Str
# The name of the dataset to be checked (without extension).
#   Note:
#       The name can also be passed as the first command line argument,
#       e.g. $ python check_dataset.py Test_N_10000
CONST_DATASET_NAME = 'Config_N_1000'
# Number of bins of the radial histogram.
CONST_NUM_OF_BINS = 8
# Continuity check of the labels.
#   The number of nearest neighbours of each position.
CONST_NUM_OF_NEIGHBOURS = 5
#   The maximum distance between the neighbours.
#       Unit: [meter]
CONST_NEIGHBOUR_DISTANCE = 0.005
#   The joint difference that is considered a discontinuity (jump), e.g. a jump of 2*PI.
#       Unit: [radian]
CONST_JUMP_THRESHOLD = Mathematics.CONST_MATH_PI

def main():
    """
    Description:
        A program to check the consistency of a dataset generated for the selected robotic structure.

        The program displays the number of samples per configuration, the share of samples within
        the joint limits, the maximum forward kinematics (FK) error of the labels, the ranges of
        the joint positions, and a radial histogram of the positions.

        Note:
            The structures of the robot are defined below:
                ../Parameters/Robot.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('FCNN_Inverse_Kinematics')[0] + 'FCNN_Inverse_Kinematics'

    # Initialization of the structure of the main parameters of the robot.
    Robot_Str = CONST_ROBOT_TYPE

    # The name of the dataset.
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else CONST_DATASET_NAME

    # Read the data from a file (*.zip).
    data = Dataset.Load(f'{project_folder}/Data/Dataset/{dataset_name}').values
    p = data[:, 0:2]; cfg = data[:, 2].astype(np.int64); theta = data[:, 3:]

    # Check the individual samples.
    info = Dataset.Check_Samples(p, cfg, theta, Robot_Str)

    print(f'[INFO] Dataset: {dataset_name}')
    print(f'[INFO] Number of samples: {data.shape[0]}')
    for cfg_i in np.unique(cfg):
        print(f'[INFO]  cfg = {cfg_i}: {np.sum(cfg == cfg_i)} samples')
    print(f'[INFO] Number of unique positions: {np.unique(p, axis=0).shape[0]}')
    print(f'[INFO] Number of duplicate samples (x, y, cfg): {data.shape[0] - np.unique(data[:, 0:3], axis=0).shape[0]}')

    print(f'[INFO] Share of samples within the joint limits: {100.0 * np.mean(info["within_limits"]):.4f} %')
    print(f'[INFO] Share of samples with a consistent configuration (cfg = 0 -> th_1 >= 0, cfg = 1 -> th_1 <= 0): {100.0 * np.mean(info["cfg_consistent"]):.4f} %')
    print(f'[INFO] FK error ||FK(theta) - p||: max = {np.max(info["fk_error"]):.3e} m, mean = {np.mean(info["fk_error"]):.3e} m')
    print(f'[INFO] Number of samples with FK error >= {Dataset.CONST_FK_TOLERANCE} m: {np.sum(info["fk_error"] >= Dataset.CONST_FK_TOLERANCE)}')

    print('[INFO] Ranges of the joint positions in degrees (data / limit):')
    for i, (th_i, th_i_limit) in enumerate(zip(theta.T, Robot_Str.Theta.Limit)):
        print(f'[INFO]  th_{i}: [{np.degrees(np.min(th_i)):8.3f}, {np.degrees(np.max(th_i)):8.3f}] / '
              f'[{np.degrees(th_i_limit[0]):8.3f}, {np.degrees(th_i_limit[1]):8.3f}], '
              f'out of limit: {np.sum((th_i < th_i_limit[0]) | (th_i > th_i_limit[1]))}')

    # Radial histogram of the unique positions.
    #   Note:
    #       The density is the number of positions per unit area of the ring, normalized
    #       by the mean density. For a Cartesian-uniform dataset, the density is approximately
    #       constant in the rings that are fully covered by the joint limits.
    r = np.linalg.norm(np.unique(p, axis=0), axis=1)
    (count, edges) = np.histogram(r, bins=CONST_NUM_OF_BINS)
    density = count / (Mathematics.CONST_MATH_PI * (edges[1:]**2 - edges[:-1]**2))
    print('[INFO] Radial histogram of the unique positions (inner -> outer):')
    for count_i, r_0, r_1, density_i in zip(count, edges[:-1], edges[1:], density / np.mean(density)):
        print(f'[INFO]  r = [{r_0:.4f}, {r_1:.4f}] m: {count_i:7d} positions, relative area density = {density_i:.3f}')

    # Continuity check of the labels.
    #   Note:
    #       For each configuration, the joint positions of neighbouring positions (closer than
    #       CONST_NEIGHBOUR_DISTANCE) must be close to each other. A jump of approximately 2*PI
    #       indicates a discontinuity of the target function.
    print(f'[INFO] Continuity check: |delta theta| between neighbours closer than {1000.0 * CONST_NEIGHBOUR_DISTANCE:.1f} mm '
          f'(k = {CONST_NUM_OF_NEIGHBOURS}):')
    is_continuous = True
    for cfg_i in np.unique(cfg):
        p_i = p[cfg == cfg_i]; theta_i = theta[cfg == cfg_i]

        # Find the nearest neighbours (the first neighbour is the position itself).
        (distance, idx) = scipy.spatial.cKDTree(p_i).query(p_i, k=CONST_NUM_OF_NEIGHBOURS + 1)
        (row, col) = np.nonzero((distance[:, 1:] < CONST_NEIGHBOUR_DISTANCE) & np.isfinite(distance[:, 1:]))
        if row.size == 0:
            print(f'[WARNING]  cfg = {cfg_i}: no neighbours closer than {1000.0 * CONST_NEIGHBOUR_DISTANCE:.1f} mm were found.')
            continue

        # Absolute joint difference between the neighbours.
        delta_theta = np.abs(theta_i[row] - theta_i[idx[row, col + 1]])
        print(f'[INFO]  cfg = {cfg_i}: {row.size} neighbour pairs')
        for j, delta_theta_j in enumerate(delta_theta.T):
            print(f'[INFO]   th_{j}: max = {np.degrees(np.max(delta_theta_j)):8.4f} deg, '
                  f'p99 = {np.degrees(np.percentile(delta_theta_j, 99)):8.4f} deg, '
                  f'jumps (> {np.degrees(CONST_JUMP_THRESHOLD):.0f} deg) = {np.sum(delta_theta_j > CONST_JUMP_THRESHOLD)}')
        is_continuous &= bool(np.all(delta_theta <= CONST_JUMP_THRESHOLD))

    if np.all(info['within_limits']) and np.all(info['cfg_consistent']) and np.all(info['fk_error'] < Dataset.CONST_FK_TOLERANCE) \
       and is_continuous:
        print('[INFO] The dataset is consistent.')
    else:
        print('[WARNING] The dataset is not consistent.')

if __name__ == "__main__":
    sys.exit(main())
