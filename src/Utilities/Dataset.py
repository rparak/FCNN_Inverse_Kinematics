# OS (Operating system interfaces)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pandas (Data analysis and manipulation) [pip3 install pandas]
import pandas as pd
# Typing (Support for type hints)
import typing as tp
# Custom Lib.:
#   ../Kinematics/Core
import Kinematics.Core
#   ../Parameters/Robot
import Parameters.Robot

"""
Description:
    Initialization of constants.
"""
# The maximum allowed absolute position error between the position of the stored sample
# and the forward kinematics (FK) of the stored joint positions.
#   Unit: [meter]
CONST_FK_TOLERANCE = 1e-6
# The number of candidate positions generated in one batch, expressed as a multiple
# of the desired number of samples.
CONST_BATCH_FACTOR = 2

def Get_Column_Names(Robot_Parameters_Str: Parameters.Robot) -> tp.List[str]:
    """
    Description:
        Get the names of the columns of the dataset.

    Args:
        (1) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Vector<string>]: The names of the columns.
                                        Note:
                                            'x_coord', 'y_coord': Coordinates of the x-axis, y-axis (in meters).
                                            'cfg': The configuration of the IK solution (0: theta_{2} >= 0, 1: theta_{2} < 0).
                                            'th_0', .., 'th_m': Absolute position of the robot's joints (in radians).
    """

    return ['x_coord', 'y_coord', 'cfg'] + [f'th_{i}' for i in range(Robot_Parameters_Str.Theta.Zero.size)]

def Check_Samples(p: tp.List[float], cfg: tp.List[int], theta: tp.List[float], Robot_Parameters_Str: Parameters.Robot) -> tp.Dict:
    """
    Description:
        Function to check the consistency of the samples of the dataset.

    Args:
        (1) p [Vector<float> nx2]: Position of the TCP (tool center point) in the x, y axes.
                                   Unit: [meter]
        (2) cfg [Vector<int> 1xn]: The configuration of the IK solution.
        (3) theta [Vector<float> nxm]: Absolute joint position in radians.
                                       Unit: [radian]
        (4) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [Dictionary {'fk_error': Vector<float> 1xn, 'within_limits': Vector<bool> 1xn,
                                   'cfg_consistent': Vector<bool> 1xn}]: Information about the individual samples.
                                                                         Note:
                                                                            'fk_error': Absolute position error ||FK(theta) - p|| in meters.
                                                                            'within_limits': The sample is within the joint limits.
                                                                            'cfg_consistent': The sign of theta_{2} corresponds to the configuration.
    """

    theta = np.asarray(theta, dtype=np.float64); cfg = np.asarray(cfg).astype(np.int64)

    return {'fk_error': np.linalg.norm(Kinematics.Core.Forward_Kinematics(theta, Robot_Parameters_Str)[1] - np.asarray(p, dtype=np.float64), axis=-1),
            'within_limits': Kinematics.Core.Is_Within_Joint_Limits(theta, Robot_Parameters_Str),
            'cfg_consistent': np.where(cfg == 0, theta[:, 1] >= 0.0, theta[:, 1] <= 0.0)}

def Generate(N: int, seed: int, Robot_Parameters_Str: Parameters.Robot) -> pd.DataFrame:
    """
    Description:
        A function to generate the dataset of the selected robotic structure.

        The positions of the TCP (tool center point) are sampled uniformly in the Cartesian workspace
        using rejection sampling in the bounding box of the workspace. For each position, both IK
        solutions are computed analytically, wrapped into the joint limits, and kept only if they
        are within the joint limits.

        The structure of the dataset is described below.
            Input of the NN:  x -> Position(x, y); configuration_id(0, 1)
            Output of the NN: y -> theta(0 .. m)

    Args:
        (1) N [int]: The number of samples (rows) to be generated.
        (2) seed [int]: The seed of the random number generator.
        (3) Robot_Parameters_Str [Robot_Parameters_Str(object)]: The structure of the main parameters of the robot.

    Returns:
        (1) parameter [pd.DataFrame]: The generated dataset (float32 values, 'cfg' as uint8).
    """

    # Initialization of the random number generator.
    rng = np.random.default_rng(seed)

    # The bounding box (radius) of the workspace.
    #   Unit: [meter]
    r_min = np.abs(Robot_Parameters_Str.DH.Standard[0, 1] - Robot_Parameters_Str.DH.Standard[1, 1]).astype(np.float64)
    r_max = np.abs(Robot_Parameters_Str.DH.Standard[0, 1] + Robot_Parameters_Str.DH.Standard[1, 1]).astype(np.float64)

    p_data = []; cfg_data = []; theta_data = []; num_of_samples = 0; num_of_candidates = 0; num_of_accepted = 0
    while num_of_samples < N:
        # Random generation of positions in the bounding box of the workspace.
        p_tmp = rng.uniform(-r_max, r_max, size=(max(CONST_BATCH_FACTOR * N, 1000), 2))
        num_of_candidates += p_tmp.shape[0]

        # Keep only the positions within the geometric workspace (annulus).
        r_tmp = np.linalg.norm(p_tmp, axis=1)
        p_tmp = p_tmp[(r_tmp >= r_min) & (r_tmp <= r_max)]

        # Obtain both solutions of the absolute positions of the joints and wrap them into the joint limits.
        th_tmp = Kinematics.Core.Wrap_To_Joint_Limits(Kinematics.Core.Inverse_Kinematics(p_tmp, Robot_Parameters_Str)[1],
                                                      Robot_Parameters_Str)

        # Keep only the solutions within the joint limits.
        #   Note:
        #       The samples are ordered per position: configuration 0, then configuration 1.
        is_valid = Kinematics.Core.Is_Within_Joint_Limits(th_tmp, Robot_Parameters_Str)
        (p_idx, cfg_idx) = np.nonzero(is_valid)

        num_of_accepted += np.count_nonzero(np.any(is_valid, axis=1))
        p_data.append(p_tmp[p_idx]); cfg_data.append(cfg_idx); theta_data.append(th_tmp[p_idx, cfg_idx])
        num_of_samples += p_idx.size

    # Concatenate the batches and keep the desired number of samples.
    p_data = np.concatenate(p_data)[:N]; cfg_data = np.concatenate(cfg_data)[:N]; theta_data = np.concatenate(theta_data)[:N]

    # Store the values as float32.
    data = np.column_stack((p_data, cfg_data, theta_data)).astype(np.float32)

    # Remove duplicates of the input data (position, configuration), if any.
    (_, idx) = np.unique(data[:, 0:3], axis=0, return_index=True)
    if idx.size != data.shape[0]:
        print(f'[WARNING] {data.shape[0] - idx.size} duplicate samples have been removed.')
    data = data[np.sort(idx)]

    # Check every stored sample. Fail loudly if any sample is not consistent.
    info = Check_Samples(data[:, 0:2], data[:, 2], data[:, 3:], Robot_Parameters_Str)
    if not (np.all(info['fk_error'] < CONST_FK_TOLERANCE) and np.all(info['within_limits']) and np.all(info['cfg_consistent'])):
        print(f'[ERROR] The generated dataset is not consistent.')
        print(f'[ERROR]  Max. FK error: {np.max(info["fk_error"])} in meters (tolerance = {CONST_FK_TOLERANCE}).')
        print(f'[ERROR]  Number of samples out of the joint limits: {np.sum(~info["within_limits"])}.')
        print(f'[ERROR]  Number of samples with an inconsistent configuration: {np.sum(~info["cfg_consistent"])}.')
        raise AssertionError('The generated dataset is not consistent.')

    print(f'[INFO] Acceptance rate of the candidate positions: {100.0 * num_of_accepted / num_of_candidates:.2f} %')

    df = pd.DataFrame(data, columns=Get_Column_Names(Robot_Parameters_Str))
    df['cfg'] = df['cfg'].astype('uint8')

    return df

def Save(file_path: str, data: pd.DataFrame) -> None:
    """
    Description:
        A function to save the dataset to the file (*.zip) which contains a 'csv' file with the data.

    Args:
        (1) file_path [string]: The specified path of the file without extension (format).
        (2) data [pd.DataFrame]: The dataset.
    """

    # Remove the file if it already exists.
    if os.path.isfile(f'{file_path}.zip'):
        os.remove(f'{file_path}.zip')

    # Define the compression of the output data.
    compression_dict = dict(method='zip', archive_name=f'{os.path.basename(file_path)}.csv')

    # Save the data to the file (*.zip).
    #   Note:
    #       f'{file_path}.zip' contains f'{file_name}.csv'
    data.to_csv(f'{file_path}.zip', index=False, compression=compression_dict)

def Load(file_path: str) -> pd.DataFrame:
    """
    Description:
        A function to load the dataset from the file (*.zip) which contains a 'csv' file with the data.

    Args:
        (1) file_path [string]: The specified path of the file without extension (format).

    Returns:
        (1) parameter [pd.DataFrame]: The dataset.
    """

    return pd.read_csv(f'{file_path}.zip', compression='zip')
