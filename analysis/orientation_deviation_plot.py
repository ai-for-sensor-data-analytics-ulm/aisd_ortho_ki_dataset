from typing import Optional, Union, Tuple, List, Dict, Any
import json
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import processing.helper_fcts as hfp
import matplotlib
matplotlib.use('Agg')

SAMPLERATE = 100.0


def extract_marker_coordinates(
    marker_data: pd.DataFrame,
    marker_name: str,
    pos: Optional[int] = None,
) -> np.ndarray:
    """Return coordinates of a marker.

    Parameters
    ----------
    marker_data : pd.DataFrame
        DataFrame with marker trajectories.
    marker_name : str
        Base name of the marker (e.g. ``"ASIS"``).
    pos : int, optional
        Frame index to extract. If ``None`` the full trajectory is returned.

    Returns
    -------
    np.ndarray
        Marker coordinates with shape ``(N, 3)`` or ``(3,)``.
    """
    axes = ['x', 'y', 'z']
    if pos is None:
        return marker_data[[f'{marker_name}_{axis}' for axis in axes]].to_numpy()
    else:
        return marker_data[[f'{marker_name}_{axis}' for axis in axes]].iloc[pos].to_numpy()


def fix_marker_coordinate_zeros(marker_coordinates: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Replace rows with zeros by the preceding valid value.

    Parameters
    ----------
    marker_coordinates : np.ndarray
        Marker coordinate array of shape ``(N, 3)``.

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        Corrected array and list of indices that were replaced.
    """
    indices = np.argwhere(marker_coordinates == 0.0)
    if len(indices) == 0:
        return marker_coordinates, []
    else:
        zero_ind = sorted(list(set(indices[:, 0])))
        for ind in zero_ind:
            marker_coordinates[ind, :] = marker_coordinates[ind - 1, :]
        return marker_coordinates, zero_ind


def calculate_orientation_deviation(
    marker_data: pd.DataFrame,
    marker_names: List[str],
    R_imu: R,
    pos: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Compute orientation deviation between markers and IMUs.

    Parameters
    ----------
    marker_data : pd.DataFrame
        Marker data in OpenSim format.
    marker_names : list[str]
        Names of the markers defining the local coordinate system.
    R_imu : Rotation
        IMU orientations in the OpenSim frame.
    pos : str
        Body segment name. Special handling for ``"toes_r"``.

    Returns
    -------
    tuple
        ``(angles, marker_quat, imu_quat, error_indices)`` where ``angles`` is
        the deviation in degrees (``N,``), ``marker_quat`` and ``imu_quat`` are
        quaternion arrays of shape ``(N, 4)`` and ``error_indices`` contains the
        indices of corrected samples.
    """
    error_indices = []

    if pos == 'toes_r':
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0])
        imu_marker_extra_2 = extract_marker_coordinates(marker_data, marker_names[1])
        _, zero_ind = fix_marker_coordinate_zeros(imu_marker_extra_2)
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2])
        imu_marker_21 = imu_marker_1 - imu_marker_extra_2
        imu_marker_2 = imu_marker_extra_2 + imu_marker_21 / 2
        imu_marker_2[zero_ind] = [0.0, 0.0, 0.0]
    else:
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0])
        imu_marker_2 = extract_marker_coordinates(marker_data, marker_names[1])
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2])

    if np.any(imu_marker_1 == 0.0) or np.any(imu_marker_2 == 0.0) or np.any(imu_marker_3 == 0.0):
        imu_marker_1, error_ind = fix_marker_coordinate_zeros(imu_marker_1)
        error_indices += error_ind
        imu_marker_2, error_ind = fix_marker_coordinate_zeros(imu_marker_2)
        error_indices += error_ind
        imu_marker_3, error_ind = fix_marker_coordinate_zeros(imu_marker_3)
        error_indices += error_ind

    marker_x_axis = imu_marker_1 - imu_marker_2
    marker_y_axis = imu_marker_3 - imu_marker_2
    marker_z_axis = np.cross(marker_x_axis, marker_y_axis)

    marker_x_axis = marker_x_axis / np.linalg.norm(marker_x_axis, axis=1).reshape(-1, 1)
    marker_y_axis = marker_y_axis / np.linalg.norm(marker_y_axis, axis=1).reshape(-1, 1)
    marker_z_axis = marker_z_axis / np.linalg.norm(marker_z_axis, axis=1).reshape(-1, 1)

    imu_marker_cs = np.zeros((marker_x_axis.shape[0], 3, 3))
    for i in range(marker_x_axis.shape[0]):
        imu_marker_cs[i, :, 0] = marker_x_axis[i, :]
        imu_marker_cs[i, :, 1] = marker_y_axis[i, :]
        imu_marker_cs[i, :, 2] = marker_z_axis[i, :]
    qualisys_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    qualisys_cs = np.tile(qualisys_cs, (marker_x_axis.shape[0], 1, 1))
    R_imu_marker = R.from_matrix(np.matmul(qualisys_cs, imu_marker_cs))

    # check if Orientation length is the same
    lens = {'R_imu_marker': len(R_imu_marker), 'R_imu': len(R_imu)}
    min_length = min(len(R_imu_marker), len(R_imu))
    R_imu_marker = R_imu_marker[:min_length]
    R_imu = R_imu[:min_length]

    q_diff = R_imu_marker * R_imu.inv()
    return np.linalg.norm(q_diff.as_rotvec(degrees=True),
                          axis=1), R_imu_marker.as_quat(), R_imu.as_quat(), error_indices


def calculate_all_deviations(
    measurement_path: Path,
    cfg: Dict[str, Any],
    subject_name: str,
    exercise: str
) -> Dict[str, Dict[str, Union[np.ndarray, List[int]]]]:
    """Compute deviations for all configured body segments.

    Parameters
    ----------
    measurement_path : Path
        Base directory of the measurement.
    cfg : dict
        Configuration dictionary with metadata and IMU-to-marker mapping.
    subject_name : str
        Subject identifier.
    exercise : str
        Name of the performed exercise.

    Returns
    -------
    dict
        Mapping of body segments to dictionaries with deviation angles,
        marker orientations, IMU orientations and error indices.
    """

    start_ts = cfg['start_ts']
    ik_dir = measurement_path / 'ik_imus'

    # load and preprocess marker data
    marker_data = hfp.read_trc_file(str(ik_dir / f'marker_data_osim_format_{subject_name}_{exercise}.trc'))
    marker_data = marker_data[(marker_data['Time'] >= start_ts)]

    imu_data = pd.read_csv(measurement_path / f'xsens_imu_data_{subject_name}_{exercise}.csv', sep=',')
    imu_data = imu_data[(imu_data['time [s]'] >= start_ts)]
    time = imu_data['time [s]'].to_numpy()

    quat_data = pd.DataFrame()
    quat_data['time'] = time

    deviation_data = {}
    for pos, infos in cfg['inverse_kinematics'].items():
        quat_cols = [f'{infos["imu_name"]}_{axis}' for axis in hfp.IMU_QUAT_AXES]
        R_imu = R.from_quat(imu_data[quat_cols].to_numpy())
        R_imu = hfp.IMU_TO_OPENSIM_ROTATION * R_imu
        q_heading_correction = hfp.calculate_heading_correction(marker_data, infos['marker_names'], R_imu, pos)
        R_corrected = q_heading_correction * R_imu


        deviation_angles, marker_orientations, imu_orientations, error_indices = calculate_orientation_deviation(
            marker_data, infos['marker_names'], R_corrected, pos)

        deviation_data[pos] = {'deviations_angles': deviation_angles, 'marker_orientations': marker_orientations,
                                    'imu_orientations': imu_orientations, 'error_indices': error_indices,}

    return deviation_data


def plot_sample(
    data: Dict[str, Dict[str, Union[np.ndarray, List[int]]]],
    subject_name: str,
    exercise_name: str,
    start_ind: int = 0,
    end_ind: int = -1
) -> None:
    """Plot deviation angles for a sample.

    Parameters
    ----------
    data : dict
        Output of :func:`calculate_all_deviations`.
    subject_name : str
        Subject identifier.
    exercise_name : str
        Exercise identifier.
    start_ind : int, optional
        First sample index to plot, by default ``0``.
    end_ind : int, optional
        Last sample index to plot, by default ``-1``.
    """
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharey=True, constrained_layout=True)
    axs = axs.flatten()
    for i, (pos, d) in enumerate(data.items()):
        max_ind = len(d['deviations_angles'])
        deviations = d['deviations_angles']
        errors = list(set(d['error_indices']))
        errors = [x for x in errors if x < max_ind]
        deviations[errors] = 0.0
        time = np.arange(0, len(deviations))[start_ind:end_ind]
        time = time / SAMPLERATE
        axs[i].plot(time, deviations[start_ind:end_ind])
        axs[i].set_title(f'{pos}')
        axs[i].set_ylabel('Deviation in Degrees')
        axs[i].set_xlabel('Time in Seconds')
    fig.suptitle(f'Deviations for measurement {subject_name} - {exercise_name}')
    plt.show()
    plt.savefig(f'subject_{subject_name}_exercise_{exercise_name}_deviations.png', format='pdf')
    plt.close()


def main():
    """Run the command line interface.

    The routine loads metadata, computes orientation deviations and plots the
    results for a given subject and exercise.
    """

    parser = argparse.ArgumentParser(
        description="Calculate and plot orientation deviations for a given subject and exercise.")
    parser.add_argument("--subject", type=str, required=True, help="Subject identifier (e.g., darryl)")
    parser.add_argument("--exercise", type=str, required=True, help="Exercise name (e.g., ng)")
    parser.add_argument("--base_path", type=str, default="../data",
                        help="Base directory where subject folders are stored")

    args = parser.parse_args()

    subject = args.subject
    exercise = args.exercise
    base_path = Path(args.base_path)
    measurement_path = base_path / subject / exercise
    metadata_path = measurement_path / f'metadata_{subject}_{exercise}.json'


    with open(metadata_path, 'r') as stream:
        processing_config = json.load(stream)

    deviation_data = calculate_all_deviations(
        measurement_path=base_path,
        cfg=processing_config,
        subject_name=subject,
        exercise=exercise
    )

    plot_sample(deviation_data, subject, exercise)

if __name__ == "__main__":
    main()