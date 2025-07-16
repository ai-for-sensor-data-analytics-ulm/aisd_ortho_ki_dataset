"""Utilities for running IMU-based inverse kinematics."""

import logging
import os
from pathlib import Path

import numpy as np
import opensim as osim
import pandas as pd
from scipy.spatial.transform import Rotation as R

import helper_fcts as hf

osim.Logger.setLevel(osim.Logger.Level_Info)
logger = logging.getLogger(__name__)


def extract_segment_orientations(model_path: str) -> dict[str, R]:
    """Load global segment orientations from an OpenSim model.

    Parameters
    ----------
    model_path : str
        Path to the ``.osim`` model file.

    Returns
    -------
    dict[str, Rotation]
        Mapping of segment names to rotations.
    """
    model = osim.Model(model_path)
    state = model.initSystem()
    segment_orientations = {}

    for body in model.getBodySet():
        body_name = body.getName()
        rotation = body.getTransformInGround(state).R().asMat33()
        rotation_matrix = np.array([
            [rotation.get(0, 0), rotation.get(0, 1), rotation.get(0, 2)],
            [rotation.get(1, 0), rotation.get(1, 1), rotation.get(1, 2)],
            [rotation.get(2, 0), rotation.get(2, 1), rotation.get(2, 2)]
        ])
        segment_orientations[body_name] = R.from_matrix(rotation_matrix)
    return segment_orientations



def load_marker_data(ik_dir: Path, subject_name: str, exercise: str, start_ts: float) -> pd.DataFrame:
    """Load and crop marker data for an exercise.

    Parameters
    ----------
    ik_dir : Path
        Directory containing inverse kinematics input files.
    subject_name : str
        Subject identifier.
    exercise : str
        Exercise name.
    start_ts : float
        Timestamp from which data should be used.

    Returns
    -------
    pandas.DataFrame
        Marker data starting at ``start_ts``.
    """
    marker_path = ik_dir / f'marker_data_osim_format_{subject_name}_{exercise}.trc'
    marker_data = hf.read_trc_file(str(marker_path))
    return marker_data[marker_data['Time'] >= start_ts]


def load_imu_data(measurement_path: Path, subject_name: str, exercise: str, start_ts: float) -> pd.DataFrame:
    """Load and crop IMU quaternion data.

    Parameters
    ----------
    measurement_path : Path
        Directory containing the raw CSV file.
    subject_name : str
        Subject identifier.
    exercise : str
        Exercise name.
    start_ts : float
        Timestamp from which to keep samples.

    Returns
    -------
    pandas.DataFrame
        IMU data starting at ``start_ts``.
    """
    imu_data = pd.read_csv(measurement_path / f'xsens_imu_data_{subject_name}_{exercise}.csv')
    return imu_data[imu_data['time [s]'] >= start_ts].reset_index(drop=True)


def compute_registered_imu_orientations(cfg, marker_data, imu_data, segment_orientations) -> dict:
    """Register raw IMU orientations to the OpenSim segment frames.

    The procedure rotates the raw data into the OpenSim convention, applies a
    heading correction using a static pose and finally calibrates each segment
    with its orientation from the model.

    Parameters
    ----------
    cfg : dict
        Configuration containing an ``inverse_kinematics`` section.
    marker_data : pandas.DataFrame
        Marker data for the static pose.
    imu_data : pandas.DataFrame
        IMU quaternion time series.
    segment_orientations : dict
        Reference segment orientations from the model.

    Returns
    -------
    dict
        Mapping of segment names to arrays of registered quaternions (``N x 4``).
    """
    if 'inverse_kinematics' not in cfg:
        logger.error("Missing 'inverse_kinematics' section in config")
        raise KeyError("Missing 'inverse_kinematics' section in config")

    registered_imu_data = {}
    for pos, infos in cfg['inverse_kinematics'].items():
        quat_cols = [f'{infos["imu_name"]}_{axis}' for axis in hf.IMU_QUAT_AXES]
        R_imu = R.from_quat(imu_data[quat_cols].to_numpy())
        R_imu = hf.IMU_TO_OPENSIM_ROTATION * R_imu
        q_heading_correction = hf.calculate_heading_correction(marker_data, infos['marker_names'], R_imu, pos)
        R_corrected = q_heading_correction * R_imu
        calibration = segment_orientations[pos].inv() * R_corrected[0]
        logger.info(f'Segment correction for {pos}:\n{calibration.as_matrix()}')
        R_segment = R_corrected * calibration.inv()
        registered_imu_data[pos] = R_segment.as_quat()

    return registered_imu_data


def build_imu_dataframe(registered_imu_data: dict, cfg, time: np.ndarray) -> pd.DataFrame:
    """Create a DataFrame with registered IMU quaternions.

    Parameters
    ----------
    registered_imu_data : dict
        Mapping of segment names to quaternion arrays.
    cfg : dict
        Configuration containing IMU name mappings.
    time : numpy.ndarray
        Time vector corresponding to the quaternions.

    Returns
    -------
    pandas.DataFrame
        DataFrame with time column and quaternion columns.
    """
    df_data = []
    col_names = []

    for pos, quat in registered_imu_data.items():
        imu_name = cfg["inverse_kinematics"][pos]["imu_name"]
        col_names += [f"{imu_name}_{axis}" for axis in hf.IMU_QUAT_AXES]
        df_data += [list(quat[:, i]) for i in range(4)]

    imu_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    imu_df['time [s]'] = time
    return imu_df[['time [s]'] + col_names]


def write_imu_quat_sto(registered_imu_data: dict, time: np.ndarray, subject_name: str, exercise: str,
                       output_path: Path):
    """Write registered IMU quaternions to an ``.sto`` file.

    Parameters
    ----------
    registered_imu_data : dict
        Mapping of segment names to quaternion arrays.
    time : numpy.ndarray
        Time vector of the motion.
    subject_name : str
        Subject identifier used in the filename.
    exercise : str
        Exercise name used in the filename.
    output_path : Path
        Directory where the file is written.

    Returns
    -------
    str
        Name of the created ``.sto`` file.
    """
    header = [
        'DataRate=100', '\nDataType=Quaternion', '\nversion=3',
        '\nOpenSimVersion=4.3-2021-08-27-4bc7ad9', '\nendheader\n'
    ]
    quat_data = pd.DataFrame({'time': time})

    for pos, quat in registered_imu_data.items():
        quat_data[pos] = [f'{q[3]}, {q[0]}, {q[1]}, {q[2]}' for q in quat]

    filename = f'segment_registered_imu_data_{subject_name}_{exercise}.sto'
    hf.write_to_mot_file(quat_data, header=header, filepath=output_path, filename=filename)
    return filename


def run_inverse_kinematics_tool(model_path: Path, orientations_path: Path, results_dir: Path, start_ts: float,
                                stop_ts: float):
    """Run OpenSim's IMU Inverse Kinematics tool.

    Parameters
    ----------
    model_path : Path
        Path to the scaled model.
    orientations_path : Path
        ``.sto`` file with registered IMU orientations.
    results_dir : Path
        Directory for the IK results.
    start_ts : float
        Start timestamp for the tool.
    stop_ts : float
        Stop timestamp for the tool.
    """
    imu_ik = osim.IMUInverseKinematicsTool()
    imu_ik.set_model_file(str(model_path))
    imu_ik.set_orientations_file(str(orientations_path))
    imu_ik.set_results_directory(str(results_dir))
    imu_ik.set_time_range(0, float(start_ts))
    imu_ik.set_time_range(1, float(stop_ts))
    imu_ik.run(False)


def perform_inverse_kinematics_w_imu_data(measurement_path: Path, cfg: dict, imu_sample_rate: float, subject_name: str,
                                          exercise: str):
    """Run the IMU-based inverse kinematics pipeline.

    Parameters
    ----------
    measurement_path : Path
        Path to the measurement directory.
    cfg : dict
        Configuration dictionary with marker/IMU mapping and timestamps.
    imu_sample_rate : float
        Sampling rate of the IMU data in hertz.
    subject_name : str
        Identifier of the subject.
    exercise : str
        Name of the exercise.
    """
    start_ts = cfg['start_ts']
    ik_dir = measurement_path / 'ik_imus'
    model_file = ik_dir / 'models' / f'scaled_model_initial_pose_{subject_name}_{exercise}.osim'
    results_dir = ik_dir / 'results_imu_ik'

    marker_data = load_marker_data(ik_dir, subject_name, exercise, start_ts)
    imu_data = load_imu_data(measurement_path, subject_name, exercise, start_ts)
    time = imu_data['time [s]'].to_numpy()
    stop_ts = time[-1]

    segment_orientations = extract_segment_orientations(str(model_file))
    registered_imu_data = compute_registered_imu_orientations(cfg, marker_data, imu_data, segment_orientations)

    imu_df = build_imu_dataframe(registered_imu_data, cfg, time)
    imu_df.to_csv(measurement_path / f'xsens_imu_data_segment_registered_{subject_name}_{exercise}.csv', index=False)

    sto_filename = write_imu_quat_sto(registered_imu_data, time, subject_name, exercise, results_dir)
    run_inverse_kinematics_tool(model_file, results_dir / sto_filename, results_dir, start_ts, stop_ts)

    return True
