import opensim as osim
from pathlib import Path
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import helper_fcts as hf
import logging

osim.Logger.setLevel(osim.Logger.Level_Info)
logger = logging.getLogger(__name__)

# Constants
IMU_QUAT_AXES = ['QX', 'QY', 'QZ', 'QW']
IMU_TO_OPENSIM_ROTATION = R.from_euler(seq='x', angles=-90, degrees=True)


def read_trc_file(filename: str) -> pd.DataFrame:
    """
    Reads a .trc (marker trajectory) file in OpenSim format and returns a DataFrame.

    Parameters:
        filename (str): Path to the .trc file.

    Returns:
        pd.DataFrame: Marker positions with columns ['Time', 'marker1_x', 'marker1_y', ..., 'markerN_z'].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the data lines do not match expected column format.
    """
    if not os.path.exists(filename):
        logger.error(f"TRC file not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'r') as file_id:
        all_lines = file_id.readlines()

    marker_names = all_lines[3].strip().split('\t')
    marker_names = [m for m in marker_names if m not in ('Frame#', 'Time', '\n', '')]

    col_names = ['Time']
    for m in marker_names:
        col_names.extend([f'{m}_x', f'{m}_y', f'{m}_z'])

    data = []
    for line_idx in range(5, len(all_lines)):
        d = all_lines[line_idx].strip().split('\t')[1:]
        if len(d) != len(col_names):
            logger.error(f"TRC line {line_idx + 1} column mismatch: {len(d)} vs {len(col_names)}")
            raise ValueError(f'Line {line_idx + 1} has {len(d)} columns, expected {len(col_names)}')
        data.append(d)

    return pd.DataFrame(data=data, columns=col_names).astype('float', errors='ignore')


def extract_marker_coordinates(marker_data: pd.DataFrame, marker_name: str, pos: int) -> np.ndarray:
    """
    Extracts the 3D coordinates of a specific marker at a given frame index.

    Parameters:
        marker_data (pd.DataFrame): Marker data as DataFrame.
        marker_name (str): Name of the marker (without _x/_y/_z).
        pos (int): Row index (frame number) to extract.

    Returns:
        np.ndarray: 3D coordinates [x, y, z] of the marker at the specified frame.
    """
    return marker_data[[f'{marker_name}_{axis}' for axis in ['x', 'y', 'z']]].iloc[pos].to_numpy()


def calculate_heading_correction(marker_data: pd.DataFrame, marker_names: list[str], R_imu: R, pos: str) -> R:
    """
    Computes a heading correction rotation for an IMU based on marker geometry in a static pose.

    The correction aligns the IMU heading with the segment's marker-defined orientation.

    Parameters:
        marker_data (pd.DataFrame): Marker positions for a static frame (e.g., first frame).
        marker_names (list[str]): List of 3 marker names defining the local segment frame.
        R_imu (Rotation): Rotation object (scipy.spatial.transform.Rotation) from IMU.
        pos (str): Segment name, used to apply special rules (e.g., for 'toes_r').

    Returns:
        Rotation: A scipy Rotation object representing the heading correction (rotation around Y-axis).
    """
    i = 0

    if pos == 'toes_r':
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0], i)
        imu_marker_extra_2 = extract_marker_coordinates(marker_data, marker_names[1], i)
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2], i)
        imu_marker_21 = imu_marker_1 - imu_marker_extra_2
        imu_marker_2 = imu_marker_extra_2 + imu_marker_21 / 2
    else:
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0], i)
        imu_marker_2 = extract_marker_coordinates(marker_data, marker_names[1], i)
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2], i)

    marker_x = imu_marker_1 - imu_marker_2
    marker_y = imu_marker_3 - imu_marker_2
    marker_z = np.cross(marker_x, marker_y)

    # Normalize axes
    marker_x /= np.linalg.norm(marker_x)
    marker_y /= np.linalg.norm(marker_y)
    marker_z /= np.linalg.norm(marker_z)

    imu_cs = np.column_stack([marker_x, marker_y, marker_z])
    qualisys_cs = np.eye(3)
    R_marker = R.from_matrix(np.linalg.inv(qualisys_cs) @ imu_cs)

    q_diff = R_marker * R_imu[0].inv()
    euler = q_diff.as_euler('yxz', degrees=True)
    return R.from_euler('y', euler[0], degrees=True)


def extract_segment_orientations(model_path: str) -> dict[str, R]:
    """
    Extracts global segment orientations (rotation matrices) from an OpenSim model.

    Parameters:
        model_path (str): Path to the .osim model file.

    Returns:
        dict[str, Rotation]: Dictionary mapping segment (body) names to scipy Rotation objects.
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
    """
    Loads and filters marker data from a .trc file for a specific subject and exercise.

    Parameters:
        ik_dir (Path): Path to the directory containing inverse kinematics input files.
        subject_name (str): Unique identifier of the subject (e.g., 'subject01').
        exercise (str): Name of the exercise (e.g., 'walking').
        start_ts (float): Timestamp (in seconds) from which the marker data should be considered.

    Returns:
        pd.DataFrame: Marker data as a DataFrame, filtered to include only data from `start_ts` onward.
    """
    marker_path = ik_dir / f'marker_data_osim_format_{subject_name}_{exercise}.trc'
    marker_data = read_trc_file(str(marker_path))
    return marker_data[marker_data['Time'] >= start_ts]


def load_imu_data(measurement_path: Path, subject_name: str, exercise: str, start_ts: float) -> pd.DataFrame:
    """
    Loads and filters IMU quaternion data from a CSV file for a specific subject and exercise.

    Parameters:
        measurement_path (Path): Base path to measurement files.
        subject_name (str): Subject identifier.
        exercise (str): Name of the exercise.
        start_ts (float): Start timestamp (in seconds) to crop the IMU data.

    Returns:
        pd.DataFrame: Filtered IMU data starting from `start_ts`.
    """
    imu_data = pd.read_csv(measurement_path / f'xsens_imu_data_{subject_name}_{exercise}.csv')
    return imu_data[imu_data['time [s]'] >= start_ts].reset_index(drop=True)


def compute_registered_imu_orientations(cfg, marker_data, imu_data, segment_orientations) -> dict:
    """
    Registers raw IMU orientations to the OpenSim segment coordinate systems.

    This includes:
    - Rotation into OpenSim convention
    - Heading correction using a static pose and marker positions
    - Segment calibration using OpenSim model orientation

    Parameters:
        cfg (dict): Configuration dictionary containing 'inverse_kinematics' with IMU and marker mappings.
        marker_data (pd.DataFrame): Filtered Qualisys marker data for static pose alignment.
        imu_data (pd.DataFrame): Time-aligned IMU quaternion data.
        segment_orientations (dict): Reference segment orientations from the OpenSim model.

    Returns:
        dict: Dictionary mapping segment names to numpy arrays of registered quaternion data (Nx4).
    """
    if 'inverse_kinematics' not in cfg:
        logger.error("Missing 'inverse_kinematics' section in config")
        raise KeyError("Missing 'inverse_kinematics' section in config")

    registered_imu_data = {}
    for pos, infos in cfg['inverse_kinematics'].items():
        quat_cols = [f'{infos["imu_name"]}_{axis}' for axis in IMU_QUAT_AXES]
        R_imu = R.from_quat(imu_data[quat_cols].to_numpy())
        R_imu = IMU_TO_OPENSIM_ROTATION * R_imu
        q_heading_correction = calculate_heading_correction(marker_data, infos['marker_names'], R_imu, pos)
        R_corrected = q_heading_correction * R_imu
        calibration = segment_orientations[pos].inv() * R_corrected[0]
        logger.info(f'Segment correction for {pos}:\n{calibration.as_matrix()}')
        R_segment = R_corrected * calibration.inv()
        registered_imu_data[pos] = R_segment.as_quat()

    return registered_imu_data


def build_imu_dataframe(registered_imu_data: dict, cfg, time: np.ndarray) -> pd.DataFrame:
    """
    Constructs a DataFrame with registered IMU quaternion components for export.

    Parameters:
        registered_imu_data (dict): Dictionary with quaternion arrays (shape Nx4) per segment.
        cfg (dict): Configuration containing IMU name mappings.
        time (np.ndarray): 1D array of time values corresponding to each quaternion row.

    Returns:
        pd.DataFrame: A DataFrame with time column and registered quaternion columns.
    """
    df_data = []
    col_names = []

    for pos, quat in registered_imu_data.items():
        imu_name = cfg["inverse_kinematics"][pos]["imu_name"]
        col_names += [f"{imu_name}_{axis}" for axis in IMU_QUAT_AXES]
        df_data += [list(quat[:, i]) for i in range(4)]

    imu_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    imu_df['time [s]'] = time
    return imu_df[['time [s]'] + col_names]


def write_imu_quat_sto(registered_imu_data: dict, time: np.ndarray, subject_name: str, exercise: str,
                       output_path: Path):
    """
    Writes the registered IMU data to an OpenSim-compatible .sto file.

    Parameters:
        registered_imu_data (dict): Dictionary with quaternion arrays (Nx4) per segment.
        time (np.ndarray): Time vector for the motion.
        subject_name (str): Subject identifier used in filename.
        exercise (str): Exercise name used in filename.
        output_path (Path): Directory where the .sto file will be saved.

    Returns:
        str: The filename of the written .sto file (not full path).
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
    """
    Runs OpenSim's IMU Inverse Kinematics tool using the given input and saves results.

    Parameters:
        model_path (Path): Path to the scaled OpenSim model (.osim).
        orientations_path (Path): Path to the .sto file containing registered IMU orientations.
        results_dir (Path): Directory where IK results will be saved.
        start_ts (float): Start timestamp for the IK computation.
        stop_ts (float): Stop timestamp for the IK computation.
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
    """
    Executes the inverse kinematics pipeline using IMU data and marker-based heading correction.

    Parameters:
        measurement_path (Path): Path to measurement root directory.
        cfg (dict): Configuration dictionary containing marker-IMU mapping and timestamps.
        imu_sample_rate (float): Sampling rate of the IMU data (Hz).
        subject_name (str): Name/ID of the subject.
        exercise (str): Name of the exercise.
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
