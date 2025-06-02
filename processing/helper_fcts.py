import shutil
import os
import opensim as osim
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


IMU_QUAT_AXES = ['QX', 'QY', 'QZ', 'QW']
IMU_TO_OPENSIM_ROTATION = R.from_euler(seq='x', angles=-90, degrees=True)


def extract_subject_name_from_path(p: str | Path) -> str:
    """
    Extracts the subject name from a given path. Assumes it's the third segment (index 2).

    Parameters:
        p (str | Path): A path object or string representing the file path.

    Returns:
        str: Subject name extracted from the path.

    Raises:
        ValueError: If the path is too short.
    """
    parts = Path(p).parts
    if len(parts) < 3:
        raise ValueError(f"Path {p} does not contain enough segments.")
    return parts[2]


def extract_exercise_from_path(p: str | Path) -> str:
    """
    Extracts the exercise name from the last segment of a path.

    Parameters:
        p (str | Path): A path object or string representing the file path.

    Returns:
        str: Exercise name.
    """
    return Path(p).name


def copy_xml_to_directory(xml_path: Path, new_path: Path, delete: bool = False) -> Path:
    """
    Copies an XML file to a new path. Optionally deletes the source file.

    Parameters:
        xml_path (Path): Original XML file.
        new_path (Path): Destination path (including filename).
        delete (bool): If True, deletes the original file after copying.

    Returns:
        Path: Path to the copied file.
    """
    new_path = shutil.copy2(xml_path, new_path)
    if delete:
        os.remove(xml_path)
    return Path(new_path)


def change_model_name(model_path: Path, new_model_name: str):
    """
    Loads an OpenSim model, changes its name, and overwrites the file.

    Parameters:
        model_path (Path): Path to the .osim file.
        new_model_name (str): New name to assign to the model.
    """
    model = osim.Model(str(model_path))
    model.setName(new_model_name)
    model.printToXML(str(model_path))
    return


def write_to_mot_file(data: pd.DataFrame, header: list[str], filepath: Path, filename: str) -> bool:
    """
    Writes a .mot (motion) file in OpenSim format with given header and data.

    Parameters:
        data (pd.DataFrame): The motion data (time + channels).
        header (list[str]): List of header lines to write at the top of the file.
        filepath (Path): Directory where the file will be saved.
        filename (str): Name of the output file.

    Returns:
        bool: True if successful.
    """
    filepath = Path(filepath)
    filepath.mkdir(parents=True, exist_ok=True)
    full_path = filepath / filename

    with open(full_path, 'w') as f:
        f.writelines(header)
        f.write('\t'.join(data.columns) + '\n')
        for row in data.itertuples(index=False):
            f.write('\t'.join(map(str, row)) + '\n')

    return True


def get_all_folder_paths(directory: Path | str) -> set[str]:
    """
    Recursively collects all subfolder paths from a given directory.

    Parameters:
        directory (Path | str): Root directory to search.

    Returns:
        set[str]: Set of folder paths (as strings).
    """
    folder_paths = set()
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            folder_path = os.path.join(root, d)
            folder_paths.add(folder_path)
    return folder_paths


def filter_paths_by_subpaths(folder_paths: set[str], subpaths: list[str]) -> set[str]:
    """
    Filters paths to include only those that contain one of the given substrings.

    Parameters:
        folder_paths (set): Set of folder paths.
        subpaths (list): List of substrings to search for.

    Returns:
        set[str]: Filtered set of matching paths.
    """

    filtered_paths = set()
    for path in folder_paths:
        if any(sub in path for sub in subpaths):
            filtered_paths.add(path)
    return filtered_paths


def remove_paths_with_patterns(folder_paths: set[str], patterns: list[str]) -> set[str]:
    """
    Removes all paths that contain any of the specified patterns.

    Parameters:
        folder_paths (set): Set of folder paths.
        patterns (list): List of substrings; any path containing one will be excluded.

    Returns:
        set[str]: Filtered set of paths.
    """
    filtered_paths = {path for path in folder_paths if not any(pattern in path for pattern in patterns)}
    return filtered_paths


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
        imu_marker_4 = extract_marker_coordinates(marker_data, marker_names[3], i)

    if any(imu_marker_1 == 0.0):
        # Handle case where marker 1 is not available
        marker_x = imu_marker_4 - imu_marker_2
        marker_y = imu_marker_3 - imu_marker_2
        marker_z = np.cross(marker_x, marker_y)
        # as marker_x and marker_y are not orthogonal, we need to recalculate marker_x
        marker_x = np.cross(marker_y, marker_z)
    else:
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

