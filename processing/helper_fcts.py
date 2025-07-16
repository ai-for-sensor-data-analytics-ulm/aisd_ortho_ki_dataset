"""Utility helpers for the processing pipeline."""

import logging
import os
import shutil
from pathlib import Path

import numpy as np
import opensim as osim
import pandas as pd
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger(__name__)

IMU_QUAT_AXES = ['QX', 'QY', 'QZ', 'QW']
IMU_TO_OPENSIM_ROTATION = R.from_euler(seq='x', angles=-90, degrees=True)


def extract_subject_name_from_path(p: str | Path) -> str:
    """Return the subject name encoded in a path.

    Parameters
    ----------
    p : str or Path
        Path containing the subject folder as the third segment.

    Returns
    -------
    str
        The extracted subject name.

    Raises
    ------
    ValueError
        If the path does not contain enough segments.
    """
    parts = Path(p).parts
    if len(parts) < 3:
        raise ValueError(f"Path {p} does not contain enough segments.")
    return parts[2]


def extract_exercise_from_path(p: str | Path) -> str:
    """Return the exercise name encoded in a path.

    Parameters
    ----------
    p : str or Path
        Path pointing to the exercise directory.

    Returns
    -------
    str
        The exercise name.
    """
    return Path(p).name


def copy_xml_to_directory(xml_path: Path, new_path: Path, delete: bool = False) -> Path:
    """Copy an XML file to ``new_path``.

    Parameters
    ----------
    xml_path : Path
        Source XML file.
    new_path : Path
        Destination including the new file name.
    delete : bool, optional
        If ``True`` delete the original file after copying.

    Returns
    -------
    Path
        Path to the copied file.
    """
    new_path = shutil.copy2(xml_path, new_path)
    if delete:
        os.remove(xml_path)
    return Path(new_path)


def change_model_name(model_path: Path, new_model_name: str) -> None:
    """Change the name of an OpenSim model file.

    Parameters
    ----------
    model_path : Path
        Path to the ``.osim`` file.
    new_model_name : str
        New model name.
    """
    model = osim.Model(str(model_path))
    model.setName(new_model_name)
    model.printToXML(str(model_path))
    return


def write_to_mot_file(
    data: pd.DataFrame,
    header: list[str],
    filepath: Path,
    filename: str,
) -> bool:
    """Write data to a ``.mot`` file.

    Parameters
    ----------
    data : pandas.DataFrame
        Motion data with time and channel columns.
    header : list[str]
        Header lines to prepend to the file.
    filepath : Path
        Directory for the output file.
    filename : str
        Name of the resulting file.

    Returns
    -------
    bool
        ``True`` on success.
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
    """Recursively collect all subfolder paths.

    Parameters
    ----------
    directory : Path or str
        Root directory to search.

    Returns
    -------
    set[str]
        Set of folder paths as strings.
    """
    folder_paths = set()
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            folder_path = os.path.join(root, d)
            folder_paths.add(folder_path)
    return folder_paths


def filter_paths_by_subpaths(folder_paths: set[str], subpaths: list[str]) -> set[str]:
    """Return only paths that contain any of ``subpaths``.

    Parameters
    ----------
    folder_paths : set[str]
        Paths to filter.
    subpaths : list[str]
        Substrings to search for.

    Returns
    -------
    set[str]
        Filtered set of matching paths.
    """

    filtered_paths = set()
    for path in folder_paths:
        if any(sub in path for sub in subpaths):
            filtered_paths.add(path)
    return filtered_paths


def remove_paths_with_patterns(folder_paths: set[str], patterns: list[str]) -> set[str]:
    """Remove all paths that contain any of the given patterns.

    Parameters
    ----------
    folder_paths : set[str]
        Collection of paths.
    patterns : list[str]
        Substrings that cause a path to be dropped if present.

    Returns
    -------
    set[str]
        Filtered set of paths.
    """
    filtered_paths = {path for path in folder_paths if not any(pattern in path for pattern in patterns)}
    return filtered_paths


def read_trc_file(filename: str) -> pd.DataFrame:
    """Read a Qualisys ``.trc`` file.

    Parameters
    ----------
    filename : str
        Path to the ``.trc`` file.

    Returns
    -------
    pandas.DataFrame
        Marker positions with columns ``['Time', '<marker>_x', '<marker>_y', '<marker>_z']``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the data lines do not match the expected column format.
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
    """Return the coordinates of ``marker_name`` at ``pos``.

    Parameters
    ----------
    marker_data : pandas.DataFrame
        DataFrame containing marker columns.
    marker_name : str
        Name of the marker without axis suffix.
    pos : int
        Row index to extract.

    Returns
    -------
    numpy.ndarray
        Array ``[x, y, z]`` of marker coordinates.
    """
    return marker_data[[f'{marker_name}_{axis}' for axis in ['x', 'y', 'z']]].iloc[pos].to_numpy()


def calculate_heading_correction(marker_data: pd.DataFrame, marker_names: list[str], R_imu: R, pos: str) -> R:
    """Compute the heading correction for an IMU.

    Parameters
    ----------
    marker_data : pandas.DataFrame
        Marker positions for a static pose.
    marker_names : list[str]
        Names of the markers defining the local frame.
    R_imu : Rotation
        Orientation of the IMU.
    pos : str
        Segment name. ``"toes_r"`` triggers a special case.

    Returns
    -------
    Rotation
        Heading correction as a rotation about the y-axis.
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

