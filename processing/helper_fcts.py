import shutil
import os
import opensim as osim
from pathlib import Path
import pandas as pd


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
