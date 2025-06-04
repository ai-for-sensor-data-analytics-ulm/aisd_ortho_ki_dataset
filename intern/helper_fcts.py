import shutil
import os
from pathlib import Path
from typing import Iterable, List, Set
import pandas as pd


def copy_xml_to_directory(xml_path: Path, new_directory: Path, delete: bool = False) -> Path:
    """Copy an XML file to a new directory.

    Parameters
    ----------
    xml_path : Path
        Path to the source XML file.
    new_directory : Path
        Destination directory.
    delete : bool, optional
        If ``True`` delete the source after copying.

    Returns
    -------
    Path
        Path to the copied XML file.
    """
    shutil.copy2(xml_path, new_directory)
    if delete:
        os.remove(xml_path)
    return new_directory


def write_to_mot_file(
    data: pd.DataFrame,
    header: List[str],
    filepath: Path,
    filename: str,
) -> bool:
    """Write motion data to a ``.mot`` file.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to write.
    header : list[str]
        Lines to prepend as file header.
    filepath : Path
        Directory for the output file.
    filename : str
        Name of the output file.

    Returns
    -------
    bool
        ``True`` if the file was written successfully.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    f = open(filepath / filename, 'w')
    f.writelines(header)
    list(data.columns)

    col_names = ''
    for col in list(data.columns):
        col_names += col + '\t'
    col_names += '\n'
    f.write(col_names)

    remaining_data = data.to_numpy()
    for i in range(remaining_data.shape[0]):
        next_row = remaining_data[i, :]
        row_str = ''
        for e in next_row:
            row_str += str(e) + '\t'
        row_str += '\n'
        f.write(row_str)
    f.close()
    return True


def get_all_folder_paths(directory: Path | str) -> Set[str]:
    """Recursively collect all subfolder paths of ``directory``.

    Parameters
    ----------
    directory : Path or str
        Root directory.

    Returns
    -------
    set[str]
        Set of folder paths.
    """
    folder_paths: Set[str] = set()
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            folder_path = os.path.join(root, d)
            folder_paths.add(folder_path)
    return folder_paths


def filter_paths_by_subpaths(folder_paths: Iterable[str], subpaths: List[str]) -> Set[str]:
    """Return only paths containing one of the given substrings.

    Parameters
    ----------
    folder_paths : Iterable[str]
        Paths to filter.
    subpaths : list[str]
        Substrings to search for.

    Returns
    -------
    set[str]
        Filtered folder paths.
    """
    filtered_paths: Set[str] = set()
    for path in folder_paths:
        if any(sub in path for sub in subpaths):
            filtered_paths.add(path)
    return filtered_paths


def remove_paths_with_patterns(folder_paths: Iterable[str], patterns: List[str]) -> Set[str]:
    """Remove all paths that contain any of the given patterns.

    Parameters
    ----------
    folder_paths : Iterable[str]
        Collection of folder paths.
    patterns : list[str]
        Substrings that, if found in a path, cause its removal.

    Returns
    -------
    set[str]
        Filtered paths without the specified patterns.
    """
    filtered_paths: Set[str] = {path for path in folder_paths if not any(pattern in path for pattern in patterns)}
    return filtered_paths