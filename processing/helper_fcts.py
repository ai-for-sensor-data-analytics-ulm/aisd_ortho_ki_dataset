import shutil
import os
import opensim as osim


def extract_subject_name_from_path(p):
    return p.split('/')[2]

def extract_exercise_from_path(p):
    return p.split('/')[-1]


def copy_xml_to_directory(xml_path, new_directory, delete=False):
    new_dir_path = shutil.copy2(xml_path, new_directory)
    if delete:
        os.remove(xml_path)
    return new_dir_path


def change_model_name(model_path, new_model_name):
    model = osim.Model(str(model_path))
    model.setName(new_model_name)
    model.printToXML(str(model_path))
    return


def write_to_mot_file(data, header, filepath, filename):
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


def get_all_folder_paths(directory):
    folder_paths = set()  # Using a set removes duplicates automatically.
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            folder_path = os.path.join(root, d)
            folder_paths.add(folder_path)
    return folder_paths


def filter_paths_by_subpaths(folder_paths, subpaths):
    """
    Filters the folder paths, returning only those that contain one of the specified subpath strings.
    The matching is done using a substring search.

    Parameters:
        folder_paths (iterable): The folder paths to filter.
        subpaths (list): A list of sub-p strings to search for in each folder p.

    Returns:
        set: A set of filtered folder paths.
    """
    filtered_paths = set()
    for path in folder_paths:
        if any(sub in path for sub in subpaths):
            filtered_paths.add(path)
    return filtered_paths


def remove_paths_with_patterns(folder_paths, patterns):
    """
    Entfernt aus den übergebenen Pfaden alle, bei denen mindestens eines der angegebenen Muster vorkommt.

    Parameter:
        folder_paths (iterable): Eine Sammlung von Pfaden (z.B. Liste oder Set).
        patterns (list): Eine Liste von Mustern (Strings), bei denen, falls sie im Pfad vorkommen,
                         der Pfad entfernt werden soll.

    Returns:
        set: Ein Set von Pfaden, in denen keines der Muster vorkommt.
    """
    filtered_paths = {path for path in folder_paths if not any(pattern in path for pattern in patterns)}
    return filtered_paths