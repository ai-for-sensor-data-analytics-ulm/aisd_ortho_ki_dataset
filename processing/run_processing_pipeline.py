import argparse
from pathlib import Path
import helper_fcts as hf
import logging
from imu_inverse_kinematics import perform_inverse_kinematics_w_imu_data
from scaling import scale_model_with_markers
import json
import os
import yaml
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_subject(measurement_dir: Path, cfg: dict) -> None:
    """Run the processing pipeline for a single measurement.

    Parameters
    ----------
    measurement_dir : Path
        Directory containing the measurement data.
    cfg : dict
        Global configuration dictionary.
    """
    subject_name = hf.extract_subject_name_from_path(measurement_dir)
    exercise = hf.extract_exercise_from_path(measurement_dir)

    with open(Path(measurement_dir) / f'metadata_{subject_name}_{exercise}.json', 'r') as stream:
        processing_config = json.load(stream)

    try:
        scale_model_with_markers(measurement_path=measurement_dir,
                                 path_template_model=Path(cfg['path_template_model']),
                                 path_template_scaling_settings=Path(cfg['path_template_scaling_settings']),
                                 cfg=processing_config,
                                 subject_name=subject_name,
                                 exercise=exercise)

        perform_inverse_kinematics_w_imu_data(measurement_path=measurement_dir,
                                              cfg=processing_config,
                                              imu_sample_rate=cfg['imu_samplerate'],
                                              subject_name=subject_name,
                                              exercise=exercise)

        logging.info(f"{subject_name}/{exercise} processed")
    except Exception as e:
        logging.error(f"Error processing {subject_name}/{exercise}: {e}")


def run_batch_pipeline(root_path: Path, config_path: Path) -> None:
    """Process all subject/exercise combinations in a directory.

    Parameters
    ----------
    root_path : Path
        Root directory with measurement folders.
    config_path : Path
        Path to the YAML configuration file.
    """
    cfg = load_config(config_path)

    all_paths = hf.get_all_folder_paths(root_path)

    subpath_filter = []
    match cfg['process_exercises']:
        case ['all']:
            subpath_filter = ['gwo', f'{os.sep}ng', 'rd', 'rgs']
        case _:
            for subpath in cfg['process_exercises']:
                if subpath not in ['gwo', 'ng', 'rd', 'rgs']:
                    raise ValueError(f"Invalid exercise type: {subpath}")
                else:
                    if subpath in ['gwo', 'rd', 'rgs']:
                        subpath_filter.append(subpath)
                    else:
                        subpath_filter.append(f'{os.sep}{subpath}')
    all_paths = hf.filter_paths_by_subpaths(all_paths, subpath_filter)

    match cfg['process_subjects']:
        case ['all']:
            pass
        case _:
            all_paths = hf.filter_paths_by_subpaths(all_paths, cfg['process_subjects'])

    all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['ik_imus']))
    print('Starting batch processing ...')
    for measurement_dir in tqdm(all_paths):
        measurement_dir = Path(measurement_dir)
        if not measurement_dir.is_dir():
            continue
        logging.info(f"Processing {measurement_dir} ...")
        process_subject(measurement_dir, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch processing pipeline.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing subject/exercise folders")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--log", type=Path, default=None, help="Optional log file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        filename=str(args.log) if args.log else None,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    run_batch_pipeline(args.root, args.config)
