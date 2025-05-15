import shutil
from pathlib import Path
import opensim as osim
import helper_fcts as hf
from xml.etree import ElementTree as ET
import os
import logging
import sys
import contextlib

logger = logging.getLogger(__name__)
# sys.stderr.write = logger.error
# sys.stdout.write = logger.info
# osim.Logger.setLevel(osim.Logger.Level_Info)


def change_default_pose(model_path: Path, mot_file_path: Path):
    """
    Sets the default pose in an OpenSim model based on the first frame of a .mot motion file.

    Parameters:
        model_path (Path): Path to the .osim model to be modified.
        mot_file_path (Path): Path to the motion (.mot) file with joint angles.
    """
    model = osim.Model(str(model_path))
    motion = osim.Storage(str(mot_file_path))
    state = model.initSystem()
    model.updWorkingState()

    column_labels = motion.getColumnLabels()
    for i in range(1, column_labels.getSize()):
        coordinate_name = column_labels.get(i)
        if 'speed' in coordinate_name:
            continue
        coordinate_name = coordinate_name.split('/')[3]
        value = motion.getStateVector(0).getData().get(i - 1)
        coordinate = model.getCoordinateSet().get(coordinate_name)
        coordinate.set_default_value(value)
    model.realizePosition(state)
    model.printToXML(str(model_path))
    return


def modify_scaling_setup(xml_path: Path,
                         path_baseline_model: Path,
                         path_scaled_model: Path,
                         path_scaled_model_w_markers: str,
                         path_trc_file: str,
                         path_output_mot_file: str,
                         path_output_marker_file: str,
                         time_range: str,
                         no_ik_for: list[str],
                         no_scaling_for: list[str]):
    """
    Modifies an OpenSim scaling XML setup file by setting input/output paths, time ranges,
    and disabling specified IK/scaling markers.

    Parameters:
        xml_path (Path): Path to the XML file to modify.
        path_baseline_model (Path): Path to the unscaled model.
        path_scaled_model (Path): Output path for scaled model.
        path_scaled_model_w_markers (str): Path (as string) for model with markers.
        path_trc_file (str): Relative path to the input TRC file.
        path_output_mot_file (str): Path to output motion file.
        path_output_marker_file (str): Path to output marker error file.
        time_range (str): Time range for scaling and marker placement (e.g. "0.0 0.02").
        no_ik_for (list[str]): List of marker names to disable in IK task set.
        no_scaling_for (list[str]): List of markers to disable in scaling.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    path_map = {
        "./ScaleTool/GenericModelMaker/model_file": path_baseline_model,
        "./ScaleTool/ModelScaler/output_model_file": path_scaled_model,
        "./ScaleTool/ModelScaler/marker_file": path_trc_file,
        "./ScaleTool/MarkerPlacer/marker_file": path_trc_file,
        "./ScaleTool/MarkerPlacer/output_model_file": path_scaled_model_w_markers,
        "./ScaleTool/MarkerPlacer/output_motion_file": path_output_mot_file,
        "./ScaleTool/MarkerPlacer/output_marker_file": path_output_marker_file,
        "./ScaleTool/MarkerPlacer/time_range": time_range,
        "./ScaleTool/ModelScaler/time_range": time_range,
    }

    for xpath, value in path_map.items():
        elem = root.find(xpath)
        if elem is None:
            logger.error(f"Missing XML entry: {xpath}")
            raise ValueError(f"Missing XML entry: {xpath}")
        elem.text = str(value)

    for marker in no_ik_for:
        ik_task = root.find(f".//MarkerPlacer/IKTaskSet/objects/IKMarkerTask[@name='{marker}']/apply")
        if ik_task is not None:
            ik_task.text = 'false'

    for name in no_scaling_for:
        scaling = root.find(f".//ModelScaler/MeasurementSet/objects/Measurement[@name='{name}']/apply")
        if scaling is not None:
            scaling.text = 'false'

    tree.write(xml_path)
    return


def scale_model_with_markers(measurement_path: Path,
                             path_template_model: Path,
                             path_template_scaling_settings: Path,
                             cfg: dict,
                             subject_name: str,
                             exercise: str):
    """
    Performs model scaling and static pose estimation using marker data and scaling settings.

    Parameters:
        measurement_path (Path): Path to the subject's measurement folder.
        path_template_model (Path): Path to the unscaled .osim template model.
        path_template_scaling_settings (Path): Path to the XML template for scaling.
        cfg (dict): Configuration dict containing 'start_ts' and scaling marker options.
        subject_name (str): Subject ID.
        exercise (str): Exercise name (used in filenames).
    """
    start_ts = cfg['start_ts']
    ik_path = measurement_path / 'ik_imus'
    models_path = ik_path / 'models'

    scaled_model_name = f'scaled_model_{subject_name}_{exercise}.osim'
    scaled_model_initial_pose_name = f'scaled_model_initial_pose_{subject_name}_{exercise}.osim'
    pose_file_name = f'scaled_initial_pose_{subject_name}_{exercise}.mot'

    os.makedirs(ik_path / 'models', exist_ok=True)

    os.makedirs(ik_path / 'results_scaling', exist_ok=True)

    path_scaling_settings = hf.copy_xml_to_directory(path_template_scaling_settings,
                                                     Path(ik_path / f'scaling_settings_{subject_name}_{exercise}.xml'))

    # adjust no_ik_for


    modify_scaling_setup(
        xml_path=path_scaling_settings,
        path_baseline_model=path_template_model.resolve(),
        path_scaled_model=(models_path / scaled_model_name).resolve(),
        path_scaled_model_w_markers=f'{os.sep}models{os.sep}{scaled_model_initial_pose_name}',
        path_trc_file=f'{os.sep}marker_data_osim_format_{subject_name}_{exercise}.trc',
        path_output_mot_file=f'{os.sep}results_scaling{os.sep}{pose_file_name}',
        path_output_marker_file=f'{os.sep}results_scaling{os.sep}scaled_markers_{subject_name}_{exercise}.xml',
        time_range=f' {start_ts} {start_ts + 0.015}',
        no_ik_for=cfg['scaling']['no_ik_for'],
        no_scaling_for=cfg['scaling']['no_scaling_for']
    )

    scale_tool = osim.ScaleTool(str(path_scaling_settings))
    scale_tool.setPathToSubject(str(ik_path.resolve()))
    scale_tool.run()
    logger.info(f"ScaleTool run completed for {subject_name} / {exercise}")

    # Ensure Geometry folder exists
    if not (models_path / 'Geometry').exists():
        shutil.copytree(path_template_model.parent / 'Geometry',
                        models_path / 'Geometry')

    # Rename model names for clarity
    hf.change_model_name(models_path / scaled_model_name,
                         f'{subject_name}_{exercise}_scaled')
    hf.change_model_name(models_path / scaled_model_initial_pose_name,
                         f'{subject_name}_{exercise}_scaled_initial_pose')

    # Set initial pose
    change_default_pose(models_path / scaled_model_initial_pose_name,
                        ik_path / 'results_scaling' / pose_file_name)
    return
