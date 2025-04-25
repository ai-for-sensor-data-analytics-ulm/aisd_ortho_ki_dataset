import shutil
from pathlib import Path
import opensim as osim
import helper_fcts as hf
from xml.etree import ElementTree as ET
import os


def change_default_pose(model_path, mot_file_path):
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


def modify_scaling_setup(xml_path, path_baseline_model, path_scaled_model, path_scaled_model_w_markers,
                         path_trc_file, path_output_mot_file, path_output_marker_file, time_range, no_ik_for,
                         no_scaling_for):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    roots = ["./ScaleTool/GenericModelMaker/model_file",
             "./ScaleTool/ModelScaler/output_model_file",
             "./ScaleTool/ModelScaler/marker_file",
             "./ScaleTool/MarkerPlacer/marker_file",
             "./ScaleTool/MarkerPlacer/output_model_file",
             "./ScaleTool/MarkerPlacer/output_motion_file",
             "./ScaleTool/MarkerPlacer/output_marker_file",
             "./ScaleTool/MarkerPlacer/time_range",
             "./ScaleTool/ModelScaler/time_range"]

    ik_roots = [f".//MarkerPlacer/IKTaskSet/objects/IKMarkerTask[@name='{marker}']/apply" for marker in no_ik_for]
    scaling_roots = [f".//ModelScaler/MeasurementSet/objects/Measurement[@name='{pos}']/apply" for pos in
                     no_scaling_for]

    for r, value in zip(roots, [path_baseline_model, path_scaled_model, path_trc_file, path_trc_file,
                                path_scaled_model_w_markers, path_output_mot_file, path_output_marker_file, time_range,
                                time_range]):
        b2tf = root.find(r)
        b2tf.text = str(value)
    for r in ik_roots + scaling_roots:
        b2tf = root.find(r)
        b2tf.text = 'false'
    tree.write(xml_path)
    return


def scale_model_with_markers(measurement_path, path_template_model, path_template_scaling_settings: Path, cfg: dict,
                             subject_name, exercise):
    start_ts = cfg['start_ts']
    ik_path = measurement_path / 'ik_imus'
    models_path = ik_path / 'models'
    scaled_model_name = f'scaled_model_{subject_name}_{exercise}.osim'
    scaled_model_initial_pose_name = f'scaled_model_initial_pose_{subject_name}_{exercise}.osim'
    pose_file_name = f'{subject_name}_scaled_initial_pose.mot'

    os.makedirs(ik_path / 'models', exist_ok=True)

    os.makedirs(ik_path / 'results_scaling', exist_ok=True)

    path_scaling_settings = hf.copy_xml_to_directory(path_template_scaling_settings,
                                                     Path(ik_path / f'scaling_settings_{subject_name}_{exercise}.xml'))

    modify_scaling_setup(
        xml_path=path_scaling_settings,
        path_baseline_model=path_template_model.resolve(),
        path_scaled_model=(models_path / scaled_model_name).resolve(),
        path_scaled_model_w_markers=f'{os.sep}models{os.sep}{scaled_model_initial_pose_name}',
        path_trc_file=f'{os.sep}marker_data_osim_format_{subject_name}_{exercise}.trc',
        path_output_mot_file=f'{os.sep}results_scaling{os.sep}{pose_file_name}',
        path_output_marker_file=f'{os.sep}results_scaling{os.sep}{subject_name}_scaled_markers.xml',
        time_range=f' {start_ts} {start_ts + 0.015}',
        no_ik_for=cfg['scaling']['no_ik_for'],
        no_scaling_for=cfg['scaling']['no_scaling_for']
    )

    scale_tool = osim.ScaleTool(str(path_scaling_settings))
    scale_tool.setPathToSubject(str(ik_path.resolve()))
    scale_tool.run()

    if not (models_path /'geometry').exists():
        shutil.copytree(path_template_model.parent / 'geometry',
                        models_path / 'geometry')

    hf.change_model_name(models_path /scaled_model_name,
                         f'{subject_name}_{exercise}_scaled')
    hf.change_model_name(models_path /scaled_model_initial_pose_name,
                         f'{subject_name}_{exercise}_scaled_initial_pose')
    change_default_pose(models_path / scaled_model_initial_pose_name,
                        ik_path / 'results_scaling' / pose_file_name)
    return

