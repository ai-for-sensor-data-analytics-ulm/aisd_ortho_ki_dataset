import yaml
from pathlib import Path
from tqdm import tqdm
import helper_fcts as hf
import scaling
import imu_inverse_kinematics
import json

##### SETTINGS #########################################################################################################
PATH_TEMPLATE_MODEL = Path('../opensim_templates/model_template/Unrestricted_Full_Body_Model.osim')
PATH_TEMPLATE_SCALING_SETTINGS = Path('../opensim_templates/template_scaling_settings.xml')
IMU_SAMPLERATE = 100

##### CODE ##
all_paths = hf.get_all_folder_paths('../data')
all_paths = hf.filter_paths_by_subpaths(all_paths, [
    'ce', 'ee'])
# all_paths = hf.filter_paths_by_subpaths(all_paths, [
#     'austra/ce'])
all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['ik_imus']))


for p in tqdm(all_paths):
    subject_name = hf.extract_subject_name_from_path(p)
    exercise = hf.extract_exercise_from_path(p)
    with open(Path(p) / f'metadata_{subject_name}_{exercise}.json', 'r') as stream:
        processing_config = json.load(stream)
    scaling.scale_model_with_markers(measurement_path=Path(p),
                                     path_template_model=PATH_TEMPLATE_MODEL,
                                     path_template_scaling_settings=PATH_TEMPLATE_SCALING_SETTINGS,
                                     cfg=processing_config,
                                     subject_name=subject_name,
                                     exercise=exercise)
    # imu_inverse_kinematics.perform_inverse_kinematics_w_imu_data(measurement_path=Path(p),
    #                                                              cfg=processing_config,
    #                                                              imu_sample_rate=IMU_SAMPLERATE,
    #                                                              subject_name=subject_name,
    #                                                             exercise=exercise)

