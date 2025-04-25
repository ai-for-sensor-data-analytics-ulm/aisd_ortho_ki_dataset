import pandas as pd
from pathlib import Path
import helper_fcts as hf
import os
import numpy as np
from tqdm import tqdm
import json
import shutil


def imu_to_csv(raw_data_path, processed_data_path, imu_samplerate):
    raw_imu_data = np.load(raw_data_path, allow_pickle=True).item()
    col_names = []
    df_data = []
    lengths = set([len(data['QW']) for data in raw_imu_data.values()])
    if max(lengths) - min(lengths) > 4:
        raise ValueError(f'IMU data lengths are not equal for IMUs in {raw_data_path}.')
    min_com_length = min(lengths)
    for imu_pos in raw_imu_data.keys():
        for axis, d in raw_imu_data[imu_pos].items():
            if axis in ['QW', 'QX', 'QY', 'QZ']:
                col_names.append(f'{imu_pos}_{axis}')
                if len(d) > min_com_length:
                    d = d[:min_com_length]
                df_data.append(list(d))
    imu_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    imu_df['time [s]'] = imu_df.index / imu_samplerate
    imu_df = imu_df[['time [s]'] + col_names]
    imu_df.to_csv(processed_data_path, index=False)
    return


def marker_to_csv(raw_data_path, processed_data_path):
    raw_marker_data = np.load(raw_data_path, allow_pickle=True).item()
    col_names = []
    df_data = []
    lengths = set([len(data) for data in raw_marker_data['Marker'].values()] + [len(raw_marker_data['Time'])])
    if max(lengths) - min(lengths) > 4:
        raise ValueError(f'Marker data lengths are not equal for Markers in {raw_data_path}.')
    min_com_length = min(lengths)
    col_names.append('time_[s]')
    df_data.append(raw_marker_data['Time'][:min_com_length])
    for marker_name, marker_data in raw_marker_data['Marker'].items():
        marker_data = np.array(marker_data)
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            col_names.append(f'{marker_name}_{axis_name}_[mm]')
            d = list(marker_data[:, i])
            if len(d) > min_com_length:
                d = d[:min_com_length]
            df_data.append(d)
    marker_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    marker_df.to_csv(processed_data_path, index=False)
    return


def timestamps_to_csv(raw_data_path, processed_data_path, label_lookup):
    raw_timestamp_data = pd.read_csv(raw_data_path, header=1)
    col_names = []
    df_data = []
    col_names.append('label')
    label_list = []
    for index, row in raw_timestamp_data.iterrows():
        metadata_value = str(row['metadata'])
        found_match = False

        for key, value in label_lookup.items():
            if key in metadata_value:
                label_list.append(value)
                found_match = True
                break

        if not found_match:
            raise ValueError(f"Label '{metadata_value}' not found in label lookup for timestamps in file {raw_data_path}.")
    df_data.append(label_list)
    col_names.append('temporal_segment_start_[s]')
    df_data.append(raw_timestamp_data['temporal_segment_start'])
    col_names.append('temporal_segment_end_[s]')
    df_data.append(raw_timestamp_data['temporal_segment_end'])
    timestamp_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    timestamp_df.to_csv(processed_data_path, index=False)
    return


##### SETTINGS #########################################################################################################
IMU_SAMPLERATE = 100
IMU_DATA_FILE_NAME = 'xsens_imu_data'
MARKER_DATA_FILE_NAME = 'qualisys_marker_data'

LABEL_LOOKUP = {'ee_corr': 'ee_no_abnormality',
                'ee_toe': 'ee_toe_lifted',
                'ee_sup': 'ee_supination',
                'ee_pro': 'ee_pronation',
                'comp_ex_correct': 'ce_no_abnormality',
                'comp_ex_foot_drops': 'ce_foot_drops',
                'comp_ex_weight_shift_supporting_leg': 'ce_weight_shift_supporting_leg',
                'comp_ex_hip_static': 'ce_hip_static'}

segment_to_marker_lookup = { 'toes_r': {'marker_names':['R_TOE1', 'R_TOE2', 'R_TOE3'],'imu_name': 'XSens_Hand_Right'},
                                                        'calcn_l': {'marker_names': ['L_FOOT1', 'L_FOOT2', 'L_FOOT3'],'imu_name': 'XSens_Foot_Left'},
                                                        'calcn_r': {'marker_names':['R_FOOT1',  'R_FOOT2', 'R_FOOT3'],'imu_name': 'XSens_Foot_Right'},
                                                        'tibia_r':{'marker_names':['R_SHIN1', 'R_SHIN2', 'R_SHIN3'],'imu_name': 'XSens_LowerLeg_Right'} ,
                                                        'tibia_l': {'marker_names':['L_SHIN1', 'L_SHIN2', 'L_SHIN3'],'imu_name': 'XSens_LowerLeg_Left'},
                                                        'femur_r': {'marker_names':['R_THI1', 'R_THI2', 'R_THI3'],'imu_name': 'XSens_UpperLeg_Right'},
                                                        'femur_l':{'marker_names':['L_THI1', 'L_THI2', 'L_THI3'],'imu_name': 'XSens_UpperLeg_Left'} ,
                                                        'pelvis': {'marker_names':['PELV1', 'PELV2', 'PELV3'],'imu_name': 'XSens_Pelvis'},
                                                        'torso': {'marker_names':['THOR1', 'THOR2', 'THOR3'],'imu_name': 'XSens_Sternum'}}

METADATAS = {
    'Latifah': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Hamit': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Laurel':{'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Darryl': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Austra':{'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': ['torso'], 'no_ik_for':['R_SAE', 'L_SAE']}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': ['torso'], 'no_ik_for':['R_SAE', 'L_SAE']}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Jung-Hee': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Rehema': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Etsuko': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 10,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Gregers': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Rushda':{'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Marquise':{'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 5,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Julia':{'ee': {'start_ts': 10, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Katee': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Neves':{'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Erna': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Hans': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': { 'toes_r': {'marker_names':['R_TOE1', 'R_TOE2', 'R_TOE3'],'imu_name': 'XSens_Hand_Right'},
                                                        'calcn_l': {'marker_names': ['L_FOOT1', 'L_FOOT2', 'L_FOOT3'],'imu_name': 'XSens_Foot_Left'},
                                                        'calcn_r': {'marker_names':['R_FOOT1',  'R_FOOT2', 'R_FOOT3'],'imu_name': 'XSens_Foot_Right'},
                                                        'tibia_r':{'marker_names':['R_SHIN1', 'R_SHIN2', 'R_SHIN3'],'imu_name': 'XSens_Prop1'} ,
                                                        'tibia_l': {'marker_names':['L_SHIN1', 'L_SHIN2', 'L_SHIN3'],'imu_name': 'XSens_LowerLeg_Left'},
                                                        'femur_r': {'marker_names':['R_THI1', 'R_THI2', 'R_THI3'],'imu_name': 'XSens_UpperLeg_Right'},
                                                        'femur_l':{'marker_names':['L_THI1', 'L_THI2', 'L_THI3'],'imu_name': 'XSens_UpperLeg_Left'} ,
                                                        'pelvis': {'marker_names':['PELV1', 'PELV2', 'PELV3'],'imu_name': 'XSens_Pelvis'},
                                                        'torso': {'marker_names':['THOR1', 'THOR2', 'THOR3'],'imu_name': 'XSens_Sternum'}}}},
    'Ziri': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'R_FME']}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Yaxkin': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 5,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup}},
    'Elodie': {'ee': {'start_ts': 0, 'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'R_FME' ]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ce': {'start_ts': 0,  'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'L_IPS' ]}, 'inverse_kinematics': segment_to_marker_lookup}}}

##### CODE #########################################################################################################

all_paths = hf.get_all_folder_paths('../raw_data')
all_paths = hf.filter_paths_by_subpaths(all_paths, ['Physio_Uebungen/Einfache_Uebung', 'Physio_Uebungen/Komplexe_Uebung'])
all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results']))
# all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results', 'Latifah', 'Austra', 'Darryl', 'Etsuko', 'Hamit', 'Jung-Hee', 'Laurel', 'Rehema', 'Yaxkin']))

save_path = Path('../data')

for p in tqdm(all_paths):
    if 'Einfache_Uebung' in str(p):
        exercise = 'ee'
    elif 'Komplexe_Uebung' in str(p):
        exercise = 'ce'
    elif 'Mit_Orthese' in str(p):
        exercise = 'go'
    elif 'Ohne_Einschraenkung' in str(p):
        exercise = 'g'

    subject_name = p.split('/')[2].split('_')[-1]
    p = Path(p)
    subject_save_path = save_path / subject_name.lower() / exercise

    # create folder structure
    os.makedirs(subject_save_path / 'ik_imus', exist_ok=True)

    # generate imu csv
    raw_imu_path = p / 'XSens' / 'xsens_data.npy'
    processed_imu_path = subject_save_path / f'{IMU_DATA_FILE_NAME}_{subject_name.lower()}_{exercise}.csv'
    imu_to_csv(raw_imu_path, processed_imu_path, IMU_SAMPLERATE)

    # generate marker csv
    raw_marker_path = p / 'QTM' / 'Dynamic OKI 1.npy'
    processed_marker_path = subject_save_path / f'{MARKER_DATA_FILE_NAME}_{subject_name.lower()}_{exercise}.csv'
    marker_to_csv(raw_marker_path, processed_marker_path)

    # generate timestamps csv
    if exercise == 'ee' or exercise == 'ce':
        timestamp_files = list(p.glob("*timestamp*.csv"))
        if not timestamp_files:
            raise FileExistsError(f"Keine Timestamp-CSV-Datei in {p} gefunden.")
        elif len(timestamp_files) > 1:
            raise ValueError(f"Mehrere Timestamp-CSV-Dateien in {p} gefunden: {timestamp_files}.")
        else:
            timestamp_file_path = timestamp_files[0]
        raw_timestamp_path = timestamp_file_path
        processed_timestamp_path = subject_save_path / f'timestamps_{subject_name.lower()}_{exercise}.csv'
        timestamps_to_csv(raw_timestamp_path, processed_timestamp_path, LABEL_LOOKUP)

    # generate metadata.json
    metadata = METADATAS[subject_name][exercise]
    with open(subject_save_path / f'metadata_{subject_name.lower()}_{exercise}.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    # cp .trc marker file
    shutil.copy2(p / 'QTM' / 'opensim_data_Dynamic OKI 1.trc', subject_save_path / 'ik_imus' / f'marker_data_osim_format_{subject_name.lower()}_{exercise}.trc')

