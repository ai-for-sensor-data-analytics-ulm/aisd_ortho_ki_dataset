import pandas as pd
from pathlib import Path
import helper_fcts as hf
import os
import numpy as np
from tqdm import tqdm
import json
import shutil
import re


def imu_to_csv(raw_data_path, processed_data_path, imu_samplerate):
    raw_imu_data = np.load(raw_data_path, allow_pickle=True).item()
    col_names = []
    df_data = []
    lengths = set([len(data['QW']) for data in raw_imu_data.values()])
    if max(lengths) - min(lengths) > 4:
        if 'austra/ng/' not in str(processed_data_path):
            raise ValueError(f'IMU data lengths are not equal for IMUs in {raw_data_path}.')
    min_com_length = min(lengths)

    # special case, as the qualisys measurement for gregers is shorter than the xsens measurement
   # if 'gregers/ng' in str(processed_data_path):
    #    min_com_length = 21803

    for imu_pos in raw_imu_data.keys():
        for axis, d in raw_imu_data[imu_pos].items():
            if axis in ['QW', 'QX', 'QY', 'QZ']:
                col_names.append(f'{imu_pos}_{axis}')
                if len(d) > min_com_length:
                    d = d[:min_com_length]
                df_data.append(list(d))
    imu_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    imu_df = imu_df.iloc[XSENS_OPENSIM_TIME_OFFSET:, :]
    imu_df.index = imu_df.index - XSENS_OPENSIM_TIME_OFFSET
    imu_df['time [s]'] = imu_df.index / imu_samplerate
    imu_df = imu_df[['time [s]'] + col_names]
    imu_df.to_csv(processed_data_path, index=False)
    return

def find_constant_speed_segments(speeds, timestamps, min_duration=2000, tolerance=0.05):
    start_idx = 0
    segments = []

    for i in range(1, len(speeds)):
        # Prüfe, ob der aktuelle Wert außerhalb der erlaubten Toleranz liegt
        if np.abs(speeds[i] - speeds[start_idx:i+1].mean()) > tolerance:
            # Prüfe Dauer
            duration = timestamps[i - 1] - timestamps[start_idx]
            if duration >= min_duration:
                segments.append((timestamps[start_idx], timestamps[i - 1]))
            # Neuer Startpunkt
            start_idx = i

    # Prüfe letzten Abschnitt
    duration = timestamps[-1] - timestamps[start_idx]
    if duration >= min_duration:
        segments.append((timestamps[start_idx], timestamps[-1]))

    return segments


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

    rel_cols = []
    for col in marker_df.columns[marker_df.iloc[0] == 0.0].tolist():
        if col == 'time_[s]':
            continue
        else:
            rel_cols.append(re.sub(r'_[XYZ]_?\[mm\]', '', col))

    return list(set(rel_cols))


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

def timestamps_treadmill_csv(processed_data_path, velocity_data, samplerate, exercise):
    segments = find_constant_speed_segments(velocity_data, np.arange(len(velocity_data)), min_duration=1000,
                                            tolerance=0.05)
    velocities = []
    for start_ts, stop_ts in segments:
        velocities.append(round(velocity_data[start_ts + ((stop_ts - start_ts)//2)] * 3.6, 2))
    if len(segments) != 3:
        if 'austra/ng/' in str(processed_data_path):
            segments = [(20059, 22638), (27055, 33000), (34001, 40474)]
            velocities = [3.54, 4.02, 4.46 ]
        elif 'yaxkin/ng/' in str(processed_data_path):
            segments = [(767, 5402), (5604, 8095.9), (8244, 11116)]
            velocities = [2.57, 3.02, 3.96]
        elif 'yaxkin/gwo/' in str(processed_data_path):
            segments = segments[1:]
            velocities = velocities[1:]
        elif 'latifah/gwo/' in str(processed_data_path):
            segments = segments[1:]
            velocities = velocities[1:]
        elif 'gregers/ng/' in str(processed_data_path):
            segments = [segments[0]] + segments[2:]
            velocities = [velocities[0]] + velocities[2:]
        elif 'marquise/ng/' in str(processed_data_path):
            pass
        elif 'neves/ng/' in str(processed_data_path):
            segments = [(3237, 7997), (9124, 15031), (16053, 20971)]
            velocities = [3.49, 4.03, 4.48]
        elif 'erna/gwo' in str(processed_data_path):
            segments = segments[1:]
            velocities = velocities[1:]
        else:
            raise ValueError(f'Irregular nr of segments for {processed_data_path}')
    labels = []
    starts = []
    stops = []

    for (start_ts, stop_ts), label in zip(segments, [f'{exercise}_v{i}' for i in range(len(segments))]):
        starts.append(start_ts / samplerate)
        stops.append(stop_ts/ samplerate)
        labels.append(label)
    timestamps_df = pd.DataFrame(data=np.array([labels, starts, stops, velocities]).T, columns=['label', 'temporal_segment_start_[s]', 'temporal_segment_end_[s]', 'velocities_[km_h]'])
    timestamps_df.to_csv(processed_data_path, index=False)
    return


##### SETTINGS #########################################################################################################
IMU_SAMPLERATE = 100
IMU_DATA_FILE_NAME = 'xsens_imu_data'
MARKER_DATA_FILE_NAME = 'qualisys_marker_data'
# time offset between xsens and qualisys in timestamps, likely due to sync delay. Only positive values are allowed and represent the number of samples which are cut from the xsens data.
XSENS_OPENSIM_TIME_OFFSET = 3

LABEL_LOOKUP = {'ee_corr': 'rd_correct',
                'ee_toe': 'rd_toes',
                'ee_sup': 'rd_supination',
                'ee_pro': 'rd_pronation',
                'comp_ex_correct': 'rgs_correct',
                'comp_ex_foot_drops': 'rgs_flexion',
                'comp_ex_weight_shift_supporting_leg': 'rgs_abduction',
                'comp_ex_hip_static': 'rgs_stork',
                'ng_v0': 'ng_v0',
                'ng_v1': 'ng_v1',
                'ng_v2': 'ng_v2',
                'gwo_v0': 'gwo_v0',
                'gwo_v1': 'gwo_v1',
                'gwo_v2': 'gwo_v2',}

segment_to_marker_lookup = { 'toes_r': {'marker_names':['R_TOE1', 'R_TOE2', 'R_TOE3'],'imu_name': 'XSens_Hand_Right'},
                                                        'calcn_l': {'marker_names': ['L_FOOT1', 'L_FOOT2', 'L_FOOT3', 'L_FOOT4'],'imu_name': 'XSens_Foot_Left'},
                                                        'calcn_r': {'marker_names':['R_FOOT1',  'R_FOOT2', 'R_FOOT3', 'R_FOOT4'],'imu_name': 'XSens_Foot_Right'},
                                                        'tibia_r':{'marker_names':['R_SHIN1', 'R_SHIN2', 'R_SHIN3', 'R_SHIN4'],'imu_name': 'XSens_LowerLeg_Right'} ,
                                                        'tibia_l': {'marker_names':['L_SHIN1', 'L_SHIN2', 'L_SHIN3', 'L_SHIN4'],'imu_name': 'XSens_LowerLeg_Left'},
                                                        'femur_r': {'marker_names':['R_THI1', 'R_THI2', 'R_THI3', 'R_THI4'],'imu_name': 'XSens_UpperLeg_Right'},
                                                        'femur_l':{'marker_names':['L_THI1', 'L_THI2', 'L_THI3', 'L_THI4'],'imu_name': 'XSens_UpperLeg_Left'} ,
                                                        'pelvis': {'marker_names':['PELV1', 'PELV2', 'PELV3', 'PELV4'],'imu_name': 'XSens_Pelvis'},
                                                        'torso': {'marker_names':['THOR1', 'THOR2', 'THOR3', 'THOR4'],'imu_name': 'XSens_Sternum'}}

scaling_segment_to_marker_lookup = {'R_IAS': ['pelvis','femur_r', 'torso'],
                                    'L_IAS': ['pelvis', 'femur_l', 'torso'],
                                    'R_FLE': ['femur_r'],
                                    'L_FLE': ['femur_l'],
                                    'R_FAX': ['tibia_r'],
                                    'R_FAL': ['tibia_r'],
                                    'L_FAX': ['tibia_l'],
                                    'L_FAL': ['tibia_l'],
                                    'R_SAE': ['torso', 'humerus_r'],
                                    'L_SAE': ['torso', 'humerus_l'],
                                    'R_HLE': ['humerus_r', 'radius_r', 'ulna_r'],
                                    'L_HLE': ['humerus_l', 'radius_l', 'ulna_l'],
                                    'R_RSP': ['radius_r', 'ulna_r'],
                                    'L_RSP': ['radius_l', 'ulna_l']}


METADATAS = {
    'Latifah': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}
                },
    'Hamit': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Laurel':{'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': ['R_FME']},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Darryl': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Austra':{'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': ['torso'], 'no_ik_for':['R_SAE', 'L_SAE']}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': ['torso'], 'no_ik_for':['R_SAE', 'L_SAE']}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Jung-Hee': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Rehema': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Etsuko': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 10,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Gregers': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Rushda':{'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Marquise':{'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 5,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Julia':{'rd': {'start_ts': 10, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Katee': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Neves':{'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Erna': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 5, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 5, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Hans': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': { 'toes_r': {'marker_names':['R_TOE1', 'R_TOE2', 'R_TOE3'],'imu_name': 'XSens_Hand_Right'},
                                                        'calcn_l': {'marker_names': ['L_FOOT1', 'L_FOOT2', 'L_FOOT3', 'L_FOOT4'],'imu_name': 'XSens_Foot_Left'},
                                                        'calcn_r': {'marker_names':['R_FOOT1',  'R_FOOT2', 'R_FOOT3', 'R_FOOT4'],'imu_name': 'XSens_Foot_Right'},
                                                        'tibia_r':{'marker_names':['R_SHIN1', 'R_SHIN2', 'R_SHIN3', 'R_SHIN4'],'imu_name': 'XSens_Prop1'} ,
                                                        'tibia_l': {'marker_names':['L_SHIN1', 'L_SHIN2', 'L_SHIN3', 'L_SHIN4'],'imu_name': 'XSens_LowerLeg_Left'},
                                                        'femur_r': {'marker_names':['R_THI1', 'R_THI2', 'R_THI3', 'R_THI4'],'imu_name': 'XSens_UpperLeg_Right'},
                                                        'femur_l':{'marker_names':['L_THI1', 'L_THI2', 'L_THI3', 'L_THI4'],'imu_name': 'XSens_UpperLeg_Left'} ,
                                                        'pelvis': {'marker_names':['PELV1', 'PELV2', 'PELV3', 'PELV4'],'imu_name': 'XSens_Pelvis'},
                                                        'torso': {'marker_names':['THOR1', 'THOR2', 'THOR3', 'THOR4'],'imu_name': 'XSens_Sternum'}}}},
    'Ziri': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'R_FME']}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Yaxkin': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 5,  'scaling': {'no_scaling_for': [], 'no_ik_for':[]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}},
    'Elodie': {'rd': {'start_ts': 0, 'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'R_FME' ]}, 'inverse_kinematics': segment_to_marker_lookup},
                'rgs': {'start_ts': 0,  'scaling': {'no_scaling_for': ['pelvis'], 'no_ik_for':['R_IPS', 'L_IPS' ]}, 'inverse_kinematics': segment_to_marker_lookup},
                'ng': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                       'inverse_kinematics': segment_to_marker_lookup},
                'gwo': {'start_ts': 0, 'scaling': {'no_scaling_for': [], 'no_ik_for': []},
                        'inverse_kinematics': segment_to_marker_lookup}}}

##### CODE #########################################################################################################

all_paths = hf.get_all_folder_paths('../raw_data')
# all_paths = hf.filter_paths_by_subpaths(all_paths, ['Physio_Uebungen/Einfache_Uebung', 'Physio_Uebungen/Komplexe_Uebung'])
all_paths = hf.filter_paths_by_subpaths(all_paths, ['Gehen/Mit_Orthese', 'Gehen/Ohne_Einschraenkung', 'Physio_Uebungen/Einfache_Uebung', 'Physio_Uebungen/Komplexe_Uebung'])
#all_paths = hf.filter_paths_by_subpaths(all_paths, ['Gregers/Gehen/Ohne_Einschraenkung'])
# all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results', 'Latifah', 'Hamit', 'Laurel',
#                                                              'Darryl', 'Austra', 'Jung-Hee', 'Rehema', 'Etsuko', 'Yaxkin', 'Elodie', 'Gregers', 'Rushda',
#                                                              'Marquise', 'Julia', 'Katee', 'Neves']))
# all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results', Latifah', 'Hamit', 'Laurel',
#                                                              'Darryl', 'Austra', 'Jung-Hee', 'Rehema', 'Etsuko', 'Yaxkin', 'Elodie', 'Gregers', 'Rushda',
#                                                              'Marquise', 'Julia', 'Katee', 'Neves']))
all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results']))
# all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['Zebris', 'XSens', 'QTM', 'models', 'IMU_IK_results', 'Latifah', 'Austra', 'Darryl', 'Etsuko', 'Hamit', 'Jung-Hee', 'Laurel', 'Rehema', 'Yaxkin']))

save_path = Path('../data')

for p in tqdm(all_paths):
    if 'Einfache_Uebung' in str(p):
        exercise = 'rd'
    elif 'Komplexe_Uebung' in str(p):
        exercise = 'rgs'
    elif 'Mit_Orthese' in str(p):
        exercise = 'gwo'
    elif 'Ohne_Einschraenkung' in str(p):
        exercise = 'ng'

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
    marker_files = list((p/'QTM').glob("*Dynamic OKI*.npy"))
    if not marker_files:
        raise FileExistsError(f"Keine Marker-Datei in {p} gefunden.")
    elif len(marker_files) > 1:
        raise ValueError(f"Mehrere Marker-Dateien in {p} gefunden: {marker_files}.")
    else:
        raw_marker_path = marker_files[0]
    processed_marker_path = subject_save_path / f'{MARKER_DATA_FILE_NAME}_{subject_name.lower()}_{exercise}.csv'
    add_no_ik_for = marker_to_csv(raw_marker_path, processed_marker_path)
    add_no_scaling_for = []
    for marker in add_no_ik_for:
        if marker in segment_to_marker_lookup.keys():
            add_no_scaling_for += segment_to_marker_lookup[marker]['marker_names']
    add_no_scaling_for = list(set(add_no_scaling_for))

    # generate timestamps csv
    if exercise == 'rd' or exercise == 'rgs':
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
    elif exercise in ['ng', 'gwo']:
        velocity_data = np.load(p / 'Zebris' / 'zebris_velocity_profile.npy', allow_pickle=True)
        velocity_data = velocity_data[:, 0] + (velocity_data[:, 1] / (10 ** np.ceil(np.log10(velocity_data[:, 1] + 1))))
        timestamps_treadmill_csv(processed_data_path=subject_save_path / f'timestamps_{subject_name.lower()}_{exercise}.csv',
                                 velocity_data=velocity_data, samplerate=IMU_SAMPLERATE, exercise=exercise)

    else:
        raise ValueError('Unrecognized Exercise')

    # generate metadata.json
    metadata = METADATAS[subject_name][exercise]
    metadata['scaling']['no_ik_for'] = list(set(metadata['scaling']['no_ik_for'] + add_no_ik_for))
    metadata['scaling']['no_scaling_for'] = list(set(metadata['scaling']['no_scaling_for'] + add_no_scaling_for))
    with open(subject_save_path / f'metadata_{subject_name.lower()}_{exercise}.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    # cp .trc marker file
    trc_files = list((p / 'QTM').glob("*opensim_data_Dynamic*.trc"))
    if not trc_files:
        raise FileExistsError(f"Keine trc_files-Datei in {p} gefunden.")
    elif len(trc_files) > 1:
        raise ValueError(f"Mehrere trc_files-Dateien in {p} gefunden: {timestamp_files}.")
    else:
        trc_file_path = trc_files[0]
    shutil.copy2(trc_file_path, subject_save_path / 'ik_imus' / f'marker_data_osim_format_{subject_name.lower()}_{exercise}.trc')

