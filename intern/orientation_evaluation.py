import opensim as osim
from math import pi
from pathlib import Path
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import helper_fcts as hf
import pickle as pkl
import json
import yaml
from tqdm import tqdm

def read_trc_file(filename):

        if not os.path.exists(filename):
            print('file does not exist')

        # read all lines
        file_id = open(filename, 'r')
        all_lines = file_id.readlines()
        file_id.close()

        marker_names = all_lines[3].split('\t')
        marker_names.remove('Frame#')
        marker_names.remove('Time')

        if '\n' in marker_names:
            marker_names.remove('\n')

        marker_names = [x for x in marker_names if x != '']
        col_names = ['Time']
        for m in marker_names:
            col_names.append(f'{m}_x')
            col_names.append(f'{m}_y')
            col_names.append(f'{m}_z')

        data = []
        line_idx = 5
        while line_idx < all_lines.__len__():
            d = all_lines[line_idx].split('\t')[1:]
            if len(d) != len(col_names):
                raise ValueError(f'Number of columns in line {line_idx} does not match number of marker columns')
            data.append(d)
            line_idx += 1

        measured_data = pd.DataFrame(data=data, columns=col_names).astype('float', errors='ignore')
        return measured_data


def extract_marker_coordinates(marker_data, marker_name, pos):
    return marker_data[[f'{marker_name}_{axis}' for axis in ['x', 'y', 'z']]].iloc[pos].to_numpy()


def calculate_heading_correction(marker_data, marker_names, R_imu, pos):
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
    marker_x_axis = imu_marker_1 - imu_marker_2
    marker_y_axis = imu_marker_3 - imu_marker_2
    marker_z_axis = np.cross(marker_x_axis, marker_y_axis)
    marker_x_axis = marker_x_axis / np.linalg.norm(marker_x_axis)
    marker_y_axis = marker_y_axis / np.linalg.norm(marker_y_axis)
    marker_z_axis = marker_z_axis / np.linalg.norm(marker_z_axis)
    imu_cs = np.array([marker_x_axis, marker_y_axis, marker_z_axis]).T
    qualisys_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    R_imu_qualisys = R.from_matrix(np.matmul(np.linalg.inv(qualisys_cs), imu_cs))
    quat_marker_data = np.array([R_imu_qualisys.as_quat()])

    # Reihenfolge wird von rnl angewendet -> erst von imu zu cs, dann von cs zu marker
    q_diff = R.from_quat(quat_marker_data[0]) * R_imu[0].inv()
    euler = q_diff.as_euler('yxz', degrees=True)
    q_heading_correction = R.from_euler('y', euler[0], degrees=True)
    return q_heading_correction

def calculate_difference(marker_data, marker_names, R_imu, pos, i=0):
   # i = 0
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
    if any(imu_marker_1 == 0.0) or any(imu_marker_2 == 0.0) or any(imu_marker_3 == 0.0):
        return 0.0
    marker_x_axis = imu_marker_1 - imu_marker_2
    marker_y_axis = imu_marker_3 - imu_marker_2
    marker_z_axis = np.cross(marker_x_axis, marker_y_axis)
    marker_x_axis = marker_x_axis / np.linalg.norm(marker_x_axis)
    marker_y_axis = marker_y_axis / np.linalg.norm(marker_y_axis)
    marker_z_axis = marker_z_axis / np.linalg.norm(marker_z_axis)
    imu_marker_cs = np.array([marker_x_axis, marker_y_axis, marker_z_axis]).T
    qualisys_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    R_imu_marker = R.from_matrix(np.matmul(np.linalg.inv(qualisys_cs), imu_marker_cs))
    quat_imu_marker = np.array([R_imu_marker.as_quat()])

    # Reihenfolge wird von rnl angewendet -> erst von imu zu cs, dann von cs zu marker
    q_diff = R_imu_marker * R_imu.inv()
    return np.linalg.norm(q_diff.as_rotvec(degrees=True))


def extract_segment_orientations(model_path):
    model = osim.Model(model_path)
    state = model.initSystem()

    segment_orientations = {}

    for body in model.getBodySet():
        body_name = body.getName()
        rotation = body.getTransformInGround(state).R().asMat33()
        rotation_matrix = np.array([
            [rotation.get(0, 0), rotation.get(0, 1), rotation.get(0, 2)],
            [rotation.get(1, 0), rotation.get(1, 1), rotation.get(1, 2)],
            [rotation.get(2, 0), rotation.get(2, 1), rotation.get(2, 2)]
        ])
        segment_orientations[body_name] = R.from_matrix(rotation_matrix)
    return segment_orientations


def perform_inverse_kinematics_w_imu_data(measurement_path, cfg, imu_sample_rate, subject_name, exercise):

    start_ts = cfg['start_ts']

    ##### CREATE PATHS ETC ############################################################################################################
    ik_dir = measurement_path / 'ik_imus'
    modelFileName = ik_dir / 'models' / f'scaled_model_initial_pose_{subject_name}_{exercise}.osim'
    resultsDirectory = ik_dir / 'results_imu_ik'

    final_quat_data = []
    positions = []

    # load and preprocess marker data
    marker_data = read_trc_file(str(ik_dir / f'marker_data_osim_format_{subject_name}_{exercise}.trc'))
    marker_data = marker_data[(marker_data['Time'] >= start_ts)]

    # load and preprocess IMU data
    # imus_data = np.load(p/ 'XSens' / 'xsens_data.npy', allow_pickle=True).item()
    # for imu_name, imu_d in imus_data.items():
    #     for axis_name, l in imu_d.items():
    #         imus_data[imu_name][axis_name] = l[int(start_ts * imu_sample_rate):]

    imus_data = pd.read_csv(measurement_path / f'xsens_imu_data_{subject_name}_{exercise}.csv', sep=',')
    time = imus_data['time [s]'][imus_data['time [s]'] >= start_ts].to_numpy()
    stop_ts = time[-1]
    quat_data = pd.DataFrame()
    quat_data['time'] = time

    # segment_orientations = extract_segment_orientations(str(modelFileName))

    registered_imu_data = {}
    for j, (pos, infos) in enumerate(cfg['inverse_kinematics'].items()):
        # if j == 5:
        #     break
        imu_data = imus_data[[f'{infos["imu_name"]}_{axis}' for axis in ['QX', 'QY', 'QZ', 'QW']]].to_numpy()
        R_imu = R.from_quat(imu_data)

        # 1. IMU Transformation: rotate imu to match opensim cs
        R_imu = R.from_euler('x', -90, degrees=True) * R_imu

        q_heading_correction = calculate_heading_correction(marker_data, infos['marker_names'], R_imu, pos)

        # 2. IMU Transformation: apply heading correction
        R_imu_correct_heading = q_heading_correction * R_imu

        deviations_angles = []
        for i, imu_rot in enumerate(tqdm(R_imu_correct_heading)):
            if i > marker_data.shape[0] -1:
                continue
            deviations_angles.append(calculate_difference(marker_data, infos['marker_names'], imu_rot, pos, i=i))

        registered_imu_data[pos] = deviations_angles


    pkl.dump(registered_imu_data, open(measurement_path / 'imu_marker_deviations.pkl', 'wb'))
    debug = True

    # col_names = []
    # df_data = []
    # for pos, quat in registered_imu_data.items():
    #     col_names += [f'{cfg["inverse_kinematics"][pos]["imu_name"]}_{axis}' for axis in ['QX', 'QY', 'QZ', 'QW']]
    #     df_data += [list(quat[:,i] )for i in range(4)]
    # imu_df = pd.DataFrame(np.array(df_data).T, columns=col_names)
    # imu_df['time [s]'] = time
    # imu_df = imu_df[['time [s]'] + col_names]
    # imu_df.to_csv(str(measurement_path / f'xsens_imu_data_segment_registered_{subject_name}_{exercise}.csv'), index=False)
    #
    #
    # ##### Write to .sto file ############################################################################################################
    # header = ['DataRate=100', '\nDataType=Quaternion', '\nversion=3',
    #           '\nOpenSimVersion=4.3-2021-08-27-4bc7ad9', '\nendheader\n']
    #
    # for pos, quat in registered_imu_data.items():
    #     str_rep_quat = []
    #     for i in range(len(time)):
    #         str_rep_quat.append(f'{quat[i,3]}, {quat[i,0]}, {quat[i,1]}, {quat[i,2]}')
    #     quat_data[f'{pos}'] = str_rep_quat
    # quat_file_name = f'segment_registered_imu_data_{subject_name}_{exercise}.sto'
    # hf.write_to_mot_file(quat_data, header=header, filepath=resultsDirectory, filename=quat_file_name)
    #
    #
    # ##### Run IMU IK ############################################################################################################
    # imuIK = osim.IMUInverseKinematicsTool()
    # imuIK.set_model_file(str(modelFileName))
    # imuIK.set_orientations_file(str(resultsDirectory / quat_file_name))
    # imuIK.set_results_directory(str(resultsDirectory))
    # imuIK.set_time_range(0, start_ts)
    # imuIK.set_time_range(1, stop_ts)
    # imuIK.run(False)


    #####
subject_name = 'darryl'
exercise = 'ce'
p = f'../data/{subject_name}/{exercise}/'
IMU_SAMPLERATE = 100

with open(Path(p) / f'metadata_{subject_name}_{exercise}.json', 'r') as stream:
    processing_config = json.load(stream)
perform_inverse_kinematics_w_imu_data(measurement_path=Path(p),
                                                                 cfg=processing_config,
                                                                 imu_sample_rate=IMU_SAMPLERATE,
                                                                 subject_name=subject_name,
                                                                exercise=exercise)

