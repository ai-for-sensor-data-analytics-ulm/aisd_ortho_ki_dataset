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
import processing.helper_fcts as hfp

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


# def extract_marker_coordinates(marker_data, marker_name, pos):
#     return marker_data[[f'{marker_name}_{axis}' for axis in ['x', 'y', 'z']]].iloc[pos].to_numpy()

def extract_marker_coordinates(marker_data, marker_name, pos=None):
    if pos is None:
        return marker_data[[f'{marker_name}_{axis}' for axis in ['x', 'y', 'z']]].to_numpy()
    else:
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


# def calculate_orientation_deviation(marker_data, marker_names, R_imu, pos, i=0):
#    # i = 0
#     if pos == 'toes_r':
#         imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0], i)
#         imu_marker_extra_2 = extract_marker_coordinates(marker_data, marker_names[1], i)
#         imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2], i)
#         imu_marker_21 = imu_marker_1 - imu_marker_extra_2
#         imu_marker_2 = imu_marker_extra_2 + imu_marker_21 / 2
#     else:
#         imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0], i)
#         imu_marker_2 = extract_marker_coordinates(marker_data, marker_names[1], i)
#         imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2], i)
#     if any(imu_marker_1 == 0.0) or any(imu_marker_2 == 0.0) or any(imu_marker_3 == 0.0):
#         return 0.0, np.array([0, 0, 0, 0])
#     marker_x_axis = imu_marker_1 - imu_marker_2
#     marker_y_axis = imu_marker_3 - imu_marker_2
#     marker_z_axis = np.cross(marker_x_axis, marker_y_axis)
#     marker_x_axis = marker_x_axis / np.linalg.norm(marker_x_axis)
#     marker_y_axis = marker_y_axis / np.linalg.norm(marker_y_axis)
#     marker_z_axis = marker_z_axis / np.linalg.norm(marker_z_axis)
#     imu_marker_cs = np.array([marker_x_axis, marker_y_axis, marker_z_axis]).T
#     qualisys_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
#     R_imu_marker = R.from_matrix(np.matmul(np.linalg.inv(qualisys_cs), imu_marker_cs))
#     quat_imu_marker = np.array([R_imu_marker.as_quat()])
#
#     # Reihenfolge wird von rnl angewendet -> erst von imu zu cs, dann von cs zu marker
#     q_diff = R_imu_marker * R_imu.inv()
#     return np.linalg.norm(q_diff.as_rotvec(degrees=True)), R_imu_marker.as_quat()

def fix_marker_coordinate_zeros(marker_coordinates):

    indices = np.argwhere(marker_coordinates == 0.0)
    if len(indices) == 0:
        return marker_coordinates, []
    else:
        zero_ind = sorted(list(set(indices[:,0])))
        for ind in zero_ind:
            marker_coordinates[ind, :] = marker_coordinates[ind-1, :]
        return marker_coordinates, zero_ind


def calculate_orientation_deviation(marker_data, marker_names, R_imu, pos):
    error_indices = []
    if pos == 'toes_r':
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0])
        imu_marker_extra_2 = extract_marker_coordinates(marker_data, marker_names[1])
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2])
        imu_marker_21 = imu_marker_1 - imu_marker_extra_2
        imu_marker_2 = imu_marker_extra_2 + imu_marker_21 / 2
    else:
        imu_marker_1 = extract_marker_coordinates(marker_data, marker_names[0])
        imu_marker_2 = extract_marker_coordinates(marker_data, marker_names[1])
        imu_marker_3 = extract_marker_coordinates(marker_data, marker_names[2])
    if np.any(imu_marker_1 == 0.0) or np.any(imu_marker_2 == 0.0) or np.any(imu_marker_3 == 0.0):
        imu_marker_1, error_ind = fix_marker_coordinate_zeros(imu_marker_1)
        error_indices += error_ind
        imu_marker_2, error_ind = fix_marker_coordinate_zeros(imu_marker_2)
        error_indices += error_ind
        imu_marker_3, error_ind = fix_marker_coordinate_zeros(imu_marker_3)
        error_indices += error_ind
    marker_x_axis = imu_marker_1 - imu_marker_2
    marker_y_axis = imu_marker_3 - imu_marker_2
    marker_z_axis = np.cross(marker_x_axis, marker_y_axis)
    marker_x_axis = marker_x_axis / np.linalg.norm(marker_x_axis, axis=1).reshape(-1,1)
    marker_y_axis = marker_y_axis / np.linalg.norm(marker_y_axis, axis=1).reshape(-1,1)
    marker_z_axis = marker_z_axis / np.linalg.norm(marker_z_axis, axis=1).reshape(-1,1)
    # imu_marker_cs = np.array([marker_x_axis, marker_y_axis, marker_z_axis]).T
    imu_marker_cs = np.zeros((marker_x_axis.shape[0], 3,3))
    for i in range(marker_x_axis.shape[0]):
        imu_marker_cs[i, :, 0] = marker_x_axis[i,:]
        imu_marker_cs[i, :, 1] = marker_y_axis[i, :]
        imu_marker_cs[i, :, 2] = marker_z_axis[i, :]
    qualisys_cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    qualisys_cs = np.tile(qualisys_cs, (marker_x_axis.shape[0], 1, 1))
    R_imu_marker = R.from_matrix(np.matmul(qualisys_cs, imu_marker_cs))

    # check if Orientation length is the same
    min_length = min(len(R_imu_marker), len(R_imu))
    R_imu_marker = R_imu_marker[:min_length]
    R_imu = R_imu[:min_length]
    # Reihenfolge wird von rnl angewendet -> erst von imu zu cs, dann von cs zu marker
    q_diff = R_imu_marker * R_imu.inv()
    return np.linalg.norm(q_diff.as_rotvec(degrees=True), axis=1), R_imu_marker.as_quat(), R_imu.as_quat(), error_indices

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

    # load and preprocess marker data
    marker_data = read_trc_file(str(ik_dir / f'marker_data_osim_format_{subject_name}_{exercise}.trc'))
    marker_data = marker_data[(marker_data['Time'] >= start_ts)]

    imus_data = pd.read_csv(measurement_path / f'xsens_imu_data_{subject_name}_{exercise}.csv', sep=',')
    imus_data = imus_data[(imus_data['time [s]'] >= start_ts)]
    time = imus_data['time [s]'].to_numpy()
    quat_data = pd.DataFrame()
    quat_data['time'] = time

    registered_imu_data = {}
    for j, (pos, infos) in tqdm(enumerate(cfg['inverse_kinematics'].items())):
        imu_data = imus_data[[f'{infos["imu_name"]}_{axis}' for axis in ['QX', 'QY', 'QZ', 'QW']]].to_numpy()
        R_imu = R.from_quat(imu_data)

        # 1. IMU Transformation: rotate imu to match opensim cs
        R_imu = R.from_euler('x', -90, degrees=True) * R_imu

        q_heading_correction = calculate_heading_correction(marker_data, infos['marker_names'], R_imu, pos)

        # 2. IMU Transformation: apply heading correction
        R_imu_correct_heading = q_heading_correction * R_imu

        deviation_angles, marker_orientations, imu_orientations, error_indices = calculate_orientation_deviation(marker_data, infos['marker_names'], R_imu_correct_heading, pos)

        # # 3. calculate deviation
        # deviations_angles = []
        # marker_orientations = []
        # for i, imu_rot in enumerate(R_imu_correct_heading):
        #     if i > marker_data.shape[0] -1:
        #         continue
        #     dev_angle, marker_or = calculate_orientation_deviation(marker_data, infos['marker_names'], imu_rot, pos, i=i)
        #     deviations_angles.append(dev_angle)
        #     marker_orientations.append(marker_or)


        registered_imu_data[pos] = {'deviations_angles': deviation_angles, 'marker_orientations': marker_orientations, 'imu_orientations': imu_orientations, 'error_indices': error_indices}


    pkl.dump(registered_imu_data, open(measurement_path / 'imu_marker_deviations_second_version.pkl', 'wb'))


    #####

all_paths = hf.get_all_folder_paths('../data')
all_paths = hf.filter_paths_by_subpaths(all_paths, [
    'ce', f'{os.sep}ee'])
# all_paths = hf.filter_paths_by_subpaths(all_paths, [
#     'jung-hee/ee'])
# all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['ik_imus']))
all_paths = sorted(hf.remove_paths_with_patterns(all_paths, ['ik_imus']))

for p in tqdm(all_paths):
    subject_name = hfp.extract_subject_name_from_path(p)
    exercise = hfp.extract_exercise_from_path(p)
    p = f'../data/{subject_name}/{exercise}/'
    IMU_SAMPLERATE = 100

    with open(Path(p) / f'metadata_{subject_name}_{exercise}.json', 'r') as stream:
        processing_config = json.load(stream)
    perform_inverse_kinematics_w_imu_data(measurement_path=Path(p),
                                                                     cfg=processing_config,
                                                                     imu_sample_rate=IMU_SAMPLERATE,
                                                                     subject_name=subject_name,
                                                                    exercise=exercise)

