import numpy as np
import pickle as pkl
import pandas as pd
from scipy.spatial.transform import Rotation as R
import opensim as osim
import matplotlib.pyplot as plt
import os
import tol_colors as tc
from pathlib import Path
from scipy.signal import butter, filtfilt


def create_opensim_file(filepath, filename, data, imu_names, samplerate):

    str_data = pd.DataFrame()
    time_increment = 1/samplerate
    str_data['time'] = np.arange(data.shape[0]/samplerate, step=time_increment)
    str_data['time'] = str_data['time'].astype(str)
    for imu in imu_names:
        quat_cols = [f"{axis}_{imu}" for axis in ['w', 'i', 'j', 'k']]
        str_data[imu] = data[quat_cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1).to_list()

    header = [f'DataRate={samplerate}', '\nDataType=Quaternion', '\nversion=3',
              '\nOpenSimVersion=4.3-2021-08-27-4bc7ad9', '\nendheader\n']

    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)

    with open(filepath / filename, 'w') as f:
        f.writelines(header)

        f.write('\t'.join(list(str_data.columns)) + '\n')

        for row in str_data.values.tolist():
            row_str = '\t'.join(row) + '\n'
            f.write(row_str)
    return

offset = 3
base_path = Path('../data/')
start_ts = 40
stop_ts = 150
subject_name = 'gregers'
exercise_name = 'ng'

data_path = base_path / subject_name / exercise_name

data = pkl.load(open(data_path / 'imu_marker_deviations_second_version.pkl', 'rb'))

quat_data = pd.DataFrame()

for imu_name, d in data.items():
    R_imu_marker = R.from_quat(d['marker_orientations'])
    R_imu = R.from_quat(d['imu_orientations'])
    q_diff = R_imu_marker[:len(R_imu_marker)-offset] * R_imu[offset:].inv()
    q_diff = q_diff * R.from_euler('XY', [180, 180], degrees=True)
    q_diff = q_diff.as_quat()
    col_names = [f"{axis}_{imu_name}" for axis in ['w', 'i', 'j', 'k']]
    quat_data[col_names] = q_diff

imu_names = list(data.keys())

create_opensim_file(data_path, 'deviations.sto', quat_data, imu_names, samplerate=100)

modelFileName = '../opensim_templates/model_template/Unrestricted_Full_Body_Model.osim'

os.makedirs(data_path / 'debug', exist_ok=True)

imuIK = osim.IMUInverseKinematicsTool()
imuIK.set_model_file(str(modelFileName))
imuIK.set_orientations_file(str(data_path/'deviations.sto'))
imuIK.set_results_directory(str(data_path/'debug'))
imuIK.set_time_range(0, start_ts)
imuIK.set_time_range(1, stop_ts)
imuIK.run(False)


debug = True
