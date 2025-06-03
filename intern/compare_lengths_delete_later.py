import numpy as np
import pandas as pd
from pathlib import Path


subject = 'neves'
exercise = 'rd'

base_path = Path(f'../data/{subject}/{exercise}')

qtm = pd.read_csv(base_path/f'qualisys_marker_data_{subject}_{exercise}.csv')
xsens = pd.read_csv(base_path/f'xsens_imu_data_{subject}_{exercise}.csv')
segment = pd.read_csv(base_path/f'xsens_imu_data_segment_registered_{subject}_{exercise}.csv')

print(f'QTM length: {len(qtm)}')
print(f'Xsens length: {len(xsens)}')
print(f'Segment length: {len(segment)}')