> ⚠️ **Repository under construction**
### Publication Status
This repository accompanies the manuscript:

**Spilz, A., Oppel, H., Werner, J., Stucke-Straub, K., Capanni, F., Munz, M.
> *GAITEX: Human motion dataset from impaired gait and rehabilitation exercises of inertial and optical sensor data*  
> Status: Under review 
Preprint: https://zenodo.org/records/15792584

The repository will be finalized and versioned upon acceptance.


# GAITEX

## Overview

This dataset enables the development and evaluation of biomechanical models and classification systems in the context of physiotherapeutic and gait-related movement analysis. It includes synchronized multimodal data from 19 healthy adults, collected during physiotherapeutic exercises and constrained/free gait tasks. Each session features:

- **9 low-cost IMUs (Xsens MTw Awinda)**  
- **35 (human motion) + 36 (imu motion) optical motion capture markers (Qualisys system)**

We provide raw and processed data, subject-specific musculoskeletal models, and code to replicate all postprocessing steps described in the paper, including orientation transformation and inverse kinematics using OpenSim.

---

## Repository Structure

```bash
aisd_ortho_ki_dataset/
├── processing/              # Scripts for postprocessing and validation
│   ├── run_scaling.py
│   ├── run_imu_ik.py
│   ├── evaluate_orientation_deviation.py
│   ├── helper_fcts.py
│   └── config/
├── analysis/               # Scripts to replicate the performed technical validation  
│   ├── orientation_deviation_plot.py  
├── opensim_templates/      # Needed templates for OpenSim InverseKinematics and Scaling
├── docs/                   # Supplementary material, e.g., UsageNotes.md
├── environment.yml         # Conda environment specification
└── UsageNotes.md           # Additional informations about the dataset and details for individual measurements
└── README.md               # You are here

```
---
This repository provides all tools necessary to reproduce the postprocessing and technical validation of the AISD-Ortho-KI dataset. Below you will find an overview of each script, how to use it, and a diagram illustrating the recommended workflow.

---
## Getting Started

### 1. Install Dependencies

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment:

```bash
conda env create -f environment.yml
conda activate aisd_ortho_ki
```

### 2. Download the Dataset
The full dataset is hosted on Zenodo (DOI to be inserted). After download, extract the contents to a local data/ directory structured as follows:

```bash
data/
├── austra/
│   ├── rd/
│   │   ├── qualisys_marker_data_austra_rd.csv
│   │   ├── xsens_imu_data_segment_registered_austra_rd.csv
│   │   ├── xsens_imu_data_austra_rd.csv
│   │   ├── timestamps_austra_rd.csv
│   │   ├── metadata_austra_rd.json
│   │   ├── ik_imus/
│   │   │   ├── models/
│   │   │   │   ├── geometry/
│   │   │   │   │   ├── ...
│   │   │   │   ├── scaled_model_austra_rd.osim
│   │   │   │   ├── scaled_model_initial_pose_austra_rd.osim
│   │   │   ├── results_scaling/
│   │   │   │   ├── scaled_markers_austra_rd.xml
│   │   │   │   ├── scaled_initial_pose_austra_rd.mot
│   │   │   ├── results_imu_ik/
│   │   │   │   ├── ik_segment_registered_imu_data_austra_rd.mot
│   │   │   │   ├── ik_segment_registered_imu_data_austra_rd_orientationErrors.sto
│   │   │   │   ├── segment_registered_imu_data_austra_rd.sto
│   │   │   ├── marker_data_osim_format_austra_rd.trc
│   │   │   ├── scaling_settings_austra_rd.xml
│   ├── rgs/
│   │   ├── ...
│   ├── ng/
│   │   ├── ...
│   ├── gwo/
│   │   ├── ...
│── darryl/
│   ├── ...
...

```
Further information about the data included can be derived from the accompaning paper. 
## Postprocessing Workflow

The main script for executing the postprocessing pipeline is [`run_processing_pipeline.py`](processing/run_processing_pipeline.py). It orchestrates all key steps required to generate joint angle trajectories from raw sensor data, including:

1. **Scaling** of a generic musculoskeletal model to match subject-specific anthropometry (`scaling.py`)
2. **Registration** of IMU orientations to the corresponding OpenSim body segments
3. **Inverse Kinematics** computation using IMU data (`imu_inverse_kinematics.py`)

All paths and settings required for processing are defined in the configuration file `processing/config/processing_config.yaml`. This file allows flexible customization of the pipeline without modifying the script itself.

---

### Key Configuration Parameters (`processing_config.yaml`)

| Key                              | Description                                                                                                   |
|----------------------------------|---------------------------------------------------------------------------------------------------------------|
| `path_template_model`            | Path to the OpenSim model file (`.osim`). We supply the model used by us in 'opensim_templates/' directory    |
| `path_template_scaling_settings` | Path to the scaling settings. We supply the scaling settings used by us in the 'opensim_templates/' directory |
| `imu_samplerate`                 | Samplerate in Hz of the used IMUs. (in our case 100 Hz)                                                       |
| `process_subjects`               | If `all` the data of all subjects is processed, otherwise only the listed subjects are processed              |
| `process_exercises`              | If `all` the data of all exercises is processed, otherwise only the listed exercises are processed            |

You can adapt this file to suit your own processing preferences.

---

### CLI Usage

You can run the pipeline for a specific subject and task directly from the command line:
```bash
python processing/run_processing_pipeline.py \
    --root ../data \
    --config processing_config.yaml \
    --log processing.log
```
### CLI Arguments

| Key                 | Description                                            |
|---------------------|--------------------------------------------------------|
| `root`              | Relative path to directory where the dataset is stored |
| `config`            | Relative path to the settings.yaml file                |
| `log`               | Relative path to an optional logging file              |

---



## Orientation Deviation Evaluation

To assess the spatial alignment between the XSens-derived and marker-based orientation estimates, this repository provides the script [`analysis/orientation_deviation_plot.py`](analysis/orientation_deviation_plot.py). It replicates the validation presented in the *Technical Validation* section of the accompanying [Nature Scientific Data manuscript](#).

### Purpose

Each IMU was equipped with a rigid marker plate, enabling us to compute its true orientation using the Qualisys optical motion capture system. The script compares:

- **Estimated IMU orientation**: as computed by the Xsens Kalman filter and transformed into the OpenSim frame.
- **Marker-based IMU orientation**: derived from the optical markers mounted on the IMU.

The deviation between these two orientations is expressed as an angular difference over time (in degrees), providing insights into calibration accuracy, alignment quality, and temporal consistency.

---

### How It Works

The script performs the following steps:

1. **Loads IMU data and marker data** for a given subject and task.
2. **Reconstructs marker-based orientations** for each IMU from the relative position of three optical markers.
3. **Applies heading correction** to align IMU and marker coordinate systems.
4. **Computes rotational deviation** as the angular difference between both orientation sources.
5. **Plots time-resolved deviations** for each IMU position.

---

### 🧾 CLI Usage

You can run the script for a specific subject and task directly from the command line:
```bash
python analysis/orientation_deviation_plot.py \
    --subject erna \
    --exercise rd \
    --base_path ../data
```
### CLI Arguments

| Key         | Description                      |
|-------------|----------------------------------|
| `subject`   | subject identifier (e.g. erna)   |
| `exercise`  | exercise identifier (e.g. rd)    |
| `base_path` | relative path to the stored data |



## Citation
If you use this dataset or the accompanying code, please cite:
    
```bibtex
@article{spilz2025ortho_ki,
  author       = {Spilz, Andreas and Oppel, Heiko and Werner, Jochen and Munz, Michael},
  title        = {TBD},
  journal      = {Nature Scientific Data},
  year         = {2025},
  note         = {Under review},
  url          = {https://github.com/ai-for-sensor-data-analytics-ulm/aisd_ortho_ki_dataset}
}
```

## License
This project is licensed under the MIT License

## Acknowledgements
We thank all study participants and staff at Ulm University of Applied Sciences.
For questions, suggestions, or collaboration inquiries, please contact Michael Munz or open an issue on GitHub.
Funded by Carl Zeiss Foundation
