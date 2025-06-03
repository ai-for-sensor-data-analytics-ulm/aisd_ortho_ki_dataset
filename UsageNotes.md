# Usage Notes

This file documents protocol deviations and sensor-specific issues encountered during the data collection and postprocessing stages. These notes are intended to improve transparency and assist researchers in interpreting the data correctly.

---

## Participant-Specific Annotations

- **Hans – dorsiflexion (rd):**  
  Hans only performed the `rd` exercise. No data is available for `rgs`, `ng`, or `gwo`.

- **Laurel – gait with orthosis (gwo):**  
  The IMU on the right tibia (`tibia_r`) was mounted rotated by approximately 180°. This orientation error was corrected during postprocessing and the corrected data is stored in `xsens_imu_data_segment_registered_laurel_gwo.csv`.

- **Hans – dorsiflexion (rd):**  
  Instead of the `tibia_r` IMU, an alternative IMU labeled `prop1` was used. The mapping is correctly specified in the corresponding metadata file.

- **Austra – normal gait (ng):**  
  The sternum IMU drops out starting at sample 39,949. This dropout has negligible impact on the data quality for this exercise.

---

## Delayed T-Pose Recordings

For the following subjects and exercises, the participant did not assume the required T-pose at the beginning of the recording. Instead, the static posture was assumed only after several seconds:

- **Etsuko – rgs**
- **Julia – rd**
- **Erna – ng and gwo**
- **Yaxkin – rgs**

As a result, sensor-to-segment alignment (used for orientation transformation and inverse kinematics) was performed using a delayed timestamp, typically 5–10 seconds after recording started. This alignment time is defined in the corresponding `metadata_<subject>_<exercise>.json` file under the field `start_ts`. The file `xsens_imu_data_segment_registered_<subject>_<exercise>.csv` starts at this later timestamp.

---

## Metadata File Structure

Each trial is accompanied by a metadata file named `metadata_<subject>_<exercise>.json`. These files contain trial-specific information essential for reproducing the processing steps accurately.

The structure is as follows:

```json
{
  "start_ts": 12.34,
  "scaling": {
    "no_scaling_for": [],
    "no_ik_for": []
  },
  "inverse_kinematics": {
    "tibia_r": {
      "imu_name": "XSens_LowerLeg_Right",
      "marker_names": ["R_SHIN1", "R_SHIN2", "R_SHIN3", "R_SHIN4"]
    },
    ...
  }
}
```

### Fields:
- start_ts: Start time (in seconds) used for sensor alignment and trimming. Defines when the subject was in T-pose for model calibration.

- scaling: Contains all deviations from standard scaling procedure. For example, if certain markers could not be tracked reliably during the T-pose, the corresponding segments were excluded from scaling.

- inverse_kinematics: Maps each IMU to its corresponding body segment and specifies the marker triplet used for orientation validation.
