import numpy as np
from pandas import DataFrame
import scipy.signal as sig

## there are given from the website
max_raw_vals = {"Acc1": 32768, "Acc2": 8192, "Gyro": 32768, "Mag": 8192, "Force": 4096}
max_sis = {"Acc1": 2, "Acc2": 2, "Gyro": 1000, "Mag": 2.4, "Force": 5.32}

def extract_data(data, calib):
    millisec = 1000  ## 1 sec
    data = apply_calibration(data, calib)

    ### Gyro Meter from degree to rad
    for axis in ('X', 'Y', 'Z'):
        data[f"Gyro {axis}"] = data[f"Gyro {axis}"] / 180 * np.pi
    ### ignore Magnetic
    truncated_samples = data.drop(['Mag X'], axis=1)
    truncated_samples = truncated_samples.drop(['Mag Y'], axis=1)
    truncated_samples = truncated_samples.drop(['Mag Z'], axis=1)

    ## resample to the same size for the input of the model
    truncated_samples = sig.resample(truncated_samples, num=millisec, axis=0)
    ## log normalize
    truncated_samples = np.sign(truncated_samples) * np.log(abs(truncated_samples) + 1)
    return truncated_samples

def apply_calibration(data: DataFrame, calibration: DataFrame) -> DataFrame:
    # Calibrate accelerometers
    for i in (1, 2):
        for axis in ("X", "Y", "Z"):
            data[f"Acc{i} {axis}"].update(
                apply_calibration_to_value(
                    data[f"Acc{i} {axis}"], max_raw_vals[f"Acc{i}"], max_sis[f"Acc{i}"],
                )
            )
    # Calibrate gyroscope
    for axis in ("X", "Y", "Z"):
        data[f"Gyro {axis}"].update(
            apply_calibration_to_value(
                data[f"Gyro {axis}"],
                max_raw_vals["Gyro"],
                max_sis["Gyro"],
                cal_bias=calibration[f"bg{axis.lower()}"].item(),
            )
        )
    # Calibrate magnetometer
    for axis in ("X", "Y", "Z"):
        data[f"Mag {axis}"].update(
            apply_calibration_to_value(
                data[f"Mag {axis}"],
                max_raw_vals["Mag"],
                max_sis["Mag"],
                cal_bias=calibration[f"bm{axis.lower()}"].item(),
                cal_scale=calibration[f"sm{axis.lower()}"].item(),
            )
        )
    # Calibrate force sensor
    data["Force"].update(
        apply_calibration_to_value(
            data["Force"], max_raw_vals["Force"], max_sis["Force"]
        )
    )
    return data

def apply_calibration_to_value(raw, max_range_raw, max_si=1, cal_bias=0, cal_scale=1):
    return ((raw - cal_bias) / (max_range_raw / max_si)) / cal_scale