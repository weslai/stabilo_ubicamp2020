import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from pandas import DataFrame

from stage_1.utils import as_windowed_np

ic.includeContext = True


max_raw_vals = {"Acc1": 32768, "Acc2": 8192, "Gyro": 32768, "Mag": 8192, "Force": 4096}
max_sis = {"Acc1": 2, "Acc2": 2, "Gyro": 1000, "Mag": 2.4, "Force": 5.32}

BLANK_CHAR_LABEL = "0"
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_CLASSES = [BLANK_CHAR_LABEL] + list(CHARS)  # Add 'no char'/blank class


def read_calibration(folder: str) -> DataFrame:
    path = os.path.join(folder, "calibration.txt")
    return pd.read_csv(path, delimiter=":", header=None, index_col=0).T


def read_sensor_data(folder: str) -> DataFrame:
    path = os.path.join(folder, "sensor_data.csv")
    return pd.read_csv(path, delimiter=";", index_col=0).drop(columns=["Time"])


def read_labels(folder: str) -> DataFrame:
    path = os.path.join(folder, "labels.csv")
    return pd.read_csv(path, delimiter=";", index_col=None)


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


def interpolate_to_equidistant(data: DataFrame) -> DataFrame:
    # TODO: Is interpolation really necessary since it could also only be related to
    # bluetooth, since it should be sampled at 100 Hz.
    # see https://stabilodigital.com/data/
    return data


def get_label_per_t(data: DataFrame, labels: DataFrame) -> DataFrame:
    """Add a label for each time step in data.

    Args:
        data (DataFrame): Sensor data of shape [T, F], where T time and F is feature
            dimension.
        labels (DataFrame): Labels of shape [N, 3] containing N char labels with start
            and end position.

    Returns:
        label_per_t (DataFrame): Label for each time step.
    """
    label_per_t = []
    label_gen = labels.iterrows()
    cur_label = next(label_gen)[1]  # Get the first label
    for data_sample in data.iterrows():
        cur_t = data_sample[0]
        cur_start = cur_label["start"]
        cur_end = cur_label["stop"]
        if cur_t >= cur_end:
            # Update the label. This assumes
            try:
                cur_label = next(label_gen)[1]
            except StopIteration:
                # We don't have any labels for these time steps anymore
                # Just use the current label with a time step in the future. This will
                # result in a blank label.
                cur_label["start"] = cur_t + 1
                cur_label["end"] = cur_t + 2
            # Also update start and end labels
            cur_start = cur_label["start"]
            cur_end = cur_label["stop"]
        if cur_start <= cur_t < cur_end:
            # TODO: Use char or index of CHAR_CLASSES here?
            label_per_t.append(cur_label["Label"])
        else:  # cur_t < cur_label or cur_t >= cur_label
            label_per_t.append(BLANK_CHAR_LABEL)
    return DataFrame(label_per_t, columns=["Label"])


def split_data_by_label(
    data: DataFrame, labels: DataFrame
) -> List[Tuple[str, DataFrame]]:
    """Split data into parts that correspond to a char.

    Args:
        data (DataFrame): Sensor data of shape [T, F], where T time and F is feature
            dimension.
        labels (DataFrame): Labels of shape [N, 3] containing N char labels with start
            and end position.

    Returns:
         labeled data (List[str, DataFrame]): List of tuples containing the label char
            and the corresponding sensor data.
    """
    out = []
    for _, label in labels.iterrows():
        df = data[(data.index >= label["start"]) & (data.index < label["stop"])]
        out.append((label["Label"], df))
    return out


def extract_blanks(
    data: DataFrame, labels: DataFrame, min_len_ms: int = 50
) -> List[Tuple[str, DataFrame]]:
    """Extract blank parts between the labeled chars.

    Args:
        data (DataFrame): Sensor data of shape [T, F], where T time and F is feature
            dimension.
        labels (DataFrame): Labels of shape [N, 3] containing N char labels with start
            and end position.

    Returns:
         labeled data (List[str, DataFrame]): List of tuples containing the blank label
            and the corresponding sensor data.
    """
    out = []
    label_gen = labels.iterrows()  # Label generator
    _, label = next(label_gen)
    for _, next_label in label_gen:
        df = data[(data.index >= label["stop"]) & (data.index < next_label["start"])]
        if df.shape[0] > 1 and df.index[-1] - df.index[0] >= min_len_ms:
            out.append((BLANK_CHAR_LABEL, df))
        label = next_label
    return out


def get_relevant_label_segment(
    sample: DataFrame, force_thresh=None, n_consecutive=1
) -> Tuple[int, int]:
    """Get the relevant segment, that according to the force only contains the part
        where the letter was actually written.

    Args:
        sample (DataFrame): A sample containing sensor data of exactly one letter.
        force_thresh (Optional[float]): Threshold that is used to detect if the pen is
            currently used to write a letter. If 'None' use 1% of max_range.
        n_consecutive (int): Number of consecutive samples that are above the threshold.

    Returns:
        start (int): Start position in [ms] where 'Force' is > 0, i.e. the pen is
            actually writing.
        end (int): End of force application.
    """
    assert isinstance(sample, DataFrame)
    force = sample["Force"].to_numpy()
    if force_thresh is None:
        force_thresh = 0.01 * max_sis["Force"]
    mask = force > force_thresh
    if n_consecutive > 1:
        mask = as_windowed_np(mask, window_length=n_consecutive)
        mask = mask.min(axis=-1)
        # Extend mask again, so it has the same shape as force
        mask = np.append(mask, [mask[-1]] * n_consecutive)
    start = mask.argmax(axis=0)
    end = mask.shape[0] - mask[::-1].argmax(axis=0) - 1
    end += n_consecutive - 1
    return start, end


def idx_to_s(index, offset=None):
    if offset is None:
        offset = 0.0
    elif offset == "min":
        offset = index.min()
    return (index - offset) / 1000


def read_and_extract_data(folder: str, include_blank=True):
    calib = read_calibration(folder)
    data = read_sensor_data(folder)
    labels = read_labels(folder)
    data = apply_calibration(data, calib)
    labeled_data = split_data_by_label(data, labels)
    if include_blank:
        blank_data = extract_blanks(data, labels)
        labeled_data.extend(blank_data)
    return labeled_data


def test_pipeline(folder):
    """Test the data loading pipeline."""
    calib = read_calibration(folder)
    data = read_sensor_data(folder)
    labels = read_labels(folder)
    data = apply_calibration(data, calib)
    labeled_data = split_data_by_label(data, labels)
    label, sample = labeled_data[0]
    start, end = get_relevant_label_segment(sample, force_thresh=0, n_consecutive=3)
    t_offset = sample.index.min()
    plt.plot(idx_to_s(sample.index, "min"), sample["Force"])
    plt.plot(
        idx_to_s(sample.index[start], t_offset), sample["Force"].iat[start], "ro",
    )
    plt.plot(idx_to_s(sample.index[end], t_offset), sample["Force"].iat[end], "ro")
    plt.savefig("test_relevant_points.png")

    feat_names = list(sample)
    fig, axes = plt.subplots(len(feat_names), figsize=(10, 20), sharex=True)
    time = idx_to_s(sample.index, "min")
    for i, f_n in enumerate(feat_names):
        feat = sample[f_n]
        axes[i].plot(time, feat)
        axes[i].set_title(f_n)
    plt.savefig("test_all_feat.png")


def plot_histograms_per_feat(base_folder):
    from collections import defaultdict

    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    features = defaultdict(list)
    # Iterate over all writer folders
    for folder in subfolders:
        data = read_and_extract_data(folder)
        feat_names = list(data[0][1])
        # Get the relevant segment where force is applied
        relevant_segments = list(
            get_relevant_label_segment(sample) for _, sample in data
        )
        # Iterate over all featues and concatenate the relevant part of the samples
        for f_n in feat_names:
            feats = [
                sample[f_n][start:end].to_numpy()
                for (_, sample), (start, end) in zip(data, relevant_segments)
            ]
            features[f_n].append(np.concatenate(feats))
    for f_n, values in features.items():
        f = plt.figure()
        plt.hist(values)
        plt.savefig(f"hist_{f_n}.png")
        plt.close(f)


if __name__ == "__main__":
    plot_histograms_per_feat(sys.argv[1])