## This file is for data splitting in train and test
import os
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import scipy.signal as sig
from stage_2.read_data_pd import read_and_extract_data, get_relevant_label_segment

## stage_2 data only letter
DATA_PATH = "../data/data_stage2/STABILO_CHALLENGE_STAGE_2/"
LETTER_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                'X': 23, 'Y': 24, 'Z': 25, 'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33,
               'i': 34, 'j': 35, 'k': 36, 'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44,
               't': 45, 'u': 46, 'v': 47, 'w': 48, 'x': 49, 'y': 50, 'z': 51, '0': 52} ## '0' blank

def read_data_to_dataset(folder: str, all_data=True, millisec=1000, resample=True):
    persons = []

    if all_data:
        sub_dir = [os.path.join(folder, o) for o in os.listdir(folder)
                   if os.path.isdir(os.path.join(folder, o))]
    else:
        sub_dir = [os.path.join(folder, '7'), os.path.join(folder, '8'), os.path.join(folder, '9'), os.path.join(folder, '10'),
                   os.path.join(folder, '11'), os.path.join(folder, '12'), os.path.join(folder, '13'), os.path.join(folder, '14'),
                   os.path.join(folder, '15'), os.path.join(folder, '16'), os.path.join(folder, '17'), os.path.join(folder, '18'),
                   os.path.join(folder, '19'), os.path.join(folder, '20'), os.path.join(folder, '21'), os.path.join(folder, '22')]
        ## use for testing model for several writer

    for sub in sub_dir:
        single_char_tuples = []
        labeled_data = read_and_extract_data(sub, include_blank= False)
        for ld in labeled_data:
            if resample:
                ### Gyro Meter from degree to rad
                for axis in ('X', 'Y', 'Z'):
                    ld[1][f"Gyro {axis}"] = ld[1][f"Gyro {axis}"] / 180 * np.pi

                ### transform force from N to kg
                # ld[1]["Force"] = ld[1]["Force"] / 9.8
                ### try dataset without Magnetization left only 10 cols
                truncated_samples = ld[1].drop(['Mag X'], axis=1)
                truncated_samples = truncated_samples.drop(['Mag Y'], axis=1)
                truncated_samples = truncated_samples.drop(['Mag Z'], axis=1)
                ## drop force
                # truncated_samples = truncated_samples.drop(['Force'], axis=1)
                # truncated_samples = ld[1]
                truncated_samples = sig.resample(truncated_samples, num=millisec, axis=0)
            else:
                truncated_samples = get_segment_data(ld[1], millisec)
            # length = len(truncated_samples.index)

            if truncated_samples is not None and np.shape(truncated_samples)[0] == millisec:    ## excluded some wierd smaples, which after truncated still longer than 1000 and smaller than 2
                if resample:
                    ### use the log to replace zero mean normalization
                    truncated_samples = np.sign(truncated_samples) * np.log(abs(truncated_samples) + 1)
                    # truncated_samples = (truncated_samples - np.mean(truncated_samples, axis= 0)) / np.std(truncated_samples, axis= 0)
                transfered_number = LETTER_DICT[ld[0]]
                single_char_tuples.append((truncated_samples, transfered_number))
        persons.append(single_char_tuples)
    # return single_char_tuples   ## use for test the single writer
    return persons

def get_segment_data(sample: DataFrame, millisec, force_threshold= None, n_consecutive=1):
    ## set all the segments to millisec length
    truncated_sample = None
    start, end = get_relevant_label_segment(sample= sample, force_thresh= force_threshold, n_consecutive= n_consecutive)
    length = abs(start - end) + 1
    middlepoint = int((start + end)/2)
    new_start = int(middlepoint - millisec/2)
    new_end = int(middlepoint + millisec/2)
    if new_start < 0:
        new_start = 0
        new_end = millisec
    elif new_end > np.shape(sample)[0]:
        new_end = np.shape(sample)[0]
        new_start = new_end - millisec
    truncated_sample = sample.iloc[new_start:new_end]
    return truncated_sample
