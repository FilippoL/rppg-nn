import _pickle as cPickle
import os
from random import choice, randint

import biosppy
import cv2
import h5py
import numpy as np
import pandas as pd
from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor
from map_creator import make_spatio_temporal_maps

# Hyper parameters
root_path = "D:\\Documents\\Programming\\Python\\thesisProject\\data\\data_in\\sample_COHFACE"
# root_path = "D:\\Documents\\Programming\\Python\\thesisProject\\data\\data_in"
# root_path = "H:\\DatasetsThesis\\COHFACE"

video_length = 60
dataset = {"map": [], "bvp": [], "hr": []}
dataset_name_pickle = "dataset_processed.pickle"
dataset_name_csv = "dataset_processed.csv"

fd = FaceDetectorSSD()
fp = FaceProcessor()

inverted = False  # Concatenate in an horizontal fashion
time_window = 10  # Time window in seconds
number_roi = 5  # Number of region of interests within a frame
filter_size = 3  # Padding filter size
masking_frequencies = list(range(1, 3))

f = open("good.txt", "r")
dirs = [line[:-1] for line in f.readlines()]

for subdir in dirs:
    for file in os.listdir(subdir):
# for subdir, dirs, files in os.walk(root_path):
#     for file in files:
        if file.endswith("hdf5"):
            hf = h5py.File(os.path.join(subdir, file), 'r')

            time = np.array(hf["time"])
            pulse = np.array(hf["pulse"])
            bins = np.arange(0, video_length, 0.5)

            binned_pulse_tuple = tuple(zip(np.digitize(time, bins) - 1, pulse))
            binned_pulse_dict = {}
            [binned_pulse_dict.setdefault(tup[0], []).append(tup[1]) for tup in binned_pulse_tuple]
            dataset["bvp"] = binned_pulse_dict
            _, _, _, t_hr, hr = biosppy.bvp.bvp(pulse, 256, show=False)
            bins = np.arange(0, video_length, 0.5)
            means = []
            binned_hr_tuple = list(zip(t_hr, hr))
            dataset["hr"] = binned_hr_tuple

        elif file.endswith("avi"):
            masking_frequency = choice(masking_frequencies) if randint(0, 1) else 0

            maps = make_spatio_temporal_maps(fd, fp, os.path.join(subdir, file),
                                             time_window,
                                             number_roi,
                                             filter_size,
                                             masking_frequency,
                                             inverted)
            names = []
            for map in maps:
                os.makedirs(os.path.join(subdir, "maps"), exist_ok=True)
                cv2.imwrite(f"{subdir}\\maps\\{str(map[0]).replace('.','_')}.jpg", map[1])
                names.append(os.path.join(subdir,"maps",f"{str(map[0]).replace('.','_')}.jpg"))

            dataset["map"] = list(zip(names, [m[1] for m in maps]))

        if len(dataset["map"]) != 0 and len(dataset["bvp"]) != 0:
            full_path_pickle = os.path.join(subdir, dataset_name_pickle)
            full_path_csv = os.path.join(subdir, dataset_name_csv)


            hr_ticks = [b[0] for b in dataset["hr"]]
            hr_values = [b[1] for b in dataset["hr"]]

            maxs_end = [float(name.split("\\")[-1][:-4].replace("_", ".")) for name in [value[0] for value in dataset["map"]]]
            maxs_start = [maxx - 10 for maxx in maxs_end]
            boolean_array = [np.logical_and(np.array(hr_ticks) >= i, np.array(hr_ticks) <= j) for (i, j) in
                             list(zip(maxs_start, maxs_end))]
            arrays = [np.array(hr_values)[arr] for arr in boolean_array]
            means = [np.mean(arr) for arr in arrays]
            names = [name[0] for name in dataset["map"]]
            pd.DataFrame(list(zip(means, names)), columns=["hr", "file"]).to_csv(full_path_csv)


            with open(full_path_pickle, "wb") as output_file:
                cPickle.dump(dataset, output_file)
            output_file.close()

            print(f"Created dataset at: {full_path_pickle}")
            dataset = {"map": [], "bvp": [], "hr": []}
