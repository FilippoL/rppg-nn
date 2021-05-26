import os

import h5py
import numpy as np

from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor
from map_creator import make_spatio_temporal_maps
import biosppy
import _pickle as cPickle

# Hyper parameters
# root_path = "./data/data_in/sample_COHFACE"
root_path = "H:\\DatasetsThesis\\COHFACE"
readings_per_second = 256
frames_per_second = 20
video_length = 60
dataset = {"map": [], "bvp": [], "hr": []}
dataset_name = "dataset_processed.pickle"

inverted = False  # Concatenate in an horizontal fashion
time_window = 0.5  # Time window in seconds
number_roi = 50  # Number of region of interests within a frame
filter_size = 3  # Padding filter size
masking_frequency = 0.1  # Frequency with which apply a mask (0 to 1)

# Instantiate face detector and processor
fd = FaceDetectorSSD()
fp = FaceProcessor()

binned_pulse_dict = {}
maps = []

for subdir, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith("hdf5"):
            hf = h5py.File(os.path.join(subdir, file), 'r')

            time = np.array(hf["time"])
            pulse = np.array(hf["pulse"])
            bins = np.arange(0,video_length, time_window)

            binned_pulse_tuple = tuple(zip(np.digitize(time, bins) - 1, pulse))
            binned_pulse_dict = {}
            [binned_pulse_dict.setdefault(tup[0], []).append(tup[1]) for tup in binned_pulse_tuple]
            print(f"Finished processing {file}")
            dataset["bvp"] = binned_pulse_dict
            _, _, _, _, hr = biosppy.bvp.bvp(pulse, 128)
            dataset["hr"] = hr

        elif file.endswith("avi"):
            maps = make_spatio_temporal_maps(fd, fp, os.path.join(subdir, file),
                                      time_window,
                                      number_roi,
                                      filter_size,
                                      masking_frequency,
                                      inverted)

            print(f"Finished processing {file}")
            dataset["map"] = maps

        if len(dataset["map"]) != 0 and len(dataset["bvp"]) != 0:
            print(f"Completed maps and signal sequencing for {subdir}")

            full_path = os.path.join(root_path, subdir, dataset_name)
            with open(full_path, "wb") as output_file:
                cPickle.dump(dataset, output_file)
            output_file.close()
            print(f"Created dataset at: {full_path}")

            # Reset
            binned_pulse_dict = {}
            maps = []
            dataset = {"map": [], "bvp": [], "hr": []}
