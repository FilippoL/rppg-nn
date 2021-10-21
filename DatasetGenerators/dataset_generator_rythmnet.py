import os
from random import choice, randint

import biosppy
import cv2
import h5py
import numpy as np
import pandas as pd

from MapCreator.MapCreator import MapCreator

root_dataset_path = r"../../../../Datasets/COHFACE"  # Path to root folder of the dataset

inverted = False  # Concatenate in an horizontal fashion
time_window = 10  # Time window in seconds
step = 0.5  # Step to move window with
number_roi = 7  # Number of region of interests within a frame
filter_size = 3  # Padding filter size
masking_frequencies = list(range(1, 5))  # Frequency of masked frames

dataset_name_csv = f"tmp_{number_roi}x{number_roi}_no_eyes.csv"  # Name for the temporary .csv files
maps_folder_name = f"maps_{number_roi}x{number_roi}_yuv_no_eyes"  # Name for the folder containing the .jpg maps
pointer_csv_name = f"dataset_pointers_{number_roi}x{number_roi}_no_eyes.csv"  # Name for the final .csv

map_creator = MapCreator()

for subdir, dirs, files in os.walk(root_dataset_path):

    data_path = [file for file in files if file.endswith("hdf5")]
    video_path = [file for file in files if file.endswith("avi")]

    # If directory doesn't contain video or data skip
    if len(data_path) == 0 or len(video_path) == 0: continue

    map_creator.set_video(os.path.join(subdir, video_path[0]))

    # Make the spatio temporal maps and return a tuple of (time_bin_end, map_image)
    maps = map_creator.make_map(masking_frequency=choice(masking_frequencies) if randint(0, 1) else 0)

    names, values = [], []
    for map in maps:
        # Crate dirs for .jpg maps
        os.makedirs(os.path.join(subdir, maps_folder_name), exist_ok=True)
        dir = os.path.join(subdir, maps_folder_name, f"{str(map[0]).replace('.', '_')}.jpg")

        # Write the maps
        cv2.imwrite(dir, map[1])
        names.append(dir), values.append(map[0])

    # Read BVP data
    pulse = h5py.File(os.path.join(subdir, data_path[0]), 'r')["pulse"]

    # Extract hr signal and timestamps
    _, _, _, hr_ticks, hr_values = biosppy.ppg.ppg(pulse, 256, show=False)

    # Time bin start and end to which the map refers
    margins = list(zip(np.subtract(values, 10), values))

    # Mask to extract the value that are in between bins
    boolean_array = [np.logical_and(np.array(hr_ticks) >= i, np.array(hr_ticks) <= j) for (i, j) in margins]

    # Take the mean of all values in given time frame
    arrays = [np.array(hr_values)[arr] for arr in boolean_array]
    means = [np.mean(arr) for arr in arrays]

    # Dump a temporary csv that will be deleted later
    full_path_csv = os.path.join(subdir, dataset_name_csv)
    pd.DataFrame(list(zip(means, names)), columns=["hr", "file"]).to_csv(full_path_csv)

    print(f"Created dataset at: {subdir}")

all_df = []

# Stack all the temps csv and save to a unique csv with columns [HR Mean Value, Path To JPG Map]
for subdir, dirs, files in os.walk(root_dataset_path):
    for file in files:
        if file.endswith("csv"):
            all_df.append(pd.read_csv(os.path.join(subdir, file), index_col=False))
            # Remove the temp csv
            os.remove(os.path.join(subdir, file))

merged = pd.concat(all_df)
merged.drop(merged.columns[0], axis=1, inplace=True)
merged.to_csv(pointer_csv_name)
