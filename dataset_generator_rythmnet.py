import _pickle as cPickle
import math
import os
from datetime import datetime
from random import randint

import cv2
import face_recognition
import h5py
import numpy as np
from tqdm import tqdm


root_path = "./data/data_in/sample_COHFACE"
# root_path = "G:\\DatasetsThesis\\COHFACE"
readings_per_second = 256
frames_per_second = 20
video_length = 60
videos_per_folder = 4
dataset = {"x": [], "y": []}
dataset_name = "dataset_face_only.pickle"

for subdir, dirs, files in os.walk(root_path):
    for file in files:
        if not os.path.isfile(os.path.join(root_path, subdir.split("\\")[-2], dataset_name)):
            file_extension = file.split(".")[1]
            if file.endswith("avi"):
                pass
            elif file.endswith("hdf5"):
                pass