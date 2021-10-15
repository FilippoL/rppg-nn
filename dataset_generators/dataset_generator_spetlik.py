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

root_path = "G:\\DatasetsThesis\\COHFACE"
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
            if file_extension == "avi":
                video_capture = cv2.VideoCapture(os.path.join(subdir, file))
                n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frames_buffer = []
                tot_frames = frames_per_second * video_length
                face = np.array
                for frame_number in tqdm(range(tot_frames)):
                    ret, frame = video_capture.read()
                    new_face_locations = face_recognition.face_locations(frame)
                    (top, right, bottom, left) = new_face_locations[0]
                    face = frame[top:bottom, left:right] if top else face
                    frames_buffer.append(face)
                cv2.imwrite(f"./data/data_out/faces/{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
                            frames_buffer[randint(0, len(frames_buffer) - 1)])
                dataset["x"].append(frames_buffer)
                if len(frames_buffer) == tot_frames:
                    print(f"Mismatching frames found at {subdir}")

            elif file_extension == "hdf5":
                hdf5_file = h5py.File(os.path.join(subdir, file), 'r')
                bvp_raw = np.array(hdf5_file['pulse'])
                seq = np.array(range(0, bvp_raw.shape[0], readings_per_second))
                splitted = np.split(np.array(bvp_raw), seq)[1:61]
                sampled = [d[::math.ceil(readings_per_second / frames_per_second)] for d in splitted]
                flattened = np.concatenate(sampled).ravel()
                dataset["y"].append(flattened)
                tot_readings = frames_per_second * video_length
                if len(flattened) == tot_readings:
                    print(f"Mismatching readings found at {subdir}")

            if len(dataset["x"]) == videos_per_folder and len(dataset["y"]) == videos_per_folder:
                folder = subdir.split("\\")[-2]
                full_path = os.path.join(root_path, folder, dataset_name)
                with open(full_path, "wb") as output_file:
                    cPickle.dump(dataset, output_file)
                output_file.close()
                dataset = {"x": [], "y": []}
                print(f"Created dataset at: {full_path}")
        else:
            print(f"Found dataset {dataset_name} at {subdir}, skipping this.")
