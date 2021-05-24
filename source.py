import math
import os
from itertools import product

import cv2
import dlib
import numpy as np
import random
from tqdm import tqdm
from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor
from FaceManager.helpers import pad

# TODO: [x] Explore signal on a few seconds range, possibly visualize on Tableau
# TODO: [x] Check documents on oximetry
# TODO: [x] Port BVP signal on Tableau
# TODO: [x] Start spatio-temporal maps creation
# TODO: [x] List all requirements first
# TODO: [x] Table for methods and their score per requirements
# TODO: [x] Expand error section on possible other metrics and why the ones have been chosen
# TODO: [x] Filter nxn conv instead of mean (filter size = 3x3)
# TODO: [x] Test validity by enhancing one color channel
# TODO: [x] Change pipeline figures with your own
# TODO: [x] Concatenate onto other dimension
# TODO: [x] Adjust face cropping after alignment
# TODO: [x] Masking/Augmentation
# TODO: [] Build ResNet18 wrapper class
# TODO: [] Study loss function


# Hyper parameters
inverted = True  # Concatenate in an horizontal fashion
time_window = 0.5  # Time window in seconds
number_roi = 5  # Number of region of interests within a frame
video_path = "./data/data_in/data.avi"  # "D:\\Downloads\\test.mp4" # Path to video file
filter_size = 3  # Padding filter size
masking_frequency = 0.1 # Frequency with which apply a mask (0 to 1)

def main():

    # Instantiate face detector and processor
    fd = FaceDetectorSSD()
    fp = FaceProcessor()

    # Check if video path is a valid directory
    assert os.path.isfile(video_path), f"{video_path} is not a valid path."

    # Instantiate video capture
    video_capture = cv2.VideoCapture(video_path)

    # Calculate number of frames in set time window
    fps = video_capture.get(cv2.cv2.CAP_PROP_FPS)
    n_tot_frames = int(video_capture.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
    n_frames_per_batch = math.ceil(time_window * fps)  # Round up

    segmented_frames = []

    for _ in tqdm(range(n_tot_frames)):
        success, img = video_capture.read()
        original_img = img.copy()
        result = fd.detect_face(img)
        if not result:
            print(f"No face could be found by {fd.detector_name} face detector.\nSkipping.")
            continue

        top, bottom, left, right = result["bbox_indices"]
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(original_img, rect)
        aligned_and_detected = fp.align(original_img, landmarks, [left, right, top, bottom])

        yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)
        h, w = yuv_aligned_face.shape[:2]
        target_w = (w + (number_roi - (w % number_roi))) if w % number_roi != 0 else w
        target_h = (h + (number_roi - (h % number_roi))) if h % number_roi != 0 else h
        yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h, filter_size)
        blocks = fp.divide_roi_blocks(yuv_align_padded_face, (number_roi, number_roi))
        segmented_frames.append(blocks)

        if len(segmented_frames) != n_frames_per_batch: continue

        size = [0, len(blocks) ** 2] if not inverted else [len(blocks) ** 2, 0]
        out_map = np.empty(size, dtype=np.float)

        for frame in segmented_frames:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                if inverted: i, j = j, i
                mean = np.mean(frame[i][j].reshape(-1, 3), 0)
                means.append(mean)
            out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)

        n_masked_frames = np.ceil(out_map.shape[1] * masking_frequency)
        mask = np.zeros_like(out_map[0,0])
        rnd_indices = random.sample(range(out_map.shape[1]), int(n_masked_frames))

        for idx in rnd_indices:
            out_map[:,idx] = mask

        out_map = out_map.reshape((-1, n_frames_per_batch, 3))
        segmented_frames = []


if __name__ == "__main__":
    main()
