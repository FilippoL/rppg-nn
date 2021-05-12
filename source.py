import math
import os
from itertools import product

import cv2
import dlib
import numpy as np

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
# TODO: [] Concatenate onto other dimension
# TODO: [] Adjust face cropping after alignment
# TODO: [] Masking/Augmentation
# TODO: [] Build ResNet18 wrapper class
# TODO: [] Study loss function


### HYPERPARAMETERS
inverted = False  # Concatenate in an horizontal fashion
time_window = 0.5  # Time window in seconds
number_roi = 5  # Number of region of interests within a frame
video_path = "./data/data_in/me.avi"  # "D:\\Downloads\\test.mp4" # Path to video file
filter_size = 3  # Padding filter size


def main():
    # Instantiate face detector and processor
    fd = FaceDetectorSSD()
    fp = FaceProcessor()

    # Check if video path is a valid directory
    assert os.path.isfile(video_path), f"{video_path} is not a valid path."

    # Instantiate video capture
    video_capture = cv2.VideoCapture(video_path)
    # Read the first frame out of the loop to set the loop condition to True
    success, image = video_capture.read()

    # Calculate number of frames in set time window
    fps = video_capture.get(cv2.cv2.CAP_PROP_FPS)
    n_frames_per_batch = math.ceil(time_window * fps)  # Round up

    segmented_frames = []
    while success:
        success, img = video_capture.read()
        original_img = img.copy()
        result = fd.detect_face(img)
        if not result:
            print(f"No face could be found by {fd.detector_name} face detector.\nSkipping.")
            continue

        indices = result["bbox_indices"]
        top, bottom, left, right = indices
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(original_img, rect)
        fp.desired_face_width = result["detected_face_img"].shape[1]
        fp.desired_face_height = result["detected_face_img"].shape[0]
        # fp.desired_left_eye_coord = landmarks[facial_landmarks_indices["exact"]["rightmost_l_eye"]]
        aligned_and_detected = fp.align(original_img, landmarks)

        yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)
        h, w = yuv_aligned_face.shape[:2]
        target_w = (w + (number_roi - (w % number_roi))) if w % number_roi != 0 else w
        target_h = (h + (number_roi - (h % number_roi))) if h % number_roi != 0 else h
        yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h, filter_size)
        blocks = fp.divide_roi_blocks(yuv_align_padded_face, (number_roi, number_roi))
        segmented_frames.append(blocks)

        if len(segmented_frames) != n_frames_per_batch: continue

        out_map = np.empty([len(blocks) ** 2, 0], dtype=np.float)
        if inverted:
            out_map = np.empty([0, len(blocks) ** 2], dtype=np.float)
        for frame in segmented_frames:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                means.append(np.mean(frame[i][j].reshape(-1, 3), 0))
            if inverted:
                out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(3, -1), axis=0)
            else:
                out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)
        segmented_frames = []


if __name__ == "__main__":
    main()
