import math
import os
import random
from itertools import product

import cv2
import dlib
import numpy as np
from tqdm import tqdm

from FaceManager.helpers import pad


def make_spatio_temporal_maps(fd, fp, video_path,
                              time_window=0.5,
                              number_roi=5,
                              filter_size=3,
                              masking_frequency=0.1,
                              inverted=True):
    '''

    :param video_path: Path to video file
    :param time_window: Time window in seconds
    :param number_roi: Number of region of interests within a frame
    :param filter_size: Padding filter size as tuple
    :param masking_frequency: Frequency with which apply a mask (0 to 1)
    :param inverted: Concatenate in an horizontal fashion
    :return: Ndarray containing map
    '''

    # Check if video path is a valid directory
    assert os.path.isfile(video_path), f"{video_path} is not a valid path."

    # Instantiate video capture
    video_capture = cv2.VideoCapture(video_path)

    # Calculate number of frames in set time window
    fps = video_capture.get(cv2.cv2.CAP_PROP_FPS)
    n_tot_frames = int(video_capture.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
    n_frames_per_batch = math.ceil(time_window * fps)  # Round up

    segmented_frames = []
    final_maps = []

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

        size = [0, len(blocks) ** 2] if inverted else [len(blocks) ** 2, 0]
        out_map = np.empty(size, dtype=np.float)

        for frame in segmented_frames:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                if inverted: i, j = j, i
                mean = np.mean(frame[i][j].reshape(-1, 3), 0)
                means.append(mean)
            out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)

        n_masked_frames = np.ceil(out_map.shape[1] * masking_frequency)
        mask = np.zeros_like(out_map[0, 0])
        rnd_indices = random.sample(range(out_map.shape[1]), int(n_masked_frames))

        for idx in rnd_indices:
            out_map[:, idx] = mask

        out_map = out_map.reshape((-1, n_frames_per_batch, 3))
        segmented_frames = []
        final_maps.append(out_map)

    return final_maps
