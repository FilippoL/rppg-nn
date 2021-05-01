import os
from itertools import product

import cv2
import dlib
import numpy as np

from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor


def pad(img, w, h):
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='mean'))


def main():
    fd = FaceDetectorSSD()
    fp = FaceProcessor()

    video_path = "./data/data_in/data.avi"
    assert os.path.isfile(video_path), f"{video_path} is not a valid path."

    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()

    n_frames_per_batch = 30
    n_roi = 50

    segmented_frames = []
    while success:
        success, img = video_capture.read()
        original_img = img.copy()
        result = fd.detect_face(img)
        if not result:
            print(f"No face could be found by {fd.detector_name} face detector.\nExiting.")
            return

        indices = result["bbox_indices"]
        top, bottom, left, right = indices
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(original_img, rect)
        fp.desired_face_width = result["detected_face_img"].shape[1]
        fp.desired_face_height = result["detected_face_img"].shape[0]
        aligned_and_detected = fp.align(original_img, landmarks)
        yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)
        h, w = yuv_aligned_face.shape[:2]
        target_w = (w + (n_roi - (w % n_roi))) if w % n_roi != 0 else w
        target_h = (h + (n_roi - (h % n_roi))) if h % n_roi != 0 else h
        yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h)
        blocks = fp.divide_roi_blocks(yuv_align_padded_face, (n_roi, n_roi), False)
        segmented_frames.append(blocks)

        if len(segmented_frames) != n_frames_per_batch: continue

        out_map = np.empty([len(blocks) ** 2, 0], dtype=np.float)
        for frame in segmented_frames:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                means.append(np.mean(frame[i][j].reshape(-1, 3), 0))
            out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)
        segmented_frames = []


if __name__ == "__main__":
    main()
