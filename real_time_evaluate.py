import math
from itertools import product

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import dlib
import numpy as np
import timeit
from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor
from FaceManager.helpers import pad

cam = cv2.VideoCapture(0)
fd = FaceDetectorSSD()  # Face Detector Instance
fp = FaceProcessor()  # Face Processor Instance
model = load_model(
    r"C:\Users\pippo\Documents\Programming\Python\rppg-nnet\model\rppg-nnet.h5")

fps = cam.get(cv2.cv2.CAP_PROP_FPS)
n_frames_per_batch = math.ceil(10 * fps)
number_roi = 7  # Number of region of interests within a frame
filter_size = 3  # Padding filter size

frames_subset = []
prediction = [[0]]
model.predict(np.zeros((49, 200, 3))[None, :, :, :])

while True:
    start = timeit.default_timer()
    ret_val, original_img = cam.read()
    img = original_img.copy()
    result = fd.detect_face(img)
    try:
        top, bottom, left, right = result["bbox_indices"]
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(img, rect)

        # Remove eye regions
        left_eye = fp.facial_landmarks_indices["generic"]["left_eye"]
        right_eye = fp.facial_landmarks_indices["generic"]["right_eye"]

        right_hull = cv2.convexHull(landmarks[right_eye[0]:right_eye[1]])
        left_hull = cv2.convexHull(landmarks[left_eye[0]:left_eye[1]])

        cv2.drawContours(img, [right_hull], -1, color=(0, 0, 0), thickness=cv2.FILLED)
        cv2.drawContours(img, [left_hull], -1, color=(0, 0, 0), thickness=cv2.FILLED)

        aligned_and_detected = fp.align(img, landmarks, [left, right, top, bottom])

        yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)
    except Exception:
        yuv_aligned_face = np.zeros((140, 140, 3))

    h, w = yuv_aligned_face.shape[:2]
    target_w = (w + (number_roi - (w % number_roi))) if w % number_roi != 0 else w
    target_h = (h + (number_roi - (h % number_roi))) if h % number_roi != 0 else h
    yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h, filter_size)
    blocks = fp.divide_roi_blocks(yuv_align_padded_face, (number_roi, number_roi))

    frames_subset.append(blocks)

    if len(frames_subset) == 200:
        size = [len(blocks) ** 2, 0]
        out_map = np.empty(size, dtype=np.float)

        for frame in frames_subset:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                mean = np.mean(frame[i][j].reshape(-1, 3), 0)
                means.append(mean)
            out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)
        prediction = model.predict(out_map.reshape((-1, 200, 3))[None, :, :, :])
        frames_subset = []

    if prediction[0][0] > 35:
        cv2.putText(original_img, f"BPM: {np.floor(prediction)[0][0]}", (original_img.shape[0] // 2 - 40,
                                                                         original_img.shape[1] - 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(original_img, f"BPM: Invalid", (original_img.shape[0] // 2 - 40,
                                                                         original_img.shape[1] - 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('RealTimeTest', original_img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
