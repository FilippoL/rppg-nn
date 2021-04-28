"""
Data studied:
    - Distance of tip of the nose landmark from center of the screen
    - X,Y coordinates of middle of bounding box
    - Roll, pitch and yaw of the head
    - In how many frames no detection happened
    - Size of the bounding box detected around face
"""

import _pickle as cPickle
import json
import os

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FacePoseEstimation import FacePoseEstimator
from FaceManager.FaceProcessing import FaceProcessor
from FaceManager.FaceStabilizer import Stabilizer

fd = FaceDetectorSSD()
fp = FaceProcessor()
fpe = FacePoseEstimator()

pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1) for _ in range(6)]

with open("../FaceManager/config/landmarks_indices.json") as json_file:
    facial_landmarks_indices = json.load(json_file)

out_dictionary = {"luminance": {"top_right": [], "top_left": [], "bottom_right": [], "bottom_left": []},
                  "distance_nose_from_center": [],
                  "head_orientation": {"pitch": [], "yaw": [], "roll": []},
                  "detected_bbox_size": [],
                  "not_detected": 0}

final_dictionary = {"luminance": {"top_right": [], "top_left": [], "bottom_right": [], "bottom_left": []},
                  "distance_nose_from_center": [],
                  "head_orientation": {"pitch": [], "yaw": [], "roll": []},
                  "detected_bbox_size": [],
                  "not_detected": 0}

videos_per_folder = 4
dataset_name = "statistics.pickle"
aggregated_dataset_name = "aggregated_statistics.pickle"
data_directory = "H:\\DatasetsThesis\\COHFACE"

not_detected = 0

for subdir, dirs, files in os.walk(data_directory):
    for file in files:
        if not file.endswith("avi"):
            continue

        video_capture = cv2.VideoCapture(os.path.join(subdir, file))
        valid, frame = video_capture.read()

        n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fpe.infer_camera_internals(frame.shape[:2])
        for f in tqdm(range(n_frames), desc="Frames"):
            valid, frame = video_capture.read()
            if type(frame) != np.ndarray or not valid:
                not_detected += 1
                continue

            result = fd.detect_face(frame)

            if not result:
                not_detected += 1
                continue

            indices = result["bbox_indices"]
            top, bottom, left, right = indices
            rect = dlib.rectangle(left, top, right, bottom)
            landmarks = fp.get_face_landmarks(frame, rect)

            nose_tip_position = landmarks[facial_landmarks_indices["exact"]["nose_tip"]]

            pose = fpe.solve_pose_by_68_points(landmarks.astype(np.double))

            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            r_mat, _ = cv2.Rodrigues(steady_pose[0])
            p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
            _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
            pitch, yaw, roll = u_angle.flatten()

            if roll != 0:
                roll = 180 - roll if roll > 0 else -(180 + roll)

            blocks = fp.divide_roi_blocks(result['detected_face_img'], (2, 2), False)

            dist = distance.euclidean((frame.shape[1] // 2, frame.shape[0] // 2),
                                      nose_tip_position)

            t_r = np.sum(np.divide(blocks[0][0], 765)) // 3
            t_l = np.sum(np.divide(blocks[0][1], 765)) // 3
            b_r = np.sum(np.divide(blocks[1][0], 765)) // 3
            b_l = np.sum(np.divide(blocks[1][1], 765)) // 3

            out_dictionary["luminance"]["top_right"].append(t_r)
            out_dictionary["luminance"]["top_left"].append(t_l)
            out_dictionary["luminance"]["bottom_right"].append(b_r)
            out_dictionary["luminance"]["bottom_left"].append(b_l)

            out_dictionary["distance_nose_from_center"].append(dist)

            out_dictionary["head_orientation"]["pitch"].append(pitch)
            out_dictionary["head_orientation"]["yaw"].append(yaw)
            out_dictionary["head_orientation"]["roll"].append(roll)

            out_dictionary["detected_bbox_size"].append(result["detected_face_img"].shape)

        out_dictionary["not_detected"] = not_detected
        full_path = os.path.join(subdir, dataset_name)
        with open(full_path, "wb") as output_file:
            cPickle.dump(out_dictionary, output_file)
        output_file.close()
        not_detected = 0

        final_dictionary["luminance"]["top_right"].extend(out_dictionary["luminance"]["top_right"])
        final_dictionary["luminance"]["top_left"].extend(out_dictionary["luminance"]["top_left"])
        final_dictionary["luminance"]["bottom_right"].extend(out_dictionary["luminance"]["bottom_right"])
        final_dictionary["luminance"]["bottom_left"].extend(out_dictionary["luminance"]["bottom_left"])
        final_dictionary["distance_nose_from_center"].extend(out_dictionary["distance_nose_from_center"])
        final_dictionary["head_orientation"]["pitch"].extend(out_dictionary["head_orientation"]["pitch"])
        final_dictionary["head_orientation"]["yaw"].extend(out_dictionary["head_orientation"]["yaw"])
        final_dictionary["head_orientation"]["roll"].extend(out_dictionary["head_orientation"]["roll"])
        final_dictionary["detected_bbox_size"].extend(out_dictionary["detected_bbox_size"])
        final_dictionary["not_detected"] += out_dictionary["not_detected"]


        out_dictionary = {"luminance": {"top_right": [], "top_left": [], "bottom_right": [], "bottom_left": []},
                          "distance_nose_from_center": [],
                          "head_orientation": {"pitch": [], "yaw": [], "roll": []},
                          "detected_bbox_size": [],
                          "not_detected": 0}

        print(f"Created dataset at: {full_path}")

full_path = os.path.join(data_directory, aggregated_dataset_name)
with open(full_path, "wb") as output_file:
    cPickle.dump(final_dictionary, output_file)
output_file.close()