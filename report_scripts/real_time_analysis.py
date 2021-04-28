import json

import cv2
import dlib
import numpy as np
from scipy.spatial import distance

from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FacePoseEstimation import FacePoseEstimator
from FaceManager.FaceProcessing import FaceProcessor
from FaceManager.FaceStabilizer import Stabilizer

fd = FaceDetectorSSD()
fp = FaceProcessor()
fpe = FacePoseEstimator()

cam = cv2.VideoCapture(0)

pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1) for _ in range(6)]

ret_val, img = cam.read()
fpe.infer_camera_internals(img.shape[:2])



with open(
        "D:\\Documents\\Programming\\Python\\thesisProject\\FaceManager\\config\\landmarks_indices.json") as json_file:
    facial_landmarks_indices = json.load(json_file)

while ret_val:
    ret_val, img = cam.read()
    result = fd.detect_face(img)

    if not result:
        print(f"No face could be found by {fd.detector_name} face detector.\nExiting.")

    indices = result["bbox_indices"]
    top, bottom, left, right = indices
    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = fp.get_face_landmarks(img, rect)

    nose_tip_position = landmarks[facial_landmarks_indices["exact"]["nose_tip"]]

    # pose = fpe.solve_pose(image_points)
    pose = fpe.solve_pose_by_68_points(landmarks.astype(np.double))

    steady_pose = []
    pose_np = np.array(pose).flatten()
    for value, ps_stb in zip(pose_np, pose_stabilizers):
        ps_stb.update([value])
        steady_pose.append(ps_stb.state[0])
    steady_pose = np.reshape(steady_pose, (-1, 3))

    fpe.draw_annotation_box(img, steady_pose[0], steady_pose[1], color=(255, 128, 128))

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    for idx in facial_landmarks_indices["exact"].values():
        cv2.circle(img, tuple(landmarks[idx]), 5, (0, 0, 255), 1)


    cv2.line(img, tuple(nose_tip_position), (img.shape[1] // 2, img.shape[0] // 2), (255, 0, 0), 2)
    dist = distance.euclidean((img.shape[1] // 2, img.shape[0] // 2),
                              nose_tip_position)


    cv2.rectangle(img, (img.shape[0] // 2 - 55, img.shape[1] - 195), (img.shape[0] // 2 + 275, img.shape[1])
                  , (0, 0, 0), -1)

    cv2.putText(img, f"Distance from center:{round(dist, 2)}", (img.shape[0] // 2 - 40, img.shape[1] - 175),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

    r_mat, _ = cv2.Rodrigues(steady_pose[0])
    p_mat = np.hstack((r_mat, np.array([[0], [0], [0]])))
    _, _, _, _, _, _, u_angle = cv2.decomposeProjectionMatrix(p_mat)
    pitch, yaw, roll = u_angle.flatten()

    blocks = fp.divide_roi_blocks(result['detected_face_img'], (2, 2), False)



    if roll != 0:
        roll = 180 - roll if roll > 0 else - (180 + roll)

    cv2.putText(img, f"{t_r}", (0, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.putText(img, f"{t_l}", (img.shape[0]//2 + 250, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.putText(img, f"{b_r}", (0, img.shape[1]//2 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.putText(img, f"{b_l}", (img.shape[0]//2 + 250, img.shape[1]//2 + 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 0), 2)

    cv2.rectangle(img, (0, 125), (175, 255), (0, 0, 0), -1)

    cv2.putText(img, f"Roll: {round(roll, 2)}", (0, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.putText(img, f"Pitch: {round(pitch, 2)}", (0, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.putText(img, f"Yaw: {round(yaw, 2)}", (0, 250), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 0, 255), 2)
    cv2.rectangle(img, (rect.left(), rect.bottom()),  (rect.right(), rect.bottom() + 25), (0, 0, 255), -1)

    cv2.putText(img, f"Size: {(result['detected_face_img'].shape[0],result['detected_face_img'].shape[1])}",
                (rect.left() + 5, rect.bottom() + 20), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

    pose = {
        "pitch": pitch / 180,
        "yaw": yaw / 180,
        "roll": roll / 180
    }

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == 27:
        break
