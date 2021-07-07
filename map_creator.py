import math
import os
from itertools import product
from random import choices

import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm

from FaceManager.helpers import pad


def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab


def apply_deeplab(deeplab, img, device):
    deeplab_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15)


def make_spatio_temporal_maps(fd, fp, video_path,
                              time_window=0.5,
                              number_roi=5,
                              filter_size=3,
                              masking_frequency=0,
                              step = 0.5,
                              inverted=True):
    '''

    :param video_path: Path to video file
    :param time_window: Time window in seconds
    :param number_roi: Number of region of interests within a frame
    :param filter_size: Padding filter size as tuple
    :param masking_frequency: How many frames get masked
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
    assert masking_frequency < n_tot_frames, "Masking frequency higher than total number of frames."

    segmented_frames = []
    final_maps = []
    frame_masked_counter = 0
    if masking_frequency == 0:
        mask_str = f"Number of masked frames: {0}"
        masking_indices = []
    else:
        masking_indices = choices(range(n_tot_frames), k=masking_frequency)
        mask_str = f"Number of masked frames: {masking_frequency}"

    print(f'''
{"*" * 25}
Processing video at: {video_path}
Time window in seconds: {time_window}
Time window in frames: {n_frames_per_batch}
{mask_str}
{"*" * 25}                
        ''')

    move_by_frames = step * fps

    for _ in tqdm(range(n_tot_frames), "Processing all frames"):
        success, img = video_capture.read()

        frame_masked_counter += 1
        original_img = img.copy()
        result = fd.detect_face(img)
        if not result:
            print(f"No face could be found by {fd.detector_name} face detector.\nSkipping.")
            continue

        top, bottom, left, right = result["bbox_indices"]
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(original_img, rect)

        # Remove eye regions
        left_eye = fp.facial_landmarks_indices["generic"]["left_eye"]
        right_eye = fp.facial_landmarks_indices["generic"]["right_eye"]

        right_hull = cv2.convexHull(landmarks[right_eye[0]:right_eye[1]])
        left_hull = cv2.convexHull(landmarks[left_eye[0]:left_eye[1]])

        cv2.drawContours(original_img, [right_hull], -1, color=(0, 0, 0), thickness=cv2.FILLED)
        cv2.drawContours(original_img, [left_hull], -1, color=(0, 0, 0), thickness=cv2.FILLED)

        aligned_and_detected = fp.align(original_img, landmarks, [left, right, top, bottom])

        # device = torch.device("cpu")
        # deeplab = make_deeplab(device)
        # img_orig = aligned_and_detected.copy()
        #
        # mask = apply_deeplab(deeplab, img_orig, device)
        #
        # masked = cv2.bitwise_and(img_orig, img_orig, mask=mask.astype(np.uint8))

        # yuv_aligned_face = aligned_and_detected
        # yuv_aligned_face = fp.rgb_to_yuv(aligned_and_detected)

        yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)
        h, w = yuv_aligned_face.shape[:2]
        target_w = (w + (number_roi - (w % number_roi))) if w % number_roi != 0 else w
        target_h = (h + (number_roi - (h % number_roi))) if h % number_roi != 0 else h
        yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h, filter_size)
        blocks = fp.divide_roi_blocks(yuv_align_padded_face, (number_roi, number_roi))

        if frame_masked_counter in masking_indices:
            blocks = np.zeros_like(blocks)

        segmented_frames.append(blocks)

    for n in tqdm(range(0, int(len(segmented_frames) - fps), int(move_by_frames)), f"Creating maps"):

        frames_subset = segmented_frames[n:n + n_frames_per_batch]

        if len(frames_subset) != n_frames_per_batch: continue

        size = [0, len(blocks) ** 2] if inverted else [len(blocks) ** 2, 0]
        out_map = np.empty(size, dtype=np.float)

        for frame in frames_subset:
            means = []
            for i, j in product(range(len(frame)), range(len(frame))):
                if inverted: i, j = j, i
                mean = np.mean(frame[i][j].reshape(-1, 3), 0)
                means.append(mean)
            out_map = np.append(out_map, np.array(means, dtype=np.float).reshape(-1, 3), axis=1)

        final_maps.append((round((n + n_frames_per_batch) / 20, 2), out_map.reshape((-1, n_frames_per_batch, 3))))

    return final_maps
