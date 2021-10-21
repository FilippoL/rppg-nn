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
from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor


class MapCreator:
    def __init__(self, video_path=""):
        """
        :param fd: Face detector instance
        :param fp: Face processor instance
        """
        self.fd, self.fp = FaceDetectorSSD(), FaceProcessor()
        self.video_path = video_path

    def set_video(self, video_path):
        self.video_path = video_path

        # Instantiate video capture
        self.video_capture = cv2.VideoCapture(video_path)

        # Calculate number of frames in set time window
        self.fps = self.video_capture.get(cv2.cv2.CAP_PROP_FPS)
        self.n_tot_frames = int(self.video_capture.get(cv2.cv2.CAP_PROP_FRAME_COUNT))

    def make_map(self, time_window=10, number_roi=7, filter_size=3, masking_frequency=0, step=0.5, inverted=False):
        """

           :param fp: Face processor instance
           :param time_window: Time window in seconds
           :param number_roi: Number of region of interests within a frame
           :param filter_size: Padding filter size as tuple
           :param masking_frequency: How many frames get masked
           :param step: Step by which the time window is shifted
           :param inverted: Concatenate in an horizontal fashion
           :return: Ndarray containing map
           """
        # Check if video path is a valid directory
        assert os.path.isfile(self.video_path), f"Provided path {self.video_path} is not valid."

        assert masking_frequency < self.n_tot_frames, f"Masking frequency higher than total number of frames ({self.n_tot_frames})."

        n_frames_per_batch = math.ceil(time_window * self.fps)  # Round up
        segmented_frames, final_maps, blocks = [], [], []
        frame_masked_counter = 0
        masking_indices = [] if masking_frequency == 0 else choices(range(self.n_tot_frames), k=masking_frequency)
        move_by_frames = step * self.fps

        print(f'{"*" * 25}')
        print(f'Processing video at: {self.video_path}')
        print(f'Time window in seconds: {time_window}')
        print(f'Time window in frames: {n_frames_per_batch}')
        print(f'Number of masked frames: {masking_frequency}')
        print(f'{"*" * 25}')

        for _ in tqdm(range(self.n_tot_frames), "Processing all frames"):
            success, img = self.video_capture.read()

            frame_masked_counter += 1
            original_img = img.copy()
            result = self.fd.detect_face(img)

            if not result: continue

            top, bottom, left, right = result["bbox_indices"]
            rect = dlib.rectangle(left, top, right, bottom)
            landmarks = self.fp.get_face_landmarks(original_img, rect)

            original_img = self.fp.remove_eyes(original_img, landmarks)

            aligned_and_detected = self.fp.align(original_img, landmarks, [left, right, top, bottom])

            yuv_aligned_face = cv2.cvtColor(aligned_and_detected, cv2.COLOR_BGR2YUV)

            if frame_masked_counter in masking_indices:
                blocks = np.zeros_like(blocks)
            else:
                blocks = self.pad_and_split_in_ROI(yuv_aligned_face, number_roi, filter_size)

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

            final_maps.append((round((n + n_frames_per_batch) / self.fps, 2),
                               out_map.reshape((-1, n_frames_per_batch, 3))))

        return final_maps

    def pad_and_split_in_ROI(self, yuv_aligned_face, number_roi, filter_size):
        h, w = yuv_aligned_face.shape[:2]
        target_w = (w + (number_roi - (w % number_roi))) if w % number_roi != 0 else w
        target_h = (h + (number_roi - (h % number_roi))) if h % number_roi != 0 else h
        yuv_align_padded_face = pad(yuv_aligned_face, target_w, target_h, filter_size)
        blocks = self.fp.divide_roi_blocks(yuv_align_padded_face, (number_roi, number_roi))
        return blocks

    def make_deeplab(self, device):
        deeplab = deeplabv3_resnet101(pretrained=True).to(device)
        deeplab.eval()
        return deeplab

    def apply_deeplab(self, deeplab, img, device):
        deeplab_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = deeplab_preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = deeplab(input_batch.to(device))['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        return output_predictions == 15
