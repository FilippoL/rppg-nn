import os

import cv2
import dlib

from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor


def main():
    fd = FaceDetectorSSD()
    fp = FaceProcessor()

    video_path = "./data/data_in/data.avi"
    assert os.path.isfile(video_path), f"{video_path} is not a valid path."

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    n_frames_per_batch = 30
    segmented_frames = []

    while success:
        success, img = vidcap.read()
        original_img = img.copy()
        result = fd.detect_face(img)
        if not result:
            print(f"No face could be found by {fd.detector_name} face detector.\nExiting.")
            return

        indices = result["bbox_indices"]
        top, bottom, left, right = indices
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = fp.get_face_landmarks(original_img, rect)
        aligned_and_detected = fp.align(original_img, landmarks)
        yuv_aligned_face = fp.project_to_yuv(aligned_and_detected)
        blocks = fp.divide_roi_blocks(yuv_aligned_face)

        segmented_frames.append(blocks)
        if len(segmented_frames) == n_frames_per_batch:
            for frame in segmented_frames:
                for blocks in frame:
                    for block in blocks:
                        for cell in block:
                            for pixel in cell:
                                for color in pixel:
                                    print(color)


if __name__ == "__main__":
    main()
