import os
import timeit

import cv2
import dlib
import numpy as np

from FaceDetection.FaceManager import FaceDetectorSSD, FaceDetectorMTCNN, FaceDetectorHOG
from FaceDetection.helpers import FACIAL_LANDMARKS_IDXS

'''
This script is used to generate stacked pictures for visual comparison of multiple face detectors, additionally 
it prints performances on the console but does not save them to file. 
The output is a stack of all the images passed to the script with the name of the model printed on them and the 
detected face is outlined by a green triangle. Additionally, it creates another file with the faces aligned and
the 68 face landmarks printed on them.
Given that for N images the output is a NxN matrix of images, suggested max number is 3.

Windows bitmaps - .bmp, .dib (always supported)
JPEG files - .jpeg, .jpg, *.jpe (might fail, for more details see bit.ly/3uTzATB)
JPEG 2000 files - *.jp2 (might fail, for more details see bit.ly/3uTzATB)
Portable Network Graphics - *.png (might fail, for more details see bit.ly/3uTzATB)
WebP - *.webp (might fail, for more details see bit.ly/3uTzATB)
Portable image format - .pbm, .pgm, .ppm .pxm, *.pnm (always supported)
PFM files - *.pfm (might fail, for more details see bit.ly/3uTzATB)
Sun rasters - .sr, .ras (always supported)
TIFF files - .tiff, .tif (might fail, for more details see bit.ly/3uTzATB)
OpenEXR Image files - *.exr (might fail, for more details see bit.ly/3uTzATB)
Radiance HDR - .hdr, .pic (always supported)
'''

face_detectors = [FaceDetectorSSD(), FaceDetectorHOG(), FaceDetectorMTCNN()]
detection_comp_stack = []
alignment_stack = []

data_directory = "../data/data_in/sample_miniset/"
supported_file_extension_str = """.jpe .jp2 .webp .pbm .pgm .ppm .pxm .pnm .pfm .sr .ras .tiff 
                                .tif .exr .hdr .pic .bmp .dib .jpg .pgm .png .svg .jpeg"""
supported_file_extensions = (tuple(supported_file_extension_str.split()))

out_img = None


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220), (0, 0, 255)]
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        if name == "jaw":
            for point in range(1, len(pts)):
                point_a = tuple(pts[point - 1])
                point_b = tuple(pts[point])
                cv2.line(overlay, point_a, point_b, colors[i], 2)
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


for subdir, dirs, files in os.walk(data_directory):
    for file_idx, file in enumerate(files):
        if not str(file).endswith(supported_file_extensions):
            continue
        original_img = cv2.imread(os.path.join(subdir, file))
        old_img = None
        for idx, fd in enumerate(face_detectors):
            print("-" * 35)
            img = original_img.copy()
            start = timeit.default_timer()
            result = fd.detect_face(img, verbose=True)
            detection_time = timeit.default_timer() - start
            print(f"Time taken by {fd.detector_name}:", detection_time)
            if not result:
                indices = 0, 0, 0, 0
                confidence = 0
            else:
                indices = result["bbox_indices"]
                confidence = result["confidence"]
            top, bottom, left, right = indices

            rect = dlib.rectangle(left, top, right, bottom)
            landmarks = fd.get_face_landmarks(original_img, rect)

            if any(indices):
                start = timeit.default_timer()
                fd.desired_face_width = result["detected_face_img"].shape[1]
                fd.desired_face_height = result["detected_face_img"].shape[0]
                aligned_and_detected = fd.align(original_img, landmarks)
                print(f"Time taken for alignment:", timeit.default_timer() - start)
                not_aligned_image = result["detected_face_img"]
                comparison = np.hstack([not_aligned_image, aligned_and_detected])
                cv2.imwrite(f"../data/data_out/comparing/{fd.detector_name}_alignment_{file_idx}.jpg", comparison)

            img = visualize_facial_landmarks(original_img, landmarks)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, fd.detector_name, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2,
                        (0, 0, 255), 4)
            img = cv2.resize(img, (original_img.shape[1] // 3, original_img.shape[0] // 3))
            if idx == 0:
                old_img = img
            else:
                old_img = np.hstack([old_img, img])

        detection_comp_stack.append(old_img)
        print("-" * 35)
        out_img = np.vstack([c for c in detection_comp_stack])
    cv2.imwrite(f"../data/data_out/comparing/final_comparison.jpg", out_img)
