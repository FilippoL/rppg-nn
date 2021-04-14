import os
import timeit

import cv2
import dlib
import pandas as pd
from tqdm import tqdm

from FaceDetection.FaceManager import FaceDetectorSSD, FaceDetectorMTCNN, FaceDetectorHOG

'''
This script is used to generate a .csv files with columns [confidence:Float, face_found:Boolean, time:Float] for each
of the face detectors investigated. The input is a data_directory which can be any folder or directory containing 
pictures. The script will look recursively for each sub directory too. Supported file types are listed below.

Output is saved in data/data_out directory as {detector name}_performance_{number of images}.csv


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
out_dict_hog = {"HOG": {"confidence": [], "face_found": [], "time": []},
                "SSD": {"confidence": [], "face_found": [], "time": []},
                "MTCNN": {"confidence": [], "face_found": [], "time": []}}

data_directory = "D:\\Downloads\\ExtendedYaleB"
supported_file_extension_str = """.jpe .jp2 .webp .pbm .pgm .ppm .pxm .pnm .pfm .sr .ras .tiff 
                                .tif .exr .hdr .pic .bmp .dib .jpg .pgm .png .svg .jpeg"""
supported_file_extensions = (tuple(supported_file_extension_str.split()))

n_files = 0

for subdir, dirs, files in os.walk(data_directory):
    for file in tqdm(files):
        n_files += len(files)
        if not str(file).endswith(supported_file_extensions):
            continue
        original_img = cv2.imread(os.path.join(subdir, file))
        old_img = None
        for idx, fd in enumerate(face_detectors):
            img = original_img.copy()
            start = timeit.default_timer()
            result = fd.detect_face(img)
            detection_time = timeit.default_timer() - start
            if not result:
                indices = 0, 0, 0, 0
                confidence = 0
            else:
                indices = result["bbox_indices"]
                confidence = result["confidence"]
            top, bottom, left, right = indices
            if any(indices):
                rect = dlib.rectangle(l, t, r, b)
                landmarks = fd.get_face_landmarks(original_img, rect)
                aligned_and_detected = fd.align(original_img, landmarks)

            out_dict_hog[fd.detector_name]["confidence"].append(confidence)
            out_dict_hog[fd.detector_name]["face_found"].append(int(any(indices)))
            out_dict_hog[fd.detector_name]["time"].append(detection_time)

            if idx % 3 == 0:
                for key, val in out_dict_hog.items():
                    pd.DataFrame.from_dict(val).to_csv(f"../data/data_out/{key}_performance.csv", mode="a",
                                                       header=False, index=False)
                    out_dict_hog = {"HOG": {"confidence": [], "face_found": [], "time": []},
                                    "SSD": {"confidence": [], "face_found": [], "time": []},
                                    "MTCNN": {"confidence": [], "face_found": [], "time": []}}

for key, val in out_dict_hog.items():
    pd.DataFrame.from_dict(val).to_csv(f"../data/data_out/{key}_performance_{n_files}.csv", mode="a", header=False,
                                       index=False)
