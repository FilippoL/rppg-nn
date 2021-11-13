import os
import sys

import cv2
import dlib
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN

from .FaceProcessing import FaceProcessor


# TODO[x]: Make it for video in real time
# TODO[]: Look at the metadata of each prediction return value
# TODO[x]: Clean up

class FaceDetector:
    """
    Parent class for Face Detectors implemented.
    """

    def __init__(self, extra_pad=(0, 0)):
        self.pad = extra_pad
        self._CONFIDENCE_STR = "Confidence"
        assert type(extra_pad) == tuple, "Parameter extra_pad expects a tuple (W,H)."

    @staticmethod
    def map_range(origin_range, target_range, value):
        (a1, a2), (b1, b2) = origin_range, target_range
        return b1 + ((value - a1) * (b2 - b1) / (a2 - a1))


class FaceDetectorMTCNN(FaceDetector):
    """
    Class wrapping the MTCNN (Multi-task Cascaded Convolutional Network)
    model provided by I. P. Centeno, the package implements the method proposed in
    the paper by Zhang, K et al. (2016).
    """

    def __init__(self, extra_pad=(0, 0)):
        super().__init__(extra_pad)
        self.detector_model = MTCNN()
        self.detector_name = "MTCNN"

    def detect_face(self, image, greyscale_out=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param greyscale_out: bool for returning greyscale image
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box if return_indices is True.
        """

        detected_face_data = self.detector_model.detect_faces(image)
        if len(detected_face_data) == 0:
            if verbose: print("No face found by MTCNN.")
            return False
        detected_face_data = detected_face_data[0]
        x, y, w, h = detected_face_data['box']
        left, right = (x - self.pad[0]), x + (w + self.pad[0])
        top, bottom = (y - self.pad[1]), y + (h + self.pad[1])
        detected_face_img = image[top:bottom, left:right]
        confidence = round(detected_face_data["confidence"], 2)
        if verbose:
            print(self._CONFIDENCE_STR + " of " + self.detector_name + ":",
                  str(100 * confidence) + " %")
        return {"detected_face_img": detected_face_img[:, :, ::-1] if greyscale_out else detected_face_img,
                "bbox_indices": [top, bottom, left, right], "confidence": confidence}


class FaceDetectorSSD(FaceDetector):
    """
    Class wrapping the ResNet SSD (Single Shot-Multibox Detector)
    model and its pre-trained weights provided by OpenCV community.
    """

    def __init__(self, extra_pad=(0, 0)):
        super().__init__(extra_pad)
        config_path = os.path.join(os.path.dirname(__file__), "config", "deploy.prototxt")
        model_path = os.path.join(os.path.dirname(__file__), "models", "res10_300x300_ssd_iter_140000.caffemodel")
        self.detector_model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.target_size = (300, 300)
        self.column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        self.detector_name = "SSD"

    def detect_face(self, image, greyscale_out=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param greyscale_out: bool for returning greyscale image
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box when return_indices is True.
        """
        base_img = image.copy()
        original_size = base_img.shape
        image = cv2.resize(image, self.target_size)
        aspect_ratio = ((original_size[1] / self.target_size[1]), (original_size[0] / self.target_size[0]))
        self.detector_model.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0)))
        detections = self.detector_model.forward()

        detections_df = pd.DataFrame(detections[0][0], columns=self.column_labels)
        detected_face_data = detections_df.sort_values("confidence", ascending=False).iloc[0]

        if not np.any(detections) or detected_face_data["confidence"] < 0.8:
            if verbose: print("No face found by SSD.")
            return {"bbox_indices": (0, 0, 0, 0), "confidence": detected_face_data["confidence"]}

        # Use index of downsized image and map them to original image coordinates so not to loose data
        left, top, right, bottom = [int(x * 300) for x in detected_face_data.iloc[-4:]]
        x_from, x_to = int(top * aspect_ratio[1] - self.pad[0]), int(bottom * aspect_ratio[1] + self.pad[0])
        y_from, y_to = int(left * aspect_ratio[0] - self.pad[1]), int(right * aspect_ratio[0] + self.pad[1])
        detected_face_img = base_img[x_from:x_to, y_from:y_to]
        mapped_coordinates = [x_from, x_to, y_from, y_to]
        confidence = detected_face_data["confidence"]
        if verbose:
            print(self._CONFIDENCE_STR + " of " + self.detector_name + ":",
                  str(round(100 * confidence, 2)) + " %")
        return {"detected_face_img": detected_face_img[:, :, ::-1] if greyscale_out else detected_face_img,
                "bbox_indices": mapped_coordinates, "confidence": confidence}


class FaceDetectorHOG(FaceDetector):
    """
        Class wrapping the HOG (histogram of oriented gradients)
        model provided by the dlib library, the python package built on the top of this
        implements the method proposed in the paper by Dalal et al. (2005).
    """

    def __init__(self, extra_pad=(0, 0)):
        super().__init__(extra_pad)
        self.detector_model = dlib.get_frontal_face_detector()
        self.detector_name = "HOG"
        self.number_of_upsampling = 1

    def detect_face(self, image, greyscale_out=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param greyscale_out: bool for returning greyscale image
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box if return_indices is True.
        """
        detected_faces, scores, _ = self.detector_model.run(image, self.number_of_upsampling)
        if len(detected_faces) == 0:
            if verbose: print("No face found by HOG.")
            return False
        detected_face = detected_faces[0]
        confidence = round(self.map_range((0, 3.5), (0, 1), scores[0]), 2)
        css = detected_face.top(), detected_face.right(), detected_face.bottom(), detected_face.left()
        y, w, h, x = max(css[0], 0), min(css[1], image.shape[1]), min(css[2], image.shape[0]), max(css[3], 0)
        left, right = (x - self.pad[0]), w + self.pad[0]
        top, bottom = (y - self.pad[1]), h + self.pad[1]
        detected_face_img = image[top:bottom, left:right]
        if verbose:
            print(self._CONFIDENCE_STR + " of " + self.detector_name + ":" + " ~",
                  str(confidence * 100) + "%")
        return {"detected_face_img": detected_face_img[:, :, ::-1] if greyscale_out else detected_face_img,
                "bbox_indices": [top, bottom, left, right], "confidence": confidence}


def main(args):
    if not args:
        print("Please provide path to supported image as argument.")
        return False
    print("No detector specified, defaulting to SSD.")
    image_path = args[0]
    assert os.path.isfile(image_path), f"{image_path} is not a valid path."
    original_img = cv2.imread(image_path)
    img = original_img.copy()
    fd = FaceDetectorSSD()
    fp = FaceProcessor()
    result = fd.detect_face(img, verbose=True)
    if not result:
        print(f"No face could be found by {fd.detector_name} face detector.\nExiting.")
        return
    indices = result["bbox_indices"]
    top, bottom, left, right = indices
    rect = dlib.rectangle(left, top, right, bottom)
    landmarks = fp.get_face_landmarks(original_img, rect)
    aligned_and_detected = fp.align(original_img, landmarks)
    path_to = f"{os.path.dirname(os.path.abspath(image_path))}\\aligned.jpg"
    print(f"Saved image to {path_to}")
    cv2.imwrite(path_to, aligned_and_detected)


if __name__ == "__main__":
    main(sys.argv[1:])
