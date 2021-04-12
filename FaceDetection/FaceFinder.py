import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from mtcnn.mtcnn import MTCNN


class FaceDetector:
    """
    Parent class for Face Detectors methods implemented.
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        self.pad = extra_pad
        assert type(extra_pad) == tuple, "Parameter extra_pad expects a tuple (W,H)."
        self.greyscale = greyscale


class FaceDetectorMTCNN(FaceDetector):
    """
    Class wrapping the MTCNN (Multi-task Cascaded Convolutional Network)
    model provided by I. P. Centeno, the package implements the method proposed in
    the paper Zhang, K et al. (2016).
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        super().__init__(extra_pad, greyscale)
        self.detector_model = MTCNN()
        self.faces = None

    def detect_face(self, image, return_indices=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param return_indices: bool for returning indices in the order top, bottom, left, right.
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box if return_indices is True.
        """
        self.faces = self.detector_model.detect_faces(image)
        face_data = self.faces[0]
        x, y, w, h = face_data['box']
        left, right = (x - self.pad[0]), x + (w + self.pad[0])
        top, bottom = (y - self.pad[1]), y + (h + self.pad[1])
        detected_face = image[top:bottom, left:right]
        if verbose:
            print("Confidence: ", str(round(100 * face_data["confidence"], 2)) + " %")
        detected_face = detected_face if self.greyscale else detected_face[:, :, ::-1]
        return detected_face if not return_indices else [detected_face, [top, bottom, left, right]]

    def get_keypoints(self):
        points = []
        for f in self.faces:
            points.append(f["keypoints"])
        return points[0]


class FaceDetectorSSD(FaceDetector):
    """
    Class wrapping the ResNet SSD (Single Shot-Multibox Detector)
    model and its pre-trained weights provided by OpenCV community.
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        super().__init__(extra_pad, greyscale)
        config_path = os.path.join(os.path.dirname(__file__), "config", "deploy.prototxt")
        model_path = os.path.join(os.path.dirname(__file__), "model", "res10_300x300_ssd_iter_140000.caffemodel")
        self.detector_model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.target_size = (300, 300)
        self.column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]

    def detect_face(self, image, return_indices=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param return_indices: bool for returning indices in the order top, bottom, left, right.
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box if return_indices is True.
        """
        base_img = image.copy()
        original_size = base_img.shape
        image = cv2.resize(image, self.target_size)
        aspect_ratio = ((original_size[1] / self.target_size[1]), (original_size[0] / self.target_size[0]))
        # Make it greyscale and resize to 300x300
        self.detector_model.setInput(cv2.dnn.blobFromImage(image=image))
        detections = self.detector_model.forward()
        detections_df = pd.DataFrame(detections[0][0], columns=self.column_labels)
        detected_face = detections_df.sort_values("confidence", ascending=False).iloc[0]
        # Use index of downsized image and map them to orginal image coordinates so not to loose data
        left, top, right, bottom = [int(x * 300) for x in detected_face.iloc[-4:]]
        x_from, x_to = int(top * aspect_ratio[1] - self.pad[0]), int(bottom * aspect_ratio[1] + self.pad[0])
        y_from, y_to = int(left * aspect_ratio[0] - self.pad[1]), int(right * aspect_ratio[0] + self.pad[1])
        detected_face = base_img[x_from:x_to, y_from:y_to]
        mapped_coordinates = [x_from, x_to, y_from, y_to]
        if verbose:
            print("Confidence: ", str(round(100 * detected_face["confidence"], 2)) + " %")

        detected_face = detected_face if self.greyscale else detected_face[:, :, ::-1]
        return detected_face if not return_indices else [detected_face, mapped_coordinates]


if __name__ == "__main__":
    fd = FaceDetectorMTCNN()
    img = cv2.imread("../data/data_in/me_slanded.jpg")
    face, idx = fd.detect_face(img, return_indices=True)
    plt.imshow(face)
    plt.show()
    print(idx)
