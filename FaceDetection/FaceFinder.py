import os
import timeit

import cv2
import dlib
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN


class FaceDetector:
    """
    Parent class for Face Detectors methods implemented.
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        self.pad = extra_pad
        self.CONFIDENCE_STR = "Confidence"
        assert type(extra_pad) == tuple, "Parameter extra_pad expects a tuple (W,H)."
        self.use_gtx_model = False
        self.greyscale = greyscale
        _gtx_model = "shape_predictor_68_face_landmarks_GTX.dat"
        _cpu_model = "shape_predictor_68_face_landmarks.dat"
        _predictor_name = _gtx_model if self.use_gtx_model else _cpu_model
        _predictor_path = os.path.join(os.path.dirname(__file__), "models", _predictor_name)
        self.predictor_model = dlib.shape_predictor(_predictor_path)

    def get_face_landmarks(self, image, rectangle):
        shape = self.predictor_model(image, rectangle)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        return landmarks

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        coordinates = np.zeros((68, 2), dtype=dtype)
        for n in range(0, 68):
            coordinates[n] = (shape.part(n).x, shape.part(n).y)
        return coordinates

    @staticmethod
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h


class FaceDetectorMTCNN(FaceDetector):
    """
    Class wrapping the MTCNN (Multi-task Cascaded Convolutional Network)
    model provided by I. P. Centeno, the package implements the method proposed in
    the paper by Zhang, K et al. (2016).
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        super().__init__(extra_pad, greyscale)
        self.detector_model = MTCNN()
        self.faces = None
        self.detector_name = "MTCNN"

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
        if len(self.faces) == 0:
            print("No face found by MTCNN.")
            return False
        face_data = self.faces[0]
        x, y, w, h = face_data['box']
        left, right = (x - self.pad[0]), x + (w + self.pad[0])
        top, bottom = (y - self.pad[1]), y + (h + self.pad[1])
        detected_face = image[top:bottom, left:right]
        if verbose:
            print(self.CONFIDENCE_STR + " of " + self.detector_name + ":",
                  str(round(100 * face_data["confidence"], 2)) + " %")
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
        model_path = os.path.join(os.path.dirname(__file__), "models", "res10_300x300_ssd_iter_140000.caffemodel")
        self.detector_model = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.target_size = (300, 300)
        self.column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
        self.detector_name = "SSD"

    def detect_face(self, image, return_rectangle=False, verbose=False):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param verbose: bool for printing confidence score.
        :param return_rectangle: bool for returning indices in the order top, bottom, left, right.
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box when return_indices is True.
        """
        base_img = image.copy()
        original_size = base_img.shape
        image = cv2.resize(image, self.target_size)
        aspect_ratio = ((original_size[1] / self.target_size[1]), (original_size[0] / self.target_size[0]))
        # Make it greyscale and resize to 300x300
        self.detector_model.setInput(cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 117.0, 123.0)))
        detections = self.detector_model.forward()
        detections_df = pd.DataFrame(detections[0][0], columns=self.column_labels)
        detected_face_data = detections_df.sort_values("confidence", ascending=False).iloc[0]
        # Use index of downsized image and map them to original image coordinates so not to loose data
        left, top, right, bottom = [int(x * 300) for x in detected_face_data.iloc[-4:]]
        x_from, x_to = int(top * aspect_ratio[1] - self.pad[0]), int(bottom * aspect_ratio[1] + self.pad[0])
        y_from, y_to = int(left * aspect_ratio[0] - self.pad[1]), int(right * aspect_ratio[0] + self.pad[1])
        detected_face = base_img[x_from:x_to, y_from:y_to]
        mapped_coordinates = [x_from, x_to, y_from, y_to]
        if verbose:
            print(self.CONFIDENCE_STR + " of " + self.detector_name + ":",
                  str(round(100 * detected_face_data["confidence"], 2)) + " %")

        detected_face = detected_face[:, :, ::-1] if self.greyscale else detected_face
        return detected_face if not return_rectangle else [detected_face, mapped_coordinates]


class FaceDetectorHOG(FaceDetector):
    """
        Class wrapping the HOG (histogram of oriented gradients)
        model provided by the dlib library, the python package built on the top of this
        implements the method proposed in the paper by Dalal et al. (2005).
    """

    def __init__(self, extra_pad=(0, 0), greyscale=False):
        super().__init__(extra_pad, greyscale)
        self.detector_model = dlib.get_frontal_face_detector()
        self.faces = None
        self.detector_name = "HOG"

    def detect_face(self, image, return_indices=False, verbose=False, n_upsample=1):
        """
        This function takes a picture of an image and returns the region where face is detected.
        :param n_upsample: number of times to upsample the image.
        :param verbose: bool for printing confidence score.
        :param return_indices: bool for returning indices in the order top, bottom, left, right.
        :param image:the image provided as a 3 dimensional array.
        :return: Face region of the provided image as a 3D numpy array
                 and indices of bounding box if return_indices is True.
        """
        self.faces, scores, _ = self.detector_model.run(image, n_upsample)
        face = self.faces[0]
        score = scores[0]
        css = face.top(), face.right(), face.bottom(), face.left()
        y, w, h, x = max(css[0], 0), min(css[1], image.shape[1]), min(css[2], image.shape[0]), max(css[3], 0)
        left, right = (x - self.pad[0]), w + self.pad[0]
        top, bottom = (y - self.pad[1]), h + self.pad[1]
        detected_face = image[top:bottom, left:right]
        if verbose:
            print(self.CONFIDENCE_STR + " of " + self.detector_name + ":" + " ~",
                  str(round(self.map_range((0, 3.5), (0, 1), score), 2)) + "%")
        detected_face = detected_face[:, :, ::-1] if self.greyscale else detected_face
        return detected_face if not return_indices else [detected_face, [top, bottom, left, right]]

    @staticmethod
    def map_range(origin_range, target_range, value):
        (a1, a2), (b1, b2) = origin_range, target_range
        return b1 + ((value - a1) * (b2 - b1) / (a2 - a1))


if __name__ == "__main__":
    face_detectors = [FaceDetectorSSD(), FaceDetectorHOG(), FaceDetectorMTCNN()]
    face_detectors_labels = ["SSD", "HOG", "MTCNN"]
    collage_pieces = []
    for i in range(3):
        original_img = cv2.imread(f"../data/data_in/me_{i}.jpg")
        old_img = None
        for idx, fd in enumerate(face_detectors):
            print("-" * 35)
            img = original_img.copy()
            start = timeit.default_timer()
            result = fd.detect_face(img, True, True)
            print(f"Time taken by {fd.detector_name}:", timeit.default_timer() - start)
            if not result:
                face_img = img
                indices = 0, 0, 0, 0
            else:
                face_img, indices = result
            t, b, l, r = indices  # top, bottom, left, right
            rect = dlib.rectangle(l, t, r, b)
            fd.get_face_landmarks(face_img, rect)
            cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(img, face_detectors_labels[idx], (50, 50), cv2.FONT_HERSHEY_DUPLEX, 2,
                        (0, 0, 255), 4)
            img = cv2.resize(img, (original_img.shape[1] // 3, original_img.shape[0] // 3))
            if idx == 0:
                old_img = img
            else:
                old_img = np.hstack([old_img, img])

        collage_pieces.append(old_img)
    print("-" * 35)
    print("\t\tDONE")
    out_img = np.vstack([c for c in collage_pieces])
    cv2.imwrite(f"final_comparison.jpg", out_img)
