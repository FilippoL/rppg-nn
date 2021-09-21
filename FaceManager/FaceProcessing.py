import json
import os
from math import cos, sin
import cv2
import dlib
import numpy as np
import tensorflow as tf


class FaceProcessor:
    """
    Class that will take care of the extra steps for our processing pipeline beside the detection.
    """

    def __init__(self, use_gtx=False):
        self.use_gtx_model = use_gtx
        _gtx_model = "shape_predictor_68_face_landmarks_GTX.dat"
        _cpu_model = "shape_predictor_68_face_landmarks.dat"
        _predictor_name = _gtx_model if self.use_gtx_model else _cpu_model
        _predictor_path = os.path.join(os.path.dirname(__file__), "models", _predictor_name)
        self.predictor_model = dlib.shape_predictor(_predictor_path)

        _facial_landmarks_indices_path = os.path.join(os.path.dirname(__file__), "config", "landmarks_indices.json")
        with open(_facial_landmarks_indices_path) as json_file:
            self.facial_landmarks_indices = json.load(json_file)

    def get_face_landmarks(self, image, rectangle):
        """
        Image that computes the landmarks using the model chosen in the constructor.
        :param image: The image for which the landmarks are to be computed.
        :param rectangle: The dlib rectangle made out of the detected face indices.
        :return: An array of facial landmarks detected.
        """
        shape = self.predictor_model(image, rectangle)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        return np.squeeze(np.asarray(landmarks))

    def align(self, image, landmarks, points):
        """
        This function takes an image and the landmarks positions and aligns the face in the image accordingly.
        :param image: The face image to be aligned.
        :param landmarks: A dictionary of landmarks with the part of the bodies marked accordingly.
        :return: The aligned image.
        """
        # This function is inspired by Adrian Rosebrock post on pyImageSearch
        (left_eye_from, left_eye_to) = self.facial_landmarks_indices["generic"]["left_eye"]
        (right_eye_from, right_eye_to) = self.facial_landmarks_indices["generic"]["right_eye"]

        left_eye_points = landmarks[left_eye_from:left_eye_to]
        right_eye_points = landmarks[right_eye_from:right_eye_to]

        left_eye_center = left_eye_points.mean(axis=0).astype("int")
        right_eye_center = right_eye_points.mean(axis=0).astype("int")

        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]

        angle = np.degrees(np.arctan2(dy, dx)) - 180

        eyes_center = (int((left_eye_center[0] + right_eye_center[0]) // 2),
                       int((left_eye_center[1] + right_eye_center[1]) // 2))

        M = cv2.getRotationMatrix2D((eyes_center), angle, scale=1.0)


        dim = (image.shape[1], image.shape[0])
        output = cv2.warpAffine(image, M, dim, flags=cv2.INTER_CUBIC)

        return output[points[2]:points[3], points[0]:points[1]]

    @staticmethod
    def divide_roi_blocks(img, n_blocks=(5, 5)):
        """
        My implementation of a function that splits a given image in NxM blocks.
        Faster than any package that achieves the same result.
        :param: image: The image to be processed.
        :param n_blocks: Number of blocks (horizontal, vertical)
        :return: An ndarray with shape (n_blocks[0], n_blocks[1]) containing all the blocks
        """
        horizontal_blocks, vertical_blocks = n_blocks
        horizontal = np.array_split(img, horizontal_blocks)
        splitted_img = [np.array_split(block, vertical_blocks, axis=1) for block in horizontal]
        return splitted_img

    @staticmethod
    def rgb_to_yuv(img):
        """
        Takes an image and projects it into YUV color space.
        :param: image: The image to be processed.
        :return: The projected image as a numpy array.
        """
        coefficient = np.array([[0.299, 0.587, 0.114], [0.169, 0.331, 0.5], [0.5, 0.419, 0.081]])
        result = np.add(img.dot(coefficient), [0, 128, 128])
        return result

    def project_to_yuv(self, img):
        """
        Takes an image and projects it into YUV color space.
        :param: image: The image to be processed.
        :return: The projected image as a numpy array.
        """
        normalised_face_data = np.divide(img, 255)
        tensor_yuv = tf.image.rgb_to_yuv(normalised_face_data)
        return np.multiply(tensor_yuv.numpy(), 255)


if __name__ == "__main__":
    image_path = "D:\\Documents\\Programming\\Python\\thesisProject\\data\\data_in\\sample_set\\aligned.jpg"
    assert os.path.isfile(image_path), f"{image_path} is not a valid path."
    aligned_face = cv2.imread(image_path)
    fp = FaceProcessor(aligned_face)
