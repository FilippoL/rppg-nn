import os

import cv2
import numpy as np
import tensorflow as tf


# TODO[]: Possibility of making working with batches of images

class FaceProcessor:
    '''
    Class that will take care of the extra steps for our processing pipeline,
    steps are: dividing the image in n blocks of interest
    '''
    def __init__(self, face_data):
        self.face_data_rgb = face_data
        self.face_data_yuv = None

    def divide_roi_blocks(self, n_blocks=(5, 5)):
        '''
        My implementation of a function that splits a given image in NxM blocks.
        Faster than any package that achieves the same result.
        :param n_blocks: Number of blocks (horizontal, vertical)
        :return: An ndarray with shape (n_blocks[0], n_blocks[1]) containing all the blocks
        '''
        horizontal_blocks, vertical_blocks = n_blocks
        horizontal = np.array_split(self.face_data_yuv, horizontal_blocks)
        splitted_img = [np.array_split(block, vertical_blocks, axis=1) for block in horizontal]
        return np.asarray(splitted_img, dtype=np.ndarray).reshape(n_blocks)

    def project_to_yuv(self, return_image = False):
        '''
        Takes the class image instantiated in the constructor and projects it into YUV color space.
        :param: return_image: boolean expressing if the function should return the image.
        :return: If return_image is true it will return the projected image as a numpy array.
        '''
        normalised_face_data = np.divide(self.face_data_rgb, 255)
        tensor_yuv = tf.image.rgb_to_yuv(normalised_face_data)
        self.face_data_yuv =  tensor_yuv.numpy()
        if return_image: return self.face_data_yuv


if __name__ == "__main__":
    image_path = "D:\\Documents\\Programming\\Python\\thesisProject\\data\\data_in\\sample_set\\aligned.jpg"
    assert os.path.isfile(image_path), f"{image_path} is not a valid path."
    aligned_face = cv2.imread(image_path)
    fp = FaceProcessor(aligned_face)
