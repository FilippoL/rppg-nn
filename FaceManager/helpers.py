import json
import numpy as np


with open(
        r"C:\Users\pippo\Documents\Programming\Python\rppg-nnet\FaceManager\config\landmarks_indices.json") as json_file:
    facial_landmarks_indices = json.load(json_file)

def pad(img, w, h, filter_size):
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)

    padded = []

    for c in range(left_pad):
        horizontal_padding = []
        to_be_padded = img if c == 0 else padded
        for i in range(0, to_be_padded.shape[0], filter_size):
            sub_array = to_be_padded[i:i + filter_size, :filter_size]
            horizontal_padding.append(np.mean(sub_array, axis=1))
        horizontal_padding = np.concatenate(horizontal_padding, axis=0).reshape(-1, 1, 3)
        horizontal_padding = np.delete(horizontal_padding, slice(to_be_padded.shape[0], horizontal_padding.shape[0]), 1)
        padded = np.hstack((horizontal_padding, to_be_padded))

    if type(padded) == list: padded = img
    for c in range(right_pad):
        horizontal_padding = []
        for i in range(0, padded.shape[0], filter_size):
            sub_array = padded[i:i + filter_size, -filter_size:]
            horizontal_padding.append(np.mean(sub_array, axis=1))
        horizontal_padding = np.concatenate(horizontal_padding, axis=0).reshape(-1, 1, 3)
        horizontal_padding = np.delete(horizontal_padding, slice(padded.shape[0], horizontal_padding.shape[0]), 1)
        padded = np.hstack((padded, horizontal_padding))

    if type(padded) == list: padded = img
    for c in range(top_pad):
        vertical_padding = []
        for i in range(0, padded.shape[1], filter_size):
            sub_array = padded[:filter_size, i:i + filter_size]
            vertical_padding.append(np.mean(sub_array, axis=1))
        vertical_padding = np.concatenate(vertical_padding, axis=0).reshape(1, -1, 3)
        vertical_padding = np.delete(vertical_padding, slice(padded.shape[1], vertical_padding.shape[1]), 1)
        padded = np.vstack((vertical_padding, padded))

    if type(padded) == list: padded = img
    for c in range(bottom_pad):
        vertical_padding = []
        for i in range(0, padded.shape[1], filter_size):
            sub_array = padded[-filter_size:, i:i + filter_size]
            vertical_padding.append(np.mean(sub_array, axis=1))
        vertical_padding = np.concatenate(vertical_padding, axis=0).reshape(1, -1, 3)
        vertical_padding = np.delete(vertical_padding, slice(padded.shape[1], vertical_padding.shape[1]), 1)
        padded = np.vstack((padded, vertical_padding))

    return padded

