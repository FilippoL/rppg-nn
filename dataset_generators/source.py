import multiprocessing as mp
import os
from dataset_generators.dataset_generator_rythmnet import generate_dataset
from FaceManager.FaceDetection import FaceDetectorSSD
from FaceManager.FaceProcessing import FaceProcessor

import tqdm
import cv2

# TODO: [x] Explore signal on a few seconds range, possibly visualize on Tableau
# TODO: [x] Check documents on oximetry
# TODO: [x] Port BVP signal on Tableau
# TODO: [x] Start spatio-temporal maps creation
# TODO: [x] List all requirements first
# TODO: [x] Table for methods and their score per requirements
# TODO: [x] Expand error section on possible other metrics and why the ones have been chosen
# TODO: [x] Filter nxn conv instead of mean (filter size = 3x3)
# TODO: [x] Test validity by enhancing one color channel
# TODO: [x] Change pipeline figures with your own
# TODO: [x] Concatenate onto other dimension
# TODO: [x] Adjust face cropping after alignment
# TODO: [x] Masking/Augmentation
# TODO: [x] Build ResNet50 wrapper class
# TODO: [x] Study loss function


if __name__ == "__main__":
    root_path = "./data/data_in/sample_COHFACE"
    # root_dataset_path = "H:\\DatasetsThesis\\COHFACE"

    # Instantiate face detector and processor
    # fd = FaceDetectorSSD()
    # fp = FaceProcessor()


    paths = []
    fds = []
    fps = []
    vcaps = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in tqdm.tqdm(filenames):
            if "avi" in f:
                paths.append(dirpath)
                fds.append(FaceDetectorSSD())
                fps.append(FaceProcessor())
                vcaps.append(cv2.VideoCapture)

    pool = mp.Pool(10)
    pool.starmap(generate_dataset, zip(paths, fds, fps, vcaps))
    pool.close()
    pool.join()

