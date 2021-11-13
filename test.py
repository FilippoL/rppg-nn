import math
import os
from itertools import groupby

import numpy as np
import pandas as pd
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Lambda, Conv1D
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbCallback

import wandb

BATCH_SIZE = 64
INPUT_SHAPE = (49, 200, 3)

path_to_pointers = r"dataset_pointers/dataset_pointers_7x7_no_eyes.csv"

df = pd.read_csv(path_to_pointers, usecols=[1, 2])
df["hr"] = df["hr"].astype(int)

with open("../../../Datasets/COHFACE/protocols/all/all.txt", "r") as f:
    test = f.readlines()

root = df["file"][0]
all_files = df["file"].to_list()

test_paths = [os.path.join("\\".join(root.split("\\")[:6]), path.strip("\n").replace("/", "\\"))[:-5] for path in test]
test_indices = np.concatenate(
    [[np.asarray([test_path in all_paths for all_paths in all_files]).nonzero()][0][0].tolist() for test_path in
     test_paths])

df_validate = df.loc[test_indices]

validation_datagen = ImageDataGenerator()

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    shuffle=False,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw')

model = load_model(
    r"C:\Users\pippo\Documents\Programming\Python\rppg-nnet\model\rppg-nnet.h5")

print("Evaluate on test data")
results = model.evaluate(val_generator)
print("Test MAE", results[0])