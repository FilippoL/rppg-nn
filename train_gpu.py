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


def pearson_correlation(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def pearson_correlation_k(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


wandb.init(project='rppg-nnet', entity='filippo')

os.environ["WANDB_SILENT"] = "true"

path_to_pointers = r"dataset_pointers/dataset_pointers_7x7_no_eyes.csv"
fine_tuning = False

EPOCHS_NUMBER = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
INPUT_SHAPE = (49, 200, 3)
TRAIN_SPLIT = 0.6
SHUFFLE = True
MODEL_NAME = "hopeful-forest"

RANDOM_STATE = 12

config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = EPOCHS_NUMBER
config.input_shape = INPUT_SHAPE
config.train_test_split = TRAIN_SPLIT
config.shuffle = SHUFFLE
config.random_state = RANDOM_STATE


def shuffle_df_chunks(in_df, columns):
    paths = list(zip(in_df[columns[0]], in_df[columns[1]]))
    paths = [[("\\".join(p[0].split("\\")[:8]), "\\".join(p[0].split("\\")[8:])), p[1]] for p in paths]

    grouped_split = [list(j) for i, j in groupby(paths, lambda x: x[0][0])]
    grouped = [[(element[0][0] + "\\" + element[0][1], element[1]) for element in group] for group in grouped_split]
    grouped = np.array(grouped, dtype=np.ndarray)
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(grouped)
    out_df = pd.DataFrame(np.concatenate(grouped), columns=["file", "hr"])
    out_df["hr"] = pd.to_numeric(out_df["hr"])
    # out_df = out_df["hr", "file"]
    return out_df


def scheduler(epoch, lr):
    drop = 0.25
    epochs_drop = 75.0
    lr = LEARNING_RATE * math.pow(drop,
                                  math.floor((1 + epoch) / epochs_drop))
    return lr if lr > 0.00001 else 0.00001


# %%
df = pd.read_csv(path_to_pointers, usecols=[1, 2])
df["hr"] = df["hr"].astype(int)

with open("../../../Datasets/COHFACE/protocols/all/test.txt", "r") as f:
    test = f.readlines()

with open("../../../Datasets/COHFACE/protocols/all/train.txt", "r") as f:
    train = f.readlines()

root = df["file"][0]
all_files = df["file"].to_list()

train_paths = [os.path.join("\\".join(root.split("\\")[:6]), path.strip("\n").replace("/", "\\"))[:-5] for path in
               train]
train_indices = np.concatenate(
    [[np.asarray([train_path in all_paths for all_paths in all_files]).nonzero()][0][0].tolist() for train_path in
     train_paths])

test_paths = [os.path.join("\\".join(root.split("\\")[:6]), path.strip("\n").replace("/", "\\"))[:-5] for path in test]
test_indices = np.concatenate(
    [[np.asarray([test_path in all_paths for all_paths in all_files]).nonzero()][0][0].tolist() for test_path in
     test_paths])

df_train = shuffle_df_chunks(df.loc[train_indices], ["file", "hr"])

df_validate = shuffle_df_chunks(df.loc[test_indices], ["file", "hr"])

# %%

training_datagen, validation_datagen = ImageDataGenerator(), ImageDataGenerator()

train_generator = training_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw',
    shuffle=False,
    steps_per_epoch=len(df_train) / BATCH_SIZE)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    shuffle=False,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw')

# %%
run_name = wandb.run.name

callbacks = [WandbCallback(
    training_data=train_generator,
    validation_data=val_generator,
    input_type="images"),
    ModelCheckpoint(
        filepath=f"models/{run_name}/",
        save_weights_only=True,
        mode='min',
        save_best_only=True),
    ReduceLROnPlateau(),
    LearningRateScheduler(scheduler)
    # EarlyStopping(patience=15, baseline=)
]


def weight_image(image_pixels):
    weight_map = np.load("report_scripts/variance_based_weights_single.npy")
    # weight_map = Conv1D(3, 1)(tf.convert_to_tensor(weight_map))
    # weight_map = np.load("report_scripts/variance_based_weights.npy")
    img = image_pixels * weight_map
    return img


if not fine_tuning:
    print("Training from scratch...\n")
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    base_model = ResNet18(input_shape=INPUT_SHAPE, weights='imagenet', include_top=False)

    base_model.trainable = False

    input_layer = Input(shape=INPUT_SHAPE)

    # weight_map = Conv1D(3, 1)(tf.convert_to_tensor(variance_weights))

    # weighting_layer = Lambda(weight_image, name="lambda_layer")(input_layer)

    base = base_model(input_layer, training=False)
    head = Flatten()(base)
    outputs = Dense(1)(head)

    model = Model(inputs=input_layer, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mean_absolute_error",
                  metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError(), pearson_correlation])
    # %%

    os.makedirs(f"models/{run_name}/", exist_ok=True)

    history = model.fit(train_generator,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        epochs=EPOCHS_NUMBER,
                        shuffle=False)

    model.save(f"models/{run_name}/rppg-nnet-rgb.h5")
else:
    print(f"Fine tuning {MODEL_NAME}...\n")

    reconstructed_model = load_model(f"models/{MODEL_NAME}/rppg-nnet-rgb.h5",
                                     custom_objects={"pearson_correlation": pearson_correlation}, compile=False)

    reconstructed_model.trainable = True

    reconstructed_model.compile(optimizer=Adam(learning_rate=1.0e-7), loss="mean_absolute_error",
                                metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()])

    reconstructed_model.fit(train_generator,
                            callbacks=callbacks[:-2],
                            validation_data=val_generator,
                            epochs=75,
                            shuffle=False)

    os.makedirs(f"models/fine_tuned/{MODEL_NAME}/", exist_ok=True)
    reconstructed_model.save(f"models/fine_tuned/{MODEL_NAME}/rppg-nnet-rgb_{75}.h5")
