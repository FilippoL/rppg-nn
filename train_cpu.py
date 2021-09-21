import math
import os
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from tensorflow.keras.models import Model
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


wandb.init(project='rppg-nnet', entity='filippo')

os.environ["WANDB_SILENT"] = "true"

path_to_pointers = r"dataset_pointers/dataset_pointers_7x7_no_eyes.csv"

EPOCHS_NUMBER = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 64
INPUT_SHAPE = (224, 224, 3)
TRAIN_SPLIT = 0.75
SHUFFLE = False

RANDOM_STATE = 12

config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = EPOCHS_NUMBER
config.input_shape = INPUT_SHAPE
config.train_test_split = TRAIN_SPLIT
config.shuffle = SHUFFLE
config.random_state = RANDOM_STATE


def scheduler(epoch, lr):
    drop = 0.5
    epochs_drop = 10.0
    lr = LEARNING_RATE * math.pow(drop,
                                  math.floor((1 + epoch) / epochs_drop))
    return lr


# %%
df = pd.read_csv(path_to_pointers, usecols=[1, 2])
if SHUFFLE: df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

train_percentual = TRAIN_SPLIT
train_cut_idx = int(len(df) * train_percentual)

df_train = df[:train_cut_idx]
#df_train = df_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df_validate = df[train_cut_idx:]
#df_validate = df_validate.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# %%

training_datagen, validation_datagen = ImageDataGenerator(), ImageDataGenerator()

train_generator = training_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw',
    steps_per_epoch=len(df_train) / BATCH_SIZE)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw')

# %%

base_model = ResNet50(include_top=False, input_shape=INPUT_SHAPE)
base_model.trainable = False

inputs = Input(shape=INPUT_SHAPE)
head = base_model(inputs, training=False)
head = Flatten()(head)
outputs = Dense(1)(head)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mean_absolute_error",
              metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError(), pearson_correlation])
# %%

run_name = wandb.run.name
os.makedirs(f"models/{run_name}/")

callbacks = [WandbCallback(
    training_data=train_generator,
    validation_data=val_generator,
    input_type="images"),
    LearningRateScheduler(scheduler),
    ModelCheckpoint(
        filepath=f"models/{run_name}/",
        save_weights_only=True,
        mode='max',
        save_best_only=True)
#    EarlyStopping(
#        mode='min',
#        min_delta=0.05,
#        patience=15)
]

history = model.fit(train_generator,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    epochs=EPOCHS_NUMBER)

model.save(f"models/{run_name}/rppg-nnet-rgb.h5")
