import pandas as pd
from keras import Input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Flatten
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback
import tensorflow as tf
import wandb
import math

wandb.init(project='rppg-nnet', entity='filippo')

EPOCHS_NUMBER = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
INPUT_SHAPE = (32,200,3)


config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = EPOCHS_NUMBER

def scheduler(epoch, lr):
   drop = 0.25
   epochs_drop = 10.0
   lr = LEARNING_RATE * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lr


# %%
df = pd.read_csv("withoutnan.csv", usecols=[2, 3])

train_percentual = 0.75
train_cut_idx = int(len(df) * train_percentual)

df_train = df[:train_cut_idx]
df_validate = df[train_cut_idx:]

# %%

training_datagen, validation_datagen = ImageDataGenerator(), ImageDataGenerator()

train_generator = training_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2],
    class_mode='raw',
    preprocessing_function=tf.image.rgb_to_yuv,
    steps_per_epoch=len(df_train) / BATCH_SIZE)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=INPUT_SHAPE[:2],
    preprocessing_function=tf.image.rgb_to_yuv,
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
              metrics=[RootMeanSquaredError(),
                       MeanAbsolutePercentageError()])

# %%



callbacks = [WandbCallback(
    training_data=train_generator,
    validation_data=val_generator,
    input_type="images"),
    LearningRateScheduler(scheduler),
    ModelCheckpoint(
        filepath="models/resnet50_checkpoint/",
        save_weights_only=True,
        mode='max',
        save_best_only=True)
]

history = model.fit(train_generator,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    epochs=EPOCHS_NUMBER)
model.save("models/rppg-nnet-rgb.h5")