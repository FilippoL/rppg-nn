import pandas as pd
from keras import Input
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

import wandb

wandb.init(project='rppg-nnet', entity='filippo')

EPOCHS_NUMBER = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

config = wandb.config
config.learning_rate = LEARNING_RATE
config.epochs = EPOCHS_NUMBER

# %%
df = pd.read_csv("withoutnan.csv", usecols=[2, 3])

train_percentual = 0.9
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
    target_size=(32, 200),
    class_mode='raw',
    steps_per_epoch=len(df_train) / BATCH_SIZE)

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    batch_size=BATCH_SIZE,
    target_size=(32, 200),
    class_mode='raw')

# %%

base_model = ResNet50(include_top=False, input_shape=(32, 200, 3))
base_model.trainable = False

inputs = Input(shape=(32, 200, 3))
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
    # EarlyStopping(restore_best_weights=True),
    ModelCheckpoint(
        filepath="models/resnet50_checkpoint",
        save_weights_only=True,
        monitor='mean_absolute_error',
        mode='max',
        save_best_only=True)
]

history = model.fit(train_generator,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    epochs=EPOCHS_NUMBER)
