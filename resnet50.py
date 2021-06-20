import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
import wandb
from wandb.keras import WandbCallback

wandb.init(entity="wandb", project="rppg-nnet")
config = wandb.config

# %%
df = pd.read_csv("dataset_pointers.csv", usecols=[1, 2])

train_percentual = 0.9
train_cut_idx = int(len(df) * train_percentual)

df_train = df[0:train_cut_idx]
df_validate = df[train_cut_idx:]

print(f"Training size: {len(df_train)}")
print(f"Validate size: {len(df_validate)}")

# %%

training_datagen = ImageDataGenerator()

train_generator = training_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="file",
    y_col="hr",
    target_size=(200, 35),
    class_mode='other')

validation_datagen = ImageDataGenerator()

val_generator = validation_datagen.flow_from_dataframe(
    dataframe=df_validate,
    x_col="file",
    y_col="hr",
    target_size=(200, 35),
    class_mode='other')

# %%

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(200, 35, 3))


head = Flatten(name="flatten")(base_model.output)
head = Dense(1, activation="linear")(head)

model = Model(inputs=base_model.inputs, outputs=head)
model.compile(optimizer="adam", loss="mean_absolute_error")

#%%


history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=25)
