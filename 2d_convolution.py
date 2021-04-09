# %%
import cv2
import h5py
import numpy as np
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy, MeanAbsoluteError, Huber
from keras import layers
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

# %%
sample_per_second = 2

# Frame per second
fps = 20
# Blood volume signal reading per second
rds = 256
# Path to pickled data file
path_to_hdf5 = "data/data_in/data.hdf5"
path_to_video = "data/data_in/data.avi"

signal_sampling_rate = rds // sample_per_second
frame_sampling_rate = fps // sample_per_second

video_capture = cv2.VideoCapture(path_to_video)
n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

resolution = (int(video_capture.get(3)),
              int(video_capture.get(4)))

hdf5_file = h5py.File(path_to_hdf5, 'r')
bvp_raw = np.array(hdf5_file['pulse'])

sampled_frames_buffer = []

for frame_number in tqdm(range(n_frames)):
    ret, frame = video_capture.read()
    if frame_number % frame_sampling_rate == 0:
        sampled_frames_buffer.append(frame)

video_capture.release()
cv2.destroyAllWindows()

print(f"Sampled {len(sampled_frames_buffer)} frames out of {n_frames}.")

sampled_frames_buffer = np.array(sampled_frames_buffer, dtype=np.float32)
bvp_raw_sampled = bvp_raw[::signal_sampling_rate]

x_train = sampled_frames_buffer
y_train = bvp_raw_sampled

# %%
logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

# %%

# model = applications.resnet50.ResNet50(input_shape=x_train.shape[1:])
# x = model.output
# x = Dropout(0.7)(x)
# prediction = Dense(1)(x)

# x_train shape -> (121, 480, 640, 3)

model = Sequential()

model.add(layers.Conv2D(64, kernel_size=(15, 10), strides=1, input_shape=x_train.shape[1:]))
model.add(layers.MaxPooling2D(pool_size=(15, 10), strides=(2, 2)))
model.add(layers.ELU())
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, kernel_size=(15, 10), strides=1))
model.add(layers.MaxPooling2D(pool_size=(15, 10), strides=(1, 1)))
model.add(layers.ELU())
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, kernel_size=(15, 10), strides=1))
model.add(layers.MaxPooling2D(pool_size=(15, 10), strides=(1, 1)))
model.add(layers.ELU())
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, kernel_size=(12, 10), strides=1))
model.add(layers.MaxPooling2D(pool_size=(15, 10), strides=(1, 1)))
model.add(layers.ELU())
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
# model.add(layers.Conv2D(1, kernel_size=(1, 1), strides=1))
model.add(layers.Dense(1, activation="linear"))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=Huber(), metrics=['mae'])
history = model.fit(x_train, y_train, epochs=100, batch_size=1)
model.save(f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
hf = h5py.File('history_data.h5', 'w')
hf.create_dataset('history', data=history)
hf.close()

