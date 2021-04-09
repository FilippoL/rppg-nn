# %%
from datetime import datetime

import cv2
import h5py
import numpy as np
from keras import Sequential
from keras.layers import MaxPooling2D, TimeDistributed, Conv2D, Flatten, Dense
from keras.losses import Huber
from keras.optimizers import Adam
from tqdm import tqdm

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

assert x_train.shape[0] == y_train.shape[0], f"Shapes not matching!"

# %%

model = Sequential()

model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', strides=(2, 2), activation='relu'),
                          input_shape=(32, 121, 480, 640, 3)))

model.add(
    TimeDistributed(
        Conv2D(64, (3, 3),
               padding='same', strides=(2, 2), activation='relu')
    )
)
model.add(
    TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2))
    )
)
# Second conv, 128
model.add(
    TimeDistributed(
        Conv2D(128, (3, 3),
               padding='same', strides=(2, 2), activation='relu')
    )
)
model.add(
    TimeDistributed(
        Conv2D(128, (3, 3),
               padding='same', strides=(2, 2), activation='relu')
    )
)
model.add(
    TimeDistributed(
        MaxPooling2D((2, 2), strides=(2, 2))
    )
)

model.add(Flatten())
model.add(Dense(1))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=Huber(), metrics=['mae'])
history = model.fit(x_train, y_train, epochs=100)
model.save(f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
hf = h5py.File('history_data.h5', 'w')
hf.create_dataset('history', data=history)
hf.close()
