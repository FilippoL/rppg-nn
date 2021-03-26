import cv2
import h5py
import numpy as np
import tensorflow
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from tqdm import tqdm

batch_size = 8
no_epochs = 30
learning_rate = 0.001
no_classes = 1
validation_split = 0.2
verbosity = 1
sample_per_second = 0.25
fps = 20
rds = 256
path_to_hdf5 = "data_in/data.hdf5"
path_to_video = "data_in/data.avi"
signal_sampling_rate = rds // sample_per_second
frame_sampling_rate = fps // sample_per_second

video_capture = cv2.VideoCapture(path_to_video)
n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

resolution = (int(video_capture.get(3)),
              int(video_capture.get(4)))

sampled_frames_buffer = []

for frame_number in tqdm(range(n_frames)):
    ret, frame = video_capture.read()
    if frame_number % frame_sampling_rate == 0:
        sampled_frames_buffer.append(frame)

video_capture.release()
cv2.destroyAllWindows()

print(f"Sampled {len(sampled_frames_buffer)} frames out of {n_frames}.")

with h5py.File(path_to_hdf5) as hf:
    bvp_raw = np.array(hf['pulse'])

    sampled_frames_buffer = np.array(sampled_frames_buffer, dtype=np.float32)
    bvp_raw_sampled = bvp_raw[::int(signal_sampling_rate)]

    x_train = sampled_frames_buffer
    target_train = bvp_raw_sampled

    sample_shape = (batch_size, resolution[0], resolution[1], 3)
    # tensorflow.keras.backend.set_image_data_format('channels_first')

    model = Sequential()
    model.add(
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D((2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(no_classes, activation='softmax'))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    history = model.fit(x_train, target_train,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        validation_split=validation_split)
