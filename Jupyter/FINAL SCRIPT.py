# DEVIATIONS FROM IPYNB MODEL:
# changed sampling length to 27 seconds
# removed filter function to exclude garbage: did it manually
# removed cardinality calculator: only used for debug
# removed the 'activation' parameter from CNN
# removed anything related to ipython
# added **kwargs and get_cofig to CNN layer
# ----- CHANGES MADE TILL HERE ----

# CHANGES TO MAKE
# change frequency cap

# THIS IS OUR BEST BET, AND FINAL MODEL
# 71/71 [==============================] - 53s 608ms/step - loss: 3.4936 - accuracy: 0.1302
# Epoch 2/20
# 71/71 [==============================] - 44s 617ms/step - loss: 2.5081 - accuracy: 0.1705
# Epoch 3/20
# 71/71 [==============================] - 44s 610ms/step - loss: 2.3549 - accuracy: 0.2238
# Epoch 4/20
# 71/71 [==============================] - 44s 617ms/step - loss: 2.1625 - accuracy: 0.2940
# Epoch 5/20
# 71/71 [==============================] - 45s 622ms/step - loss: 2.0590 - accuracy: 0.3390
# Epoch 6/20
# 71/71 [==============================] - 44s 620ms/step - loss: 2.0047 - accuracy: 0.3642
# Epoch 7/20
# 71/71 [==============================] - 45s 623ms/step - loss: 1.7722 - accuracy: 0.4324
# Epoch 8/20
# 71/71 [==============================] - 44s 620ms/step - loss: 1.5067 - accuracy: 0.5211
# Epoch 9/20
# 71/71 [==============================] - 45s 625ms/step - loss: 1.2604 - accuracy: 0.6127
# Epoch 10/20
# 71/71 [==============================] - 44s 617ms/step - loss: 1.1344 - accuracy: 0.6635
# Epoch 11/20
# 71/71 [==============================] - 45s 625ms/step - loss: 0.9333 - accuracy: 0.7297
# Epoch 12/20
# 71/71 [==============================] - 45s 626ms/step - loss: 0.7737 - accuracy: 0.7840
# Epoch 13/20
# 71/71 [==============================] - 45s 629ms/step - loss: 0.6958 - accuracy: 0.8259
# Epoch 14/20
# 71/71 [==============================] - 45s 624ms/step - loss: 0.5824 - accuracy: 0.8596

import keras
from keras import layers
import tensorflow as tf
# import matplotlib.pyplot as plt # unused in python script
import os
import numpy as np
import librosa  # library for audio analysis
import sys

# made change: removed v1 eager mode and experimental debug mode
print(tf.__version__)
tf.config.run_functions_eagerly(True)

print("Executing eagerly?", tf.executing_eagerly())

print("GPU:", tf.config.list_physical_devices('GPU'))

# declare global variables
TRAIN_SET_DIR = os.path.join(
    '..', 'archive', 'TRAIN_V2', 'data_out_2')  # 2 epochs
# TRAIN_SET_DIR = os.path.join('..', 'halved') # ?
# TRAIN_SET_DIR = os.path.join('..', 'minimized') #15 epochs max
RANDOM_SEED = 42
VALIDATION_RATE = 0.1
FREQ_CAP = 1024
SAMPLING_RATE = 5512
SAMPLING_LENGTH = 27  # seconds
MEL_DETAIL = 64
BATCH_SIZE = 64
EPOCHS = 15

# create array to store all the training song paths
song_paths = []
unique_genre_nums = os.listdir(TRAIN_SET_DIR)
song_genre_nums = []

# loop over each music genre
for genre_num in unique_genre_nums:

    # in each music genre, add the paths of all the music to the array
    for song_path in os.listdir(os.path.join(TRAIN_SET_DIR, genre_num)):
        song_paths.append(os.path.join(TRAIN_SET_DIR, genre_num, song_path))
        song_genre_nums.append(int(genre_num))

# convert both of these to np arrays for efficiency
song_paths = np.array(song_paths)
song_genre_nums = np.array(song_genre_nums)

# suffle both arrays with the same shuffle seed
np.random.seed(RANDOM_SEED)
np.random.shuffle(song_paths)
np.random.shuffle(song_genre_nums)

# split both into training and testing sets
num_songs = len(song_genre_nums)
num_validation = int(num_songs * VALIDATION_RATE)

validation_genres = song_genre_nums[-num_validation:]
validation_paths = song_paths[-num_validation:]

training_genres = song_genre_nums[: -num_validation]
training_paths = song_paths[: -num_validation]

print("number of validations")
print(len(validation_genres))
print('number of trainings')
print(len(training_paths))

# ------- this should roughly be the number of songs that we're using -------
# number of validations
# 1990
# number of trainings
# 17919


def create_dataset(audio_paths, audio_classes):

    # create zip dataset
    ds = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(audio_paths),
            tf.data.Dataset.from_tensor_slices(audio_classes)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # REMOVED due to performance overhead
    # exclude short tracks (garbage data) from the dataset
    # ds = ds.filter(lambda x, y: tf.py_function(
    #     exclude_short_tracks, [x, y], tf.bool))

    # map each path to a spectrogram
    # contains the mel from all sources' first [SAMPLING_LENGTH] seconds.
    ds = ds.map(
        lambda x, y: (tf.py_function(make_mel, [x], tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH_SIZE)
    # print(ds)

    return ds

# return true only if the file is longer than SAMPLING_LENGTH


def exclude_short_tracks(path, label):
    path = path.numpy().decode('ascii')
    # print("==========THIS IS MY DATA:", path, label)
    # print("path:", path)
    length = librosa.get_duration(path=path)
    # print("length:",length)
    return length > SAMPLING_LENGTH

# get path, read audio data, pass it into next func to get mel, then return it
# this will be used in map (look above)


def make_mel(path):
    # the first x seconds of the track are imported
    path = path.numpy().decode('ascii')

    audio_data, _ = librosa.load(
        path, sr=SAMPLING_RATE, duration=SAMPLING_LENGTH
    )
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=SAMPLING_RATE, n_mels=MEL_DETAIL, fmax=FREQ_CAP
    )

    # expand dimensions so you have 4
    mel = tf.expand_dims(mel, axis=-1)

    return mel


# convert path and class arrays to dataset...
train_ds = create_dataset(training_paths, training_genres)
valid_ds = create_dataset(validation_paths, validation_genres)

# DEBUG: see if running these first solves the matrix resizing
# this is very intensive, comment if you want more performance
# print("Training dataset cardinality:",
#       train_ds.reduce(0, lambda x, _: x + 1).numpy())
# print("Validation dataset cardinality:",
#       valid_ds.reduce(0, lambda x, _: x + 1).numpy())

# DEBUG: print the dimensions of all tensors different from (64, 64, 1163, 1)
# doesn't print anything...

# print("Abnormal shapes:")
# for item_tuple in train_ds:
#     shape = item_tuple[0].shape
#     print(shape)
#     if (shape != (BATCH_SIZE, MEL_DETAIL, 1163, 1)):
#         print(shape)

# model -----------------------------


class CNN(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CNN, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(filters, kernel_size)
        self.normalization = layers.ReLU()
        self.pooling = layers.MaxPool2D()

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
        })
        return config

    def call(self, input):
        # print("=====================\n",input)
        x = self.conv(input)
        x = self.normalization(x)
        x = self.pooling(x)
        return x


model = keras.Sequential()

# convolutional layers
model.add(CNN(32, 3))
model.add(CNN(64, 3))
model.add(CNN(128, 3))

# flatten
model.add(layers.Flatten())

# dense layers

# ORIGINAL
#
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))

#
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))

# 11 --> 17 --> 24 --> 36
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))

# BEST
# 13 --> 19 --> 34
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(19, activation='linear'))

# compile
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), optimizer='adam', metrics=['accuracy'])

model.build(input_shape=(BATCH_SIZE, MEL_DETAIL, 291, 1))
print(model.summary())

# CHANGES BEFORE FINAL BUILD
# set epochs to 20


# testing Ashkan's model

# fit
model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# evaluate
model.evaluate(valid_ds, batch_size=BATCH_SIZE)

# save model
model.save('models/mel2-ash-FULL.keras')
