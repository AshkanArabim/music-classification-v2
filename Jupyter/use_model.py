import tensorflow as tf
import keras
from keras import layers
import numpy as np
import librosa  # library audio analysis

# define custom layer (yes, again)


class CNN(keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CNN, self).__init__(**kwargs)
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
        x = self.conv(input)
        x = self.normalization(x)
        x = self.pooling(x)
        return x


# load model to see if it evaluates
saved_model = keras.models.load_model(
    'models/mel2-ash-FULL.keras', custom_objects={'CNN': CNN})

FREQ_CAP = 7000
SAMPLING_RATE = 22050
SAMPLING_LENGTH = 27  # seconds
MEL_DETAIL = 64

# some useful funcs...


def make_mel(path):
    # the first x seconds of the track are imported
    audio_data, _ = librosa.load(
        path, sr=SAMPLING_RATE, duration=SAMPLING_LENGTH
    )
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=SAMPLING_RATE, n_mels=MEL_DETAIL, fmax=FREQ_CAP
    )

    # expand to 4 dimensions
    mel = tf.expand_dims(mel, axis=0)
    mel = tf.expand_dims(mel, axis=3)

    return mel


pred_names = [
    "Electronic",
    "Rock",
    "Punk",
    "Experimental",
    "Hip-Hop",
    "Folk",
    "Chiptune / Glitch",
    "Instrumental",
    "Pop",
    "International",
    "Ambient Electronic",
    "Classical",
    "Old-Time / Historic",
    "Jazz",
    "Country",
    "Soul-RnB",
    "Spoken",
    "Blues",
    "Easy Listening"
]

# filepath = '../testing music/elevator-music-bossa-nova-background-music-version-60s-10900.mp3'
filepath = "../archive/TRAIN_V2/data_out_2/0/313.wav"  # should be electronic
input = make_mel(filepath)

# make prediction
prediction = saved_model.predict(input)  # gives a numpy array
prediction = np.argmax(prediction, axis=1)[0]

# convert class number to genre name
prediction = pred_names[prediction]

print(prediction)
