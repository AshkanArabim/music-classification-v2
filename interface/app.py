import flask
from flask import Flask, render_template, request, flash, redirect
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import librosa  # library audio analysis

print("flask version", flask.__version__)

UPLOAD_FOLDER = "./userfiles/"
ALLOWED_EXTENSIONS = {'wav'}

# some useful funcs...
FREQ_CAP = 1024
SAMPLING_RATE = 5512
SAMPLING_LENGTH = 27  # seconds
MEL_DETAIL = 64
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

def classify(path):
    # define custom layer...
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

    # load model 
    saved_model = keras.models.load_model(
        '../Jupyter/models/mel2-ash-FULL.keras', custom_objects={'CNN': CNN})
    
    # make prediction on file
    audio_data = make_mel(path)
    prediction = saved_model.predict(audio_data)

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

    # convert output (numpy array) to class number
    prediction = np.argmax(prediction, axis=1)[0]

    # convert class number to genre name
    prediction = pred_names[prediction]

    return prediction

def allowed_file(filename): # returns
    extension = filename.rsplit('.', 1)[1]
    return '.' in filename and extension in ALLOWED_EXTENSIONS

# Site stuff
app = Flask(__name__)
app.secret_key = "bruh"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# main page with submit button
@app.route('/', methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        print("request:", request)
        print(request.files)
        if 'file' not in request.files:
            flash('No file part!')
            print("No file part!")
            return redirect(request.url)
        
        file = request.files['file']
        filename = file.filename;

        print("filename", filename)

        if filename == '':
            flash('No file selected!')
            print('No file selected!')
            return redirect(request.url)
        
        # in case file isn't a wav
        if (not allowed_file(filename)):
            flash("File isn't wav!")
            print("File isn't wav!")
            return redirect(request.url)

        file.save(f"{UPLOAD_FOLDER}/userfile.wav")

        # process file with keras
        result = classify(f"{UPLOAD_FOLDER}/userfile.wav")

        print("genre:", result)

        # litearlly just log it to the std output
        flash(result)

        return render_template('index.html', length=SAMPLING_LENGTH, result=result)
    else:
        return render_template('index.html', length=SAMPLING_LENGTH) 

if (__name__ == '__main__'):
    app.run()
