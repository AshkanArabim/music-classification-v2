# Music genre classification

Original project was done by me, [Princess Thomas](https://github.com/pthomas234), and [Emi Reuth](https://github.com/erueth) as our final project for AI4ALL. I decided to expand it on my own to improve the performance and user experience. **Work in progress...**

This is mainly a learning-project, so there is definitely room for improvement. 

## Environment setup
This project used Conda as its main package manager. All the necessary (and unnecessary :p) packages can be found in the `requirements.txt` file ^^^. 

## Genre prediction
If you wish to predict the genre of a `.wav` track that you have lying around, you can simply go to the Interface folder and run the Flask server (`app.py`). Then, simply open `http://127.0.0.1:5000/` in your preferred browser and follow the instructions. **If you are getting a FileNotFoundError when running the server, try changing your directory to the "interface" folder.**

## Training
For training script to work, you can download the dataset and put it in a folder called "archive" in the root directory. Link can be found below.
Then, you can open the "FINAL SCRIPT.py" file found in the folder "Jupyter".

The dataset can be downloaded from from this link:
https://www.kaggle.com/datasets/asisheriberto/music-classification-wav

## Questions?
Something doesn't work? Please reach out to me at ashkan.arabim@gmail.com.
