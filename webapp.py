"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
import datetime
from PIL import Image
from gtts import gTTS
import torch
from flask import Flask, render_template, request, redirect, url_for,send_file
import base64
import pygame
import time

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

img_savename = ""
audio_filename = ""

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if request.form.get("button") == "display":
            return redirect(url_for("display"))
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])
        results.render()
        ow_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        global img_savename
        img_savename = f"static/recognized/recognized.jpg"
        Image.fromarray(results.ims[0]).save(img_savename)
         # updates results.imgs with boxes and labels
         # Perform label recognition and determine the room type and detected labels
        results = perform_label_recognition(img)

        # Generate audio based on the room type and detected labels
        generated_audio = generate_audio(results)

        # Save the generated audio as an mp3 file
        global audio_filename
        audio_filename =f"static/recognized/recognized.mp3"

        # Check if the file already exists
        if os.path.exists(audio_filename):
            # Close the file if it is open
            try:
                # Open the file in read mode to acquire a file handle
                file_handle = open(audio_filename, 'r')

                # Close the file handle
                file_handle.close()
            except IOError:
                pass

            # Wait for a short period (e.g., 1 second)
            time.sleep(1)

            # Attempt to remove the existing file
            try:
                os.remove(audio_filename)
            except OSError as e:
                print(f"Error occurred while removing the file: {e}")

        # Save the new audio file
        generated_audio.save(audio_filename)


        # Play the background audio
        play_background_audio(audio_filename)
        return redirect(img_savename)
       

    return render_template("index.html")

@app.route("/display")
def display():
    image_data = img_savename
    audio_data = audio_filename

    if image_data is None:
        return "Image data not provided."

    return render_template("display.html", image_data=image_data, audio_data=audio_data)



    

def perform_label_recognition(image):
    # Extract labels from the YOLOv5 model's predictions
    results = model(image)
    labels = results.pred[0][:, -1].cpu().numpy().astype(int)
    label_names = model.module.names if hasattr(model, 'module') else model.names
    detected_labels = [label_names[label] for label in labels]
    print(detected_labels)
    bedroom = ["Bed","CupBoard"]
    dinningTable = ["Chair","DinningTable"]
    Gameroom = ["BowlingBall","BowligPins"]
    for i in detected_labels:
        print(i)
        if i in bedroom:
            result = "The Detected Image is bedroom as the algorithm detected bed and cupboards"
        elif i in dinningTable:
            result = "The Detected Image is dinning table as the algorithm detected dining table and chairs "
        elif i in Gameroom:
            result = "The Detected Image is gameroom as the algorithm detected bowling ball and bowling pinns "
        else:
            result = "Choose correct input image"
        print(result)
    return result
   
def generate_audio(result):
    # Generate audio based on the room type and detected labels
    
    # Generate audio using gTTS and save as an MP3 file
    tts = gTTS(text=result, lang="en")
    return tts

def play_background_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0",debug=True,port=args.port)  # debug=True causes Restarting with stat
