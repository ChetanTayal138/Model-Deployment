from flask import Flask, render_template, request, jsonify, url_for, redirect
from werkzeug.utils import secure_filename
import base64
import json
import os
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
import tensorflow.keras as keras
import sys
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """Load models before first request."""


@app.route('/', methods=["GET", "POST"])
def upload_file():
    return render_template("upload.html")


@app.route('/upload', methods=["GET", "POST"])
def get_file():
    """Save uploaded file to server."""
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("uploads",secure_filename(f.filename)))
        print("IMAGE SAVED : ",f.filename )
        return redirect(url_for('image_classifier', filename=f.filename))

@app.before_first_request
def before_first_request():
    """Load models before first request."""
    global encoder
    global decoder
    encoder = keras.models.load_model("models/encoder")
    decoder = keras.models.load_model("models/decoder")
    print("LOADED MODELS")

@app.route('/predict/<filename>', methods=['GET'])
def image_classifier(filename):
    """Endpoint for returning prediction json."""

    loaded_img = os.path.join("data",filename)
    img = cv2.imread(loaded_img, cv2.IMREAD_GRAYSCALE)/255.
    print("LOADED IMAGE OF SHAPE : ", img.shape)
    imgs = []
    imgs.append(img)
    imgs = np.array(imgs)
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)
    print(imgs.shape)

    encoded_imgs = encoder.predict(imgs)
    print("ENCODED IMAGES")
    print(encoded_imgs.shape)
    decoded_imgs = decoder.predict(encoded_imgs)
    print(decoded_imgs.shape)

    plt.imsave("predictions/predicted.jpg", decoded_imgs[0].reshape(28,28))

    
    return "Success"
    #return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])



if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)
