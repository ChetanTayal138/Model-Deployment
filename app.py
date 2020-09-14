from flask import Flask, render_template,  url_for, redirect, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow.keras as keras
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from utils import get_greyscale_image, get_prediction

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """Load models before first sending first request."""
    global encoder
    global decoder
    encoder = keras.models.load_model("models/encoder")
    decoder = keras.models.load_model("models/decoder")
    print("LOADED MODELS")


@app.route('/', methods=["GET", "POST"])
def upload_file():
    """Renders page for uploading the image."""
    return render_template("upload.html")


@app.route('/upload', methods=["GET", "POST"])
def get_file():
    """Save uploaded file to server."""
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("static/uploads",secure_filename(f.filename)))
        print("IMAGE SAVED : ",f.filename )
        
        return redirect(url_for('image_classifier', filename=f.filename))


@app.route('/predict/<filename>', methods=['GET'])
def image_classifier(filename):
    """Carries out conversion from noisy to clean image."""
    upload_path = os.path.join("static/uploads", filename)
    save_path = os.path.join("static/predictions", filename)

    imgs = get_greyscale_image(upload_path)

    # Encode the image through our encoder block
    encoded_imgs = encoder.predict(imgs)
    # Decode image through decoder block that has the layers made up from our trained autoencoder
    decoded_imgs = decoder.predict(encoded_imgs)
    # Get the final processed prediction
    prediction = get_prediction(decoded_imgs)
    
    # Save the result in our predictions folder
    plt.imsave(save_path, prediction)

    return render_template('index.html', noisy_image=upload_path,denoised_image=save_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='80', debug=True)
