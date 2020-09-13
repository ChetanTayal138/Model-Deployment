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

@app.route('/', methods=["GET", "POST"])
def upload_file():
    """Renders page for uploading the image."""
    
@app.route('/upload', methods=["GET", "POST"])
def get_file():
    """Save uploaded file to server."""

@app.route('/predict/<filename>', methods=['GET'])
def image_classifier(filename):
    """Carries out conversion from noisy to clean image."""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='80', debug=True)
