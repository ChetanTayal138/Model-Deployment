import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_greyscale_image(upload_path):
    img = plt.imread(upload_path)/255. # Divide by 255. to normalize pixel values between 0 and 1 as our model has been trained on the same distribution
    r,g,b = img[:,:,0], img[:,:,1], img[:,:,2]
    img = 0.2989*r + 0.5870*g + 0.1140*b # Convert the image to grayscale    
    imgs = np.array([img]) 
    imgs = imgs.reshape(imgs.shape[0],imgs.shape[1],imgs.shape[2],1)

    return imgs

def get_prediction(decoded_imgs):
    prediction = decoded_imgs[0].reshape(28,28)
    kernel = np.ones((2,2), np.float)
    prediction = cv2.erode(decoded_imgs[0].reshape(28,28), kernel)
    prediction = cv2.convertScaleAbs(prediction, alpha=1.5, beta=0)

    return prediction
    
