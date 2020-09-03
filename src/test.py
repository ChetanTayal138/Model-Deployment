from keras.models import Model
from keras.datasets import mnist
from keras.layers import MaxPooling2D, Conv2D, UpSampling2D, Input
import numpy as np
import matplotlib.pyplot as plt



def load_dataset():
    """Load our dataset"""
    (x_train,_), (x_test,_) = mnist.load_data()

    # Normalize values between 0 and 1
    x_train = x_train.astype("float32") / 255.
    x_test = x_test.astype("float32") / 255.

    # Reshape the data so we can feed it into our model
    x_train = np.reshape(x_train, (len(x_train),28,28,1))
    x_test = np.reshape(x_test, (len(x_test),28,28,1))

    return x_train, x_test  

def generate_noisy_data(x_train, x_test, nf=0.5):
    """Generates noisy data"""

    # Add noise to the training and testing data
    x_train_noise = x_train + nf * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noise = x_test + nf * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    # Clip the values between 0 and 1 
    x_train_noise = np.clip(x_train_noise, 0., 1.)
    x_test_noise = np.clip(x_test_noise, 0., 1.)

    return x_train_noise, x_test_noise

def encoder(input_image):
    x = Conv2D(32, 3, activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    
    return x

def decoder(encoded_image):
    x = Conv2D(32,3,activation='relu',padding='same')(encoded_image)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32,3,activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(1,3,activation='sigmoid',padding='same')(x)
    
    return x

def autoencoder(input_image, decoded):
    ae = Model(input_image, decoded)
    ae.compile(optimizer='adam', loss='binary_crossentropy')

    return ae


def train():
    x_train, x_test = load_dataset()
    x_train_noise, x_test_noise = generate_noisy_data(x_train, x_test)
    input_image = Input(shape=(28,28,1))
    print(input_image.shape)
    encoded_image = encoder(input_image)
    print(encoded_image.shape)
    decoded_image = decoder(encoded_image)
    print(decoded_image.shape)

    ae = autoencoder(input_image, decoded_image)

    print(x_train.shape)
    print(x_test.shape)
    print(x_train_noise.shape)
    print(x_test_noise.shape)


    history = ae.fit(x_train_noise, x_train, epochs=10,batch_size=256,shuffle=True,validation_data=(x_test_noise, x_test))
    return history


if __name__ == "__main__":
    train()
    
