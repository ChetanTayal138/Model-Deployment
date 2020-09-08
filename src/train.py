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

def encode(input_image):
    x = Conv2D(32, 3, activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    
    return x

def decode(encoded_image):
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


if __name__ == "__main__":

    input_image = Input(shape=(28,28,1))
    encoded_input = Input(shape=(7,7,32))
    
    # Load dataset 
    x_train, x_test = load_dataset()

    # Generate noisy dataset
    x_train_noise, x_test_noise = generate_noisy_data(x_train, x_test)
    
    # Generate encodede image shape
    encoded_image = encode(input_image)

    # Generate decoded image shape
    decoded_image = decode(encoded_image)

    # Instantiate utoencoder model
    ae = autoencoder(input_image, decoded_image)

    # Run training loop
    history = ae.fit(x_train_noise, x_train, epochs=5,batch_size=256,shuffle=True,validation_data=(x_test_noise, x_test))

    # Create Encoder and save it
    encoder = Model(inputs=input_image, outputs=encoded_image)
    encoder.save("models/encoder/")

    # Create Decoder and save it
    decoder1, decoder2, decoder3, decoder4, decoder5 = ae.layers[-5:]
    decoder = Model(inputs=encoded_input, outputs=decoder5(decoder4(decoder3(decoder2(decoder1(encoded_input))))))
    decoder.save("models/decoder")
    
    # Show structure of the decoder 
    decoder.summary()
    print("X TEST SHAPE")
    print(x_test_noise.shape)
    # run noisy test data through the encoder
    encoded_imgs = encoder.predict(x_test_noise)
    # run encoded noisy test image back through the decoder
    decoded_imgs = decoder.predict(encoded_imgs)

    # make sense of the shapes
    print(encoded_imgs.shape)
    print(decoded_imgs.shape)

    # display the images
    n = 100
    plt.figure(figsize=(30,6))
    for i in range(n):
      # noisy images
      ax = plt.subplot(3,n,i+1)
      plt.imshow(x_test_noise[i].reshape(28,28))
      #plt.imsave(f"./data/test_{i}.jpg", x_test_noise[i].reshape(28,28)) #saving these images for creating client data
      plt.gray()
      
      # denoised images
      ax = plt.subplot(3,n,i+1+n)
      plt.imshow(decoded_imgs[i].reshape(28,28))
      plt.gray()
      
      # original images
      ax = plt.subplot(3,n,i+1+n*2)
      plt.imshow(x_test[i].reshape(28,28))
      plt.gray()

    plt.show()
