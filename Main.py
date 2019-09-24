## Main Program for Low Light Image Processing
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2 
import skimage
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import image
from os import listdir
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# dimensions of our images.
img_width, img_height = 28, 28

train_data_dir = 'Images Night'
validation_data_dir = 'Images Night'
nb_train_samples = 6
nb_validation_samples = 5
epochs = 2
batch_size = 16

if backend.image_data_format() == 'channels_first':
    input_img =Input(shape=(3,img_width, img_height))
else:
    input_img = Input(shape=(img_width, img_height, 3))  # adapt this if using `channels_first` image data format

loaded_images = list()
## Model Architecture
## -- Encoding
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
loaded_images.append(x)
x = MaxPooling2D((2, 2), padding='same')(x)
loaded_images.append(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
loaded_images.append(x)
x = MaxPooling2D((2, 2), padding='same')(x)
loaded_images.append(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
loaded_images.append(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
loaded_images.append(x)
## -- Dencoding
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
loaded_images.append(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
loaded_images.append(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
loaded_images.append(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
loaded_images.append(x)
## -- Model created
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

## Take images from folder 
loaded_images = list()
for filename in listdir('Images Night'):
	# load image
	img_data = image.imread('Images Night\\' + filename)
	# store loaded image
	loaded_images.append(img_data)
	print('> loaded %s %s' % (filename, img_data.shape))

## Resize the images 
loaded_images = [skimage.transform.resize(image, (28,28,3)) for image in loaded_images]
x_train=np.asarray(loaded_images[:5])
x_test=np.asarray(loaded_images[1])

## Giving Input Data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 
autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=25,
                shuffle=True,
                validation_data=(x_train, x_train),)
decoded_imgs = autoencoder.predict(x_test)
## Check Image 
# create a data generator
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory('.',classes=['Images Night'],target_size=(28, 28))

img_path = 'Images Night'
img = list() 
for filename in listdir('Images Night'):
    img_tensor=image.load_img('Images Night\\' + filename, target_size=(28, 28)) 
    img_tensor = image.img_to_array(img_tensor)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    img.append(img_tensor)
plt.imshow(img_tensor[0])
plt.show()
​
print(img_tensor.shape)


## check intermediat result
layer_outputs = [layer.output for layer in autoencoder.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=autoencoder.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input 
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
## Check images 
n=5
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i +n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()