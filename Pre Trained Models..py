from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
resnet50 = ResNet50(weights='imagenet')

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt 

img = load_img('Images Night\\frame0.jpg',
            target_size =(224,224))
img = img_to_array(img)
img = np.expand_dims(img,axis=0)
img.shape
plt.imshow(np.uint8(img[0,:,:,:]))

resnet_img = preprocess_input(img)
prediction = resnet50(resnet_img)
label = decode_predictions(prediction)
print('Predicted:',np.max(prediction))

import pydot_ng as pydot
import graphviz 
from tensorflow.keras.utils import plot_model
plot_model(resnet50)