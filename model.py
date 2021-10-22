
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.utils import np_utils
#from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras import backend as K
import pickle
import random

DATADIR = "/Users/bidhanbashyal/Acoustic/TrainingData"

CATEGORIES = ["Airport", "Bus","Mall","Metro", "Metrostation",
"Park", "Pedestrain", "Square",
"StreetTraffic", "Tram"]

for category in CATEGORIES:
path = os.path.join(DATADIR,category)
for img in os.listdir(path):
img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
plt.imshow(img_array)
plt.show()

    break 
break  
IMG_SIZE1 =256
IMG_SIZE2 = 256
#IMG_SIZE1=338
#IMG_SIZE2=220

new_array = cv2.resize(img_array, (IMG_SIZE1, IMG_SIZE2))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
for category in CATEGORIES:

    path = os.path.join(DATADIR,category)  
    class_num = CATEGORIES.index(category)  

    for img in tqdm(os.listdir(path)):  
        try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE) 
            new_array = cv2.resize(img_array, (IMG_SIZE1, IMG_SIZE2)) 
            training_data.append([new_array, class_num])  
        except Exception as e:  
            pass
random.shuffle(training_data)

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
X=X/255.0

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=X.shape[1:]))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
sgd = SGD(lr=0.1, decay=1e-6, nesterov=True)

model.compile(loss='sparse_categorical_crossentropy',
optimizer=sgd,
metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
