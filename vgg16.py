import keras
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Lambda, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import get_file, to_categorical
import numpy as np
import pandas as pd

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def vgg_preprocess(x):

	x = x - vgg_mean
	return x[:, ::-1]

class Vgg16:

	def __init__(self):
		self.WEIGHT_FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
		self.weight_filename = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'

		self.CLASS_FILE_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
		self.create()
		self.compile()


	def create(self):
		self.model = Sequential()

		self.model.add(Lambda(vgg_preprocess, input_shape=(224, 224, 3)))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(64, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(64, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(128, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(128, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(256, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(256, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(256, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(ZeroPadding2D((1, 1)))
		self.model.add(Conv2D(512, (3 , 3)))
		self.model.add(Activation('relu'))
		self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		self.model.add(Flatten())

		self.model.add(Dense(4096, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(4096, activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(1000, activation='softmax'))

		self.model.load_weights(get_file(self.weight_filename, self.WEIGHT_FILE_PATH + self.weight_filename, cache_subdir='models'))


	def compile(self):
		self.model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])