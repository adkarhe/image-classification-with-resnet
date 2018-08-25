from __future__ import print_function, division
import os, json, csv
import numpy as np
import pandas as pd
from glob import glob
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Input, Activation, merge
from keras.layers.core import Dense, Dropout, Lambda, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import get_file, to_categorical
from keras.optimizers import SGD, Adam, RMSprop


resnet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

def vgg_preprocess(x):

	x = x - resnet_mean
	return x[:, ::-1]


class Resnet:

	def __init__(self):
		self.WEIGHT_FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'
		self.weight_filename = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

		self.CLASS_FILE_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
		self.create()
		self.compile()


	def identity_block(self, input_tensor, kernel_size, filters, stage, block):

		filters1, filters2, filters3 = filters
		bn_axis = 3

		conv_name_base = 'res' + str(stage) + block + '_branch'
		bn_name_base = 'bn' + str(stage) + block + '_branch'

		x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		x = layers.add([x, input_tensor])
		x = Activation('relu')(x)

		return x

	def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

		filters1, filters2, filters3 = filters
		bn_axis = 3
		
		conv_name_base = 'res' + str(stage) + block + '_branch'
		bn_name_base = 'bn' + str(stage) + block + '_branch'

		x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		x = Activation('relu')(x)

		x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
		x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
		shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

		x = layers.add([x, shortcut])
		x = Activation('relu')(x)
		
		return x


	def create(self):
		bn_axis = 3

		img_input = Input(shape=(224, 224, 3))

		x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
		x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
		x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
		x = Activation('relu')(x)
		x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
		x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
		x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

		x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
		x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
		x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
		x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

		x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
		x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
		x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
		x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
		x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
		x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

		x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
		x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
		x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

		x = AveragePooling2D((7, 7), name='avg_pool')(x)
		x = Flatten()(x)
		x = Dense(1000, activation='softmax', name='fc1000')(x)

		self.model = Model(img_input, x, name='resnet50')

		self.model.load_weights(get_file(self.weight_filename, self.WEIGHT_FILE_PATH + self.weight_filename, cache_subdir='models'))
		
	def compile(self):
		self.model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])