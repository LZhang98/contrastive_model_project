'''
Architecture for paired-cell inpainting.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu

Modified from:
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
'''
import tensorflow as tf
try:
    from tensorflow.contrib import keras as keras
    print ('load keras from tensorflow package')
except:
    print ('update your tensorflow')
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers

class Pair_Model():
    def __init__(self):
        print ('Building model...')

    '''Create the architecture'''
    def create_model(self, x_shape, y_shape):
        # Specify inputs (size is given in opts.py file)
        x_in = layers.Input(shape=x_shape, name='x_in')
        y_rfp = layers.Input(shape=y_shape, name='y_rfp')

        # First two conv layers of source cell encoder
        conv1 = layers.Conv2D(96, (3, 3), activation='relu', padding='same', name='conv1_1')(x_in)
        conv1 = layers.BatchNormalization()(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
        conv2 = layers.BatchNormalization()(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Last three conv layers of source cell encoder
        conv3 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
        conv3 = layers.BatchNormalization()(conv3)
        conv4 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4_1')(conv3)
        conv4 = layers.BatchNormalization()(conv4)
        conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5_1')(conv4)
        conv5 = layers.BatchNormalization()(conv5)

        # Convert source cell encorder into classifier
        flat = layers.Flatten(data_format='channels_last')(conv5)

        # Decoder layers
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6_1')(conv5)
        conv7 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv7_1')(conv6)
        conv8 = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='conv8_1')(conv7)
        up_conv9 = layers.UpSampling2D(size=(2, 2))(conv8)
        conv9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv9_1')(up_conv9)
        up_conv10 = layers.UpSampling2D(size=(2, 2))(conv9)
        conv10 = layers.Conv2D(96, (3, 3), activation='relu', padding='same', name='conv10_1')(up_conv10)
        conv10 = layers.Conv2D(1, (1, 1), activation=None, name='y_gfp')(conv10)

        # Paired cell inpainting output
        model = models.Model(inputs=[x_in, y_rfp], outputs=conv10)

        return model