import tensorflow as tf
try:
    from tensorflow import keras as keras
    print ('load keras from tensorflow package')
except:
    print ('update your tensorflow')
from tensorflow.keras import models
from tensorflow.keras import layers

import opts

class Model():
    def __init__(self):
        print ('Building model...')

    '''Create the architecture'''
    def create_model(self, x_shape, num_classes):
        # Specify inputs (size is given in opts.py file)
        x_in = layers.Input(shape=x_shape, name='x_in')

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
        # 1) Flatten last conv layer
        flat = layers.Flatten(data_format='channels_last')(conv5)
        # 2) Add two fully connected layers
        # TODO: figure out dimensionality
        fc1 = layers.Dense(opts.dense_layer_size, activation='relu')(flat)
        fc2 = layers.Dense(opts.dense_layer_size, activation='relu')(fc1)
        # 3) Add a classification output layer
        classifier = layers.Dense(num_classes, activation='softmax')(fc2)

        # Paired cell inpainting output
        model = models.Model(inputs=x_in, outputs=classifier)

        return model