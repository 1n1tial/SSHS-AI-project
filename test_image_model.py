
import pandas as pd
from keras.applications.inception_v3 import InceptionV3


from keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.optimizers.schedules import ExponentialDecay

from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D,MaxPooling1D
from keras.layers import Activation,Dropout,Flatten,BatchNormalization
from keras.models import Sequential

import keras








batch_size = 64 
epochs = 100 
verbose = 1







input_size = 64
hdf5_path = './data/2017_2019_images_pv_processed.hdf5'

import h5py
import tensorflow as tf
import numpy as np

class generator:
    def __init__(self, file, type):
        self.file = file
        self.type = type

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            train_imarr = np.array(hf[self.type]['images_log'])
            train_pvlog = np.array(hf[self.type]['pv_log'])
            train_imarr = tf.image.convert_image_dtype(train_imarr, tf.float32)
            for i in range(len(train_imarr)):
                yield train_imarr[i], np.array([train_pvlog[i]])
            



image_train_generator = tf.data.Dataset.from_generator(
    generator(hdf5_path, 'trainval'), 
    output_types=(tf.uint16, tf.float32),
    output_shapes=(tf.TensorShape([input_size,input_size,3]), tf.TensorShape([1]))
    ).batch(batch_size)


image_test_generator = tf.data.Dataset.from_generator(
    generator(hdf5_path, 'test'), 
    output_types=(tf.uint16, tf.float32),
    output_shapes=(tf.TensorShape([input_size,input_size,3]), tf.TensorShape([1]))
    ).batch(batch_size)


# define model characteristics
num_filters = 24
kernel_size = [3,3]
pool_size = [2,2]
strides = 2
dense_size = 1024
drop_rate = 0.4

## input
### input image logs with shape (64,64,24)
x_in = Input(shape=(input_size, input_size, 3))

## 1st convolution block
x = keras.layers.Conv2D(3,[3,3],padding="same",activation='relu')(x_in)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D([2,2], 2)(x)

## 2nd convolution block
x = keras.layers.Conv2D(num_filters*2,kernel_size,padding="same",activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size, strides)(x)

## two fully connected nets
x = keras.layers.Flatten()(x)

x = keras.layers.Dense(dense_size, activation='relu')(x)
x = keras.layers.Dropout(drop_rate)(x)
x = keras.layers.Dense(dense_size, activation='relu')(x)
x = keras.layers.Dropout(drop_rate)(x)

## regression to prediction target
y_out = keras.layers.Dense(units=1)(x)

# construct the model
model = keras.Model(inputs=x_in,outputs=y_out)



        

initial_learning_rate = 0.1 

"""
lr_schedule = ExponentialDecay(
            initial_learning_rate,
                decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
"""

model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1),
    metrics=['mse'],
    loss='mse'
)

model.summary()

stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='./data/model/best_iv3_model_simple.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit(
    image_train_generator, 
    validation_data = image_test_generator,
    batch_size=batch_size,
    verbose=verbose,
    epochs=epochs,
    callbacks=[stop, best]
)


model.save('./data/model/best_iv3_model_simple.h5')