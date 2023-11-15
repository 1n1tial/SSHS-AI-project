
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




base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(input_size*4, input_size*4, 3)))

for layer in base_model.layers:
    layer.trainable=False
    
img_input = Input(shape=(input_size, input_size, 3))

upsamp1 = tf.keras.layers.UpSampling2D((2,2))(img_input)
upsamp2 = tf.keras.layers.UpSampling2D((2,2))(upsamp1)
x = base_model(upsamp2, training=False)
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
output_cnn = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense_hidden')(x) #256
last_layer = Dense(1, activation='linear')(output_cnn)

model = Model(inputs=img_input, outputs=last_layer)


        

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