
import pandas as pd
import numpy as np
import pickle as pkl

from keras.layers import BatchNormalization, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, Activation, Dropout, Dense, Input, Multiply
from keras.models import Sequential, Model
from keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History, LearningRateScheduler
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

from tensorflow.keras.utils import load_img, img_to_array

from attention import Attention
import keras

X_train_structured_std = pd.read_csv('./data/X_train_structured.csv', sep = ',')
X_test_structured_std = pd.read_csv('./data/X_test_structured.csv', sep = ',')
print(X_train_structured_std.shape)

X_train_image_array = np.load('./data/X_train_image.npy')
X_test_image_array =  np.load('./data/X_test_image.npy')

y_train_std = pd.read_csv('./data/y_train_structured.csv', sep = ',')
y_test_std = pd.read_csv('./data/y_test_structured.csv', sep = ',')



fine_tuned_inceptionv3 = keras.models.load_model('./data/model_archive/best_iv3_model_simple.hdf5', compile=False)
fine_tuned_structured = keras.models.load_model('./data/model/best_structured_model.hdf5', compile=False)

fine_tuned_inceptionv3_layer = Model(inputs=fine_tuned_inceptionv3.input, outputs=fine_tuned_inceptionv3.layers[-2].output)
fine_tuned_structured_layer = Model(inputs=fine_tuned_structured.input, outputs=fine_tuned_structured.layers[-2].output)

for layer in fine_tuned_structured_layer.layers:
    layer._name = layer.name + '_structured'
    
for layer in fine_tuned_inceptionv3_layer.layers:
    layer._name = layer.name + '_img'
    
x = keras.layers.concatenate([fine_tuned_inceptionv3_layer.output, 
                        fine_tuned_structured_layer.output])
x = Dense(1024, activation='relu', name='dense_hidden_final')(x)
last_layer = Dense(1, activation='linear')(x)


model = keras.Model(inputs=[fine_tuned_inceptionv3_layer.input, 
                            fine_tuned_structured_layer.input
                           ], 
                    outputs=[last_layer], name='hybrid_model')

initial_learning_rate = 0.1

'''
lr_schedule = ExponentialDecay(
            initial_learning_rate,
                decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
'''

model.compile(optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1), loss="mean_squared_error", metrics=[ MeanSquaredError()])
print(model.summary())

print('saving...')
model.save('./data/model/best_final_model_simple.h5')
print('saved DONE!')

print('loading...')
reconstructed_model = keras.models.load_model("./data/model/best_final_model_simple.h5", compile=False, custom_objects={'Attention': Attention})
print('loaded DONE!')

reconstructed_model.compile(optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1), loss="mean_squared_error", metrics=[ MeanSquaredError()])
print(reconstructed_model.summary())

stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='./data/model/best_final_model_weights_only.h5', 
                       save_best_only=True, 
                       save_weights_only=False, 
                       monitor='val_loss', 
                       mode='min', verbose=1)

results = reconstructed_model.fit([X_train_image_array,
                    X_train_structured_std],
                    y_train_std,                   
                    epochs=500,
                    batch_size = 250,
                    validation_data=([X_test_image_array, 
                                       X_test_structured_std], 
                                      y_test_std),
                    callbacks=[stop, best],
                    )

results.save('./data/model/final_hybrid_model.h5')