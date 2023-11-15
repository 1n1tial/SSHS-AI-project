
from tensorflow import keras



from keras.layers import BatchNormalization, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, Activation, Dropout, Dense, Input, Multiply
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

import keras


fine_tuned_inceptionv3 = keras.models.load_model('/data/model/best_iv3_model_simple.hdf5', compile=False)
fine_tuned_structured = keras.models.load_model('/data/model/best_structured_model.hdf5', compile=False)

fine_tuned_inceptionv3_layer = Model(inputs=fine_tuned_inceptionv3.input, outputs=fine_tuned_inceptionv3.layers[-2].output)
fine_tuned_structured_layer = Model(inputs=fine_tuned_structured.input, outputs=fine_tuned_structured.layers[-2].output)

for layer in fine_tuned_structured_layer.layers:
    layer._name = layer.name + '_structured'


for layer in fine_tuned_inceptionv3_layer.layers:
    layer._name = layer.name + '_img'
    
x = keras.layers.concatenate([fine_tuned_inceptionv3_layer.output,
                        fine_tuned_structured_layer.output])
x = Dense(50, activation='relu', name='dense_hidden_final')(x)
last_layer = Dense(1, activation='linear')(x)


model = keras.Model(inputs=[fine_tuned_inceptionv3_layer.input,
                            fine_tuned_structured_layer.input
                           ], 
                    outputs=[last_layer])

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

from keras.utils import plot_model
plot_model(model, to_file="final_model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    dpi=350)

model.save('/data/model/best_final_model.h5')

print(model.summary())
