from keras import layers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import pandas as pd
import warnings

from sklearn.preprocessing import MinMaxScaler
warnings.simplefilter(action="ignore", category=FutureWarning)

X_train_structured_std = pd.read_csv('./data/X_train_structured.csv')
X_test_structured_std = pd.read_csv('./data/X_test_structured.csv')

min_max_scaler = MinMaxScaler()
X_train_structured_std = min_max_scaler.fit_transform(X_train_structured_std)
X_test_structured_std = min_max_scaler.transform(X_test_structured_std)

y_train_std = pd.read_csv('./data/y_train_structured.csv')
y_test_std = pd.read_csv('./data/y_test_structured.csv')
    

# declare the final model inputs and outputs
final_model = Sequential(
    name='structured_data_model',
    layers=[
        Dense(X_train_structured_std.shape[1], 
                 activation='relu', 
                 kernel_initializer='he_normal',
                 input_shape=(X_train_structured_std.shape[1],),
                 name='dense_input'),
        BatchNormalization(),
        Dense(100, activation='relu', name='dense_hidden1'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='linear', name='output_layer')
    ]
)

print(final_model.summary())

initial_learning_rate = 0.2
# lr_schedule = ExponentialDecay(
#     initial_learning_rate,
#         decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)

stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode='min', verbose=1)
best = ModelCheckpoint(filepath='./data/model/best_structured_model.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# compile the model
final_model.compile(optimizer=Adam(learning_rate=initial_learning_rate, epsilon=1), 
                    loss="mean_squared_error", 
                    metrics=[MeanSquaredError()])

# plot_model(final_model, show_shapes=True, to_file='model1_structured.png')


results = final_model.fit(
            X_train_structured_std, y_train_std,
            epochs=500,
            batch_size = 250,
            callbacks=[stop, best],
            validation_data=([X_test_structured_std, y_test_std])
            )

final_model.save('./data/model/best_structured_model.h5')
