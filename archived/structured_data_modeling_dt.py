from sklearn.tree import DecisionTreeRegressor
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import pandas as pd
import warnings
import sklearn

from sklearn.preprocessing import MinMaxScaler

X_train_structured_std = pd.read_csv('./data/X_train_structured.csv')
X_test_structured_std = pd.read_csv('./data/X_test_structured.csv')

min_max_scaler = MinMaxScaler()
X_train_structured_std = min_max_scaler.fit_transform(X_train_structured_std)
X_test_structured_std = min_max_scaler.transform(X_test_structured_std)

y_train_std = pd.read_csv('./data/y_train_structured.csv')
y_test_std = pd.read_csv('./data/y_test_structured.csv')

# model = DecisionTreeRegressor(random_state=44)
# model.fit(X_train_structured_std, y_train_std)
# predictions = model.predict(X_test_structured_std)
# print(sklearn.metrics.mean_squared_error(y_test_std, predictions))

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train_structured_std, y_train_std)
predictions = lin_reg.predict(X_test_structured_std)
print(sklearn.metrics.mean_squared_error(y_test_std, predictions))