import numpy as np
import pandas as pd
import pickle as pkl
from attention import Attention
import keras
from sklearn import metrics

print('loading...')
final_model = keras.models.load_model("/data/model/best_final_model_weights_only.h5", compile=False, custom_objects={'Attention': Attention})
fine_tuned_inceptionv3 = keras.models.load_model('/data/model/best_iv3_model_simple.hdf5', compile=False)
fine_tuned_structured = keras.models.load_model('/data/model/best_structured_model.hdf5', compile=False)
print('loaded DONE!')

X_train_structured_std = pd.read_csv('/data/X_train_structured.csv')
X_test_structured_std = pd.read_csv('/data/X_test_structured.csv')

file = open("/data/model/X_train_image_array.pkl",'rb')
X_train_image_array = pkl.load(file)
file.close()

file = open("/data/model/X_test_image_array.pkl",'rb')
X_test_image_array = pkl.load(file)
file.close()

y_train_std = pd.read_csv('/data/y_train_std.csv', sep = ';')
y_test_std = pd.read_csv('/data/y_test_std.csv', sep = ';')


y_pred_img = fine_tuned_inceptionv3.predict(X_test_image_array)
y_pred_structured = fine_tuned_structured.predict(X_test_structured_std)
y_pred_all = final_model.predict([X_test_image_array, X_test_structured_std])

y_pred_compare = pd.DataFrame()
y_pred_compare['Y_TRUE'] = y_test_std['pv_log']
y_pred_compare['Y_PRED_STRUCTURED'] = y_pred_structured
y_pred_compare['Y_PRED_IMG'] = y_pred_img
y_pred_compare['Y_PRED_HYBRID_ALL'] = y_pred_all


print('R2')
print('STRUCTURED', metrics.r2_score(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED']))
print('IMG', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG']))
print('HYBRID_ALL', metrics.r2_score( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL']))
print('---------------')
print('RMSE')
print('STRUCTURED', np.sqrt(metrics.mean_squared_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED'])))
print('IMG', np.sqrt(metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG'])))
print('HYBRID_ALL',np.sqrt( metrics.mean_squared_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL'])))
print('---------------')
print('MAE')
print('STRUCTURED', metrics.mean_absolute_error(y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_STRUCTURED']))
print('IMG', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_IMG']))
print('HYBRID_ALL', metrics.mean_absolute_error( y_pred_compare['Y_TRUE'], y_pred_compare['Y_PRED_HYBRID_ALL']))