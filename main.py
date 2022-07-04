# https://ithelp.ithome.com.tw/articles/10206312

num_of_epochs = 10
num_of_batch_size = 32
timesteps = 60
days_forward = 0 # predicting how many days forward, 0 being the immediate next
features = ['Open','High', 'Low', 'Close', 'Volume']
target = ['Open']
num_of_features = len(features)
test_size_portion = 0.1
dropout_rate = 0.2

# ----------------------------------------------------------------------------------------
# Import the libraries
from turtle import shape
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# process data
# Import the dataset
dataset_in = pd.read_csv('googl.us.csv')

# extract features
X_raw = dataset_in[features]
# extract targets
y_raw = dataset_in[target]

# calculate total num of data
total_num_data = len(X_raw)


# split training / testing
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size = test_size_portion, shuffle = False)


# scaling
from sklearn.preprocessing import MinMaxScaler
scale_X_train = MinMaxScaler(feature_range = (0, 1))
X_train_scale = scale_X_train.fit_transform(X_train_raw)

scale_y_train = MinMaxScaler(feature_range = (0, 1))
y_train_scale = scale_y_train.fit_transform(y_train_raw)

scale_X_test = MinMaxScaler(feature_range = (0, 1))
X_test_scale = scale_X_test.fit_transform(X_test_raw)

scale_y_test = MinMaxScaler(feature_range = (0, 1))
y_test_scale = scale_y_test.fit_transform(y_test_raw)


# generate epochs
X_train = []   #預測點的前 timesteps 天的資料
y_train = []   #預測點
for i in range(timesteps, len(X_train_scale) - days_forward):
    X_train.append( X_train_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_train.append( y_train_scale [ (i + days_forward) , 0] ) # data of the target value

X_test = []
y_test = []
for i in range(timesteps, len(X_test_scale) - days_forward):
    X_test.append( X_test_scale [ (i - timesteps) : i , 0 : num_of_features ] ) # data of features
    y_test.append( y_test_scale [ (i + days_forward) , 0] ) # data of the target value


# convert to numpy array
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)   # 轉成numpy array的格式，以利輸入 RNN


# reshape data X to 3-dimension for model
assert num_of_features == X_train.shape[2]
X_train = np.reshape( X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2] ) )

assert num_of_features == X_test.shape[2]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))




# ----------------------------------------------------------------------------------------
# setup model
# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import BatchNormalization


# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
assert num_of_features == X_train.shape[2]
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(dropout_rate))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(dropout_rate))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(dropout_rate))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(dropout_rate))

# Adding the output layer
model.add(Dense(units = 1))



import keras.backend as K
import tensorflow as tf

def test_loss():
    def loss(y_true, y_pred):
        # y_diff = tf.square(y_pred - y_true)
        # custom_loss = 0
        # for i in range(1, num_of_batch_size):
        #     custom_loss = custom_loss + ( (y_true[i] - y_true[i-1] ) * ( y_pred[i] - y_pred[i-1] ) ) * y_diff[i]
        
        # # calculating squared difference between target and predicted values 
        # loss = K.square(y_pred - y_true)  # (batch_size, 2)
        # print(loss)
        # # multiplying the values with weights along batch dimension
        # loss = loss * [0.3, 0.7]          # (batch_size, 2)
        # # summing both loss values along batch dimension 
        # custom_loss = K.sum(loss, axis=1)        # (batch_size,)
        
        
        
        # # https://towardsdatascience.com/customize-loss-function-to-make-lstm-model-more-applicable-in-stock-price-prediction-b1c50e50b16c
        # #extract the "next day's price" of tensor
        # y_true_next = y_true[1:] 
        # y_pred_next = y_pred[1:]
        # #extract the "today's price" of tensor
        # y_true_tdy = y_true[:-1] 
        # y_pred_tdy = y_pred[:-1]
        
        #  #substract to get up/down movement of the two tensors
        # y_true_diff = tf.subtract(y_true_next, y_true_tdy) 
        # y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)
        # #create a standard tensor with zero value for comparison
        # standard = tf.zeros_like(y_pred_diff)
        # #compare with the standard; if true, UP; else DOWN
        # y_true_move = tf.greater_equal(y_true_diff, standard) 
        # y_pred_move = tf.greater_equal(y_pred_diff, standard)
        
        # #find indices where the directions are not the same
        # condition = tf.not_equal(y_true_move, y_pred_move) 
        # indices = tf.where(condition)
        # ones = tf.ones_like(indices) 
        # indices = tf.add(indices, ones)

        # direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32') 
        # updates = K.cast(tf.ones_like(indices), dtype='float32')
        # alpha = 1000
        # direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices,  alpha*updates )
        # custom_loss = K.mean(tf.multiply(K.square(y_true - y_pred), direction_loss), axis=-1)
        
        
        different_dir = tf.logical_or(
        tf.logical_and ( tf.greater(y_true[1:] - y_true[:-1], 0.0) , tf.less(y_pred[1:] - y_pred[:-1], 0.0) ),
        tf.logical_and ( tf.less(y_true[1:] - y_true[:-1], 0.0) , tf.greater(y_pred[1:] - y_pred[:-1], 0.0) )
        )
        
        custom_loss = tf.keras.metrics.mean_squared_error(y_true, y_pred) * ( tf.cast(different_dir, tf.float32) * (3) - 1 )
        
        
        
        
        
        return custom_loss
        
    
    return loss



# Compiling
# model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["acc"])
model.compile(optimizer = 'adam', loss=test_loss(), metrics=["acc"])


# ----------------------------------------------------------------------------------------
# run model
history = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = num_of_epochs, batch_size = num_of_batch_size)


# ----------------------------------------------------------------------------------------
# predict
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scale_y_test.inverse_transform(predicted_stock_price)  # to get the original scale


# ----------------------------------------------------------------------------------------
# shift right timesteps

predicted_stock_price_shifted = []
for i in range(0, timesteps):
    predicted_stock_price_shifted.append(None)
for i in predicted_stock_price:
    predicted_stock_price_shifted.append(i[0])


# ----------------------------------------------------------------------------------------
# visualize 

import matplotlib.pyplot as plt  # for ploting results


plot_loss = plt.subplot2grid((2, 2), (0, 0))                #, colspan=2)
plot_accu = plt.subplot2grid((2, 2), (0, 1))                #, rowspan=3, colspan=2)
plot_test = plt.subplot2grid((2, 2), (1, 0), colspan=2)     #, rowspan=2)

# Visualising the loss
plot_loss.plot(history.history['loss'])
plot_loss.plot(history.history['val_loss'])
plot_loss.set_title('model loss')
plot_loss.set_ylabel('loss')
plot_loss.set_xlabel('epoch')
plot_loss.legend(['Train', 'Validation'], loc='upper left')


# Visualising the accuracy
plot_accu.plot(history.history['acc'])
plot_accu.plot(history.history['val_acc'])
plot_accu.set_title('model accuracy')
plot_accu.set_ylabel('accuracy')
plot_accu.set_xlabel('epoch')
plot_accu.legend(['Train', 'Validation'], loc='upper left')


# Visualising the test results
real_stock_price = scale_y_test.inverse_transform(y_test_scale)
plot_test.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plot_test.plot(predicted_stock_price_shifted, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plot_test.set_title('Google Stock Price Prediction')
plot_test.set_xlabel('Time', loc='left')
plot_test.set_ylabel('Google Stock Price')
plot_test.legend()

# turn features to 1 string
feature_all = ''
for i in features:
    feature_all = feature_all + i + ", "

# Packing all the plots and displaying them
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps) \
    + " , Days forward: " + str(days_forward) + " , Test size: " + str(test_size_portion) + " , Dropout rate: " + str(dropout_rate) \
    + "\nFeatures: " + feature_all + "Target: " + target[0]
plt.figtext(0.9, 0.01, configuration, horizontalalignment = 'right', verticalalignment = 'bottom', wrap = True, fontsize = 12)
plt.tight_layout()
plt.show()