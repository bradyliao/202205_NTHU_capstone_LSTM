import h5py

  
num_of_epochs = 10
num_of_batch_size = 3
timesteps = 60
days_forward = 0 # predicting how many days forward, 0 being the immediate next
features = ['Open','High', 'Low', 'Close', 'Volume']
target = ['Close']
num_of_features = len(features)
test_size_portion = 0.1
dropout_rate = 0.2

validation_split_portion = 0.1

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
        different_dir = tf.logical_or(
            tf.logical_and ( tf.greater(y_true[1:] - y_true[:-1], 0.0) , tf.less(y_pred[1:] - y_pred[:-1], 0.0) ),
            tf.logical_and ( tf.less(y_true[1:] - y_true[:-1], 0.0) , tf.greater(y_pred[1:] - y_pred[:-1], 0.0) )
            )
        
        custom_loss = tf.abs ( tf.subtract(y_true[-1], y_pred[-1]) ) * ( tf.cast(different_dir[-1], tf.float32) * (3) - 1 )
        
    
        return custom_loss
    return loss



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


# Compiling
# model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["acc"])
model.compile(optimizer = 'adam', loss=test_loss(), metrics=["acc"])

model.load_weights("weights.best.hdf5")






model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")



# serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)
#saving the weights of the model
model.save_weights('FashionMNIST_weights.h5')








# for layer in model.layers: print(layer.get_config(), layer.get_weights())













weights = model.get_weights()

print(weights)