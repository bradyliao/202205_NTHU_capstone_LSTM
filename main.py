# https://ithelp.ithome.com.tw/articles/10206312

num_of_epochs = 40
num_of_batch_size = 32
timesteps = 60
days_forward = 10 # predicting how many days forward, 0 being the immediate next
features = ['Open','High', 'Low', 'Close']
target = ['Open']
num_of_features = len(features)
test_size_portion = 0.1

# ----------------------------------------------------------------------------------------
# Import the libraries
from turtle import shape
import numpy as np
import matplotlib.pyplot as plt  # for ploting results
import pandas as pd

# ----------------------------------------------------------------------------------------
# Import the dataset
dataset_in = pd.read_csv('googl.us.csv')

# extract features
X_raw = dataset_in[features]
y_raw = dataset_in[target]

# calculate total num of data
total_num_data = len(X_raw)


from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size = test_size_portion, shuffle = False)


from sklearn.preprocessing import MinMaxScaler
scale_X_train = MinMaxScaler(feature_range = (0, 1))
X_train_scale = scale_X_train.fit_transform(X_train_raw)

scale_y_train = MinMaxScaler(feature_range = (0, 1))
y_train_scale = scale_y_train.fit_transform(y_train_raw)

scale_X_test = MinMaxScaler(feature_range = (0, 1))
X_test_scale = scale_X_test.fit_transform(X_test_raw)

scale_y_test = MinMaxScaler(feature_range = (0, 1))
y_test_scale = scale_y_test.fit_transform(y_test_raw)


X_train = []   #預測點的前 timesteps 天的資料
y_train = []   #預測點
for i in range(timesteps, len(X_train_scale) - days_forward):
    X_train.append( X_train_scale [ i - timesteps : i , 0 : num_of_features ] ) # data of features
    y_train.append( y_train_scale [ i + days_forward , 0] ) # data of the target value

X_test = []
y_test = []
for i in range(timesteps, len(X_test_scale) - days_forward):
    X_test.append( X_test_scale [ i - timesteps : i , 0 : num_of_features ] ) # data of features
    y_test.append( y_test_scale [ i + days_forward , 0] ) # data of the target value


X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)   # 轉成numpy array的格式，以利輸入 RNN


X_train = np.reshape( X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2] ) )
assert num_of_features == X_train.shape[2]

assert num_of_features == X_test.shape[2]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # Reshape 成 3-dimension




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
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=["acc"])


# ----------------------------------------------------------------------------------------
# run model
history = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = num_of_epochs, batch_size = num_of_batch_size)











# ----------------------------------------------------------------------------------------



predicted_stock_price = model.predict(X_test)
predicted_stock_price = scale_y_test.inverse_transform(predicted_stock_price)  # to get the original scale
temp = []
for i in range(0, timesteps):
    temp.append(None)
for i in predicted_stock_price:
    temp.append(i[0])

#print(temp)
#print(y_test_raw)

'''

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

'''

# Visualising the results
real_stock_price = scale_y_test.inverse_transform(y_test_scale)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plt.plot(temp, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps) + " , \nDays forward: " + str(days_forward) + " , Test size: " + str(test_size_portion)
plt.figtext(0.6, 0.02, configuration, wrap = True, fontsize=12)
plt.legend()
plt.show()







'''


fig, axd = plt.subplot_mosaic([['left', 'right'],['bottom', 'bottom']], constrained_layout=True)

axd['left'].plot(history.history['loss'])
axd['left'].plot(history.history['val_loss'])
#axd['left'].title('model loss')
axd['left'].ylabel('loss')
axd['left'].xlabel('epoch')
axd['left'].legend(['Train', 'Validation'], loc='upper left')

axd['right'].plot(history.history['acc'])
axd['right'].plot(history.history['val_acc'])
axd['right'].title('model accuracy')
axd['right'].ylabel('accuracy')
axd['right'].xlabel('epoch')
axd['right'].legend(['Train', 'Validation'], loc='upper left')

real_stock_price = scale_y_test.inverse_transform(y_test_scale)
axd['bottom'].plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
axd['bottom'].plot(temp, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
axd['bottom'].title('Google Stock Price Prediction')
axd['bottom'].xlabel('Time')
axd['bottom'].ylabel('Google Stock Price')
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps)
axd['bottom'].figtext(0.6, 0.02, configuration, wrap = True, fontsize=12)
axd['bottom'].legend()
plt.show()

'''