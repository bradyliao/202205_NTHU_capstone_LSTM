# https://ithelp.ithome.com.tw/articles/10206312

num_of_epochs = 40
num_of_batch_size = 32
timesteps = 60
days_forward = 20 # predicting how many days forward, 0 being the immediate next
features = ['Open','High', 'Low', 'Close']
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
dataset = dataset_in[features]

# calculate total num of data
total_num_data = len(dataset)

# ----------------------------------------------------------------------------------------
# new



X = []   #預測點的前 timesteps 天的資料
y = []   #預測點
for i in range(timesteps, total_num_data - days_forward):
    X.append( dataset.iloc [ i - timesteps : i , 0 : num_of_features ]) # data of features
    y.append( dataset.iloc [ i + days_forward , 0:1] ) # data of the target value


from sklearn.preprocessing import MinMaxScaler

scale_X = MinMaxScaler(feature_range = (0, 1))
X = scale_X.fit_transform(X)

scale_y = MinMaxScaler(feature_range = (0, 1))
y = scale_y.fit_transform(y)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_portion, shuffle = False)

print(X_test)


X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)   # 轉成numpy array的格式，以利輸入 RNN


print(y_test)


'''
from sklearn.preprocessing import MinMaxScaler

scale_X_train = MinMaxScaler(feature_range = (0, 1))
X_train = scale_X_train.fit_transform(X_train)

scale_y_train = MinMaxScaler(feature_range = (0, 1))
y_train = scale_y_train.fit_transform(y_train)

scale_X_test = MinMaxScaler(feature_range = (0, 1))
X_test = scale_X_test.fit_transform(X_test)

scale_y_test = MinMaxScaler(feature_range = (0, 1))
y_test = scale_y_test.fit_transform(y_test)







print(y_train.shape)

X_train = np.reshape( X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2] ) )
assert num_of_features == X_train.shape[2]


# new
# ----------------------------------------------------------------------------------------

'''










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






# ----------------------------------------------------------------------------------------



predicted_stock_price = model.predict(X_test)
predicted_stock_price = scale_y.inverse_transform(predicted_stock_price)  # to get the original scale




# Visualising the results
real_stock_price = scale_y.inverse_transform(y_test)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps)
plt.figtext(0.6, 0.02, configuration, wrap = True, fontsize=12)
plt.legend()
plt.show()


