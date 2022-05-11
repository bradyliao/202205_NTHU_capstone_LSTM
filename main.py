# https://ithelp.ithome.com.tw/articles/10206312

num_of_epochs = 40
num_of_batch_size = 32
timesteps = 60
days_forward = 0 # predicting how many days after, 0 being the immediate next
features = ['Open','High', 'Low', 'Close']
num_of_features = len(features)

# ----------------------------------------------------------------------------------------
# Import the libraries
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
total_num_training = int(total_num_data * 0.9)
total_num_testing = int(total_num_data * 0.1)


# training set
training_set = dataset.iloc[:total_num_training].values
assert num_of_features == training_set.shape[1]

# testing set
testing_set = dataset[total_num_training-timesteps:].values
assert num_of_features == testing_set.shape[1]



# ----------------------------------------------------------------------------------------
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

sc1 = MinMaxScaler(feature_range = (0, 1))
temp = sc1.fit_transform(training_set[:,0:1])


testing_set = testing_set.reshape(-1, testing_set.shape[1])
testing_set_scaled = sc.transform(testing_set) # Feature Scaling


# ----------------------------------------------------------------------------------------
# generate batch
X_train = []   #預測點的前 timesteps 天的資料
y_train = []   #預測點
for i in range(timesteps, total_num_training):
    X_train.append( training_set_scaled [i-timesteps:i, 0:num_of_features] ) # data of features
    y_train.append( training_set_scaled [i, 0] ) # data of the target value
X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN
#print(y_train.shape)

X_train = np.reshape( X_train, ( X_train.shape[0], X_train.shape[1], X_train.shape[2] ) )
assert num_of_features == X_train.shape[2]





X_test = []
for i in range(timesteps, timesteps + total_num_testing):  
    X_test.append(testing_set_scaled[i-timesteps:i, 0:num_of_features])
X_test = np.array(X_test)
assert num_of_features == X_test.shape[2]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))  # Reshape 成 3-dimension












# ----------------------------------------------------------------------------------------
# setup model
# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

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
predicted_stock_price = sc1.inverse_transform(predicted_stock_price)  # to get the original scale




# Visualising the results
real_stock_price = dataset.iloc[total_num_training:, 1:2].values
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')  # 紅線表示真實股價
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')  # 藍線表示預測股價
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
configuration = "Epochs: " + str(num_of_epochs) + " , Batch size: " + str(num_of_batch_size) + " , Timesteps: " + str(timesteps)
plt.figtext(0.6, 0.02, configuration, wrap = True, fontsize=12)
plt.legend()
plt.show()




