import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.lines as mlines
import os
tf.random.set_seed(0)

Stock = input('Enter ticker : ')

# download the data
df = yf.download(tickers=[Stock], period='1y')
y = df['Close'].fillna(method='ffill') # A way to "plug" any holes that are in the data, by filling null values with NaN
y = y.values.reshape(-1, 1) # reshape the data -1 making the number of rows unknown and 1 making the amount of columns 1, so you get a table-like array with undefined number of rows and 1 column

# scale the data
# Scaling the data is a way to normalize the data and boost the performance of the neural network
# Scaling makes the data much easier to process and it also is a way of sorting the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# Define the input and output sequences
n_lookback = 60  # length of input sequences (lookback period), in this case the number of days the model will use to train
n_forecast = 20  # length of output sequences (forecast period), in this case the number of days the model will try to predict

# Create blank lists for data to be stored in
X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# Define the model type and build the model
model = Sequential()
# Create a model variable that contains a model from keras.models
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
# Input shape is the dimensions of our inputs, no. of samples (batchsize, in this case n_lookback is 60, because we want the last 60 days price, and we add the dimension of this data which is 1, only the price)
# Add an LSTM layer to our model
# Units = The layer will contain multiple parallel LSTM units, structurally identical but each eventually "learning to remember" some different thing.
# You must set return_sequences=True when stacking LSTM layers so that the second LSTM layer has a three-dimensional sequence input.
model.add(LSTM(units=50))
# Add another LSTM layer to our model
# LSTM stands for Long Short Term Memory comes under RNN. LSTM has mostly used the time or sequence-dependent behavior example texts, stock prices, electricity.
model.add(Dense(n_forecast))
# A dense layer revcieves input from all other neurons, the dense layer helps in changing the dimensionality of the output from the preceding layer so that the model can easily define the relationship between the values of the data in which the model is working.
# Dense has a units variable that needs to be filled, here it is n_forecast or 60


model.compile(loss='mean_squared_error', optimizer='adam')
# loss specified in the model.compile ensures that the MSE between Y_Pred and Y_Actual is reduced
# optimizer is a function or an algorithm that modifies the attributes of the neural network, such as weights and learning rates. Thus, it helps in reducing the overall loss and improving accuracy
model.fit(X, Y, epochs=100, steps_per_epoch=2, batch_size=32, verbose=2)
# verbose = display method of fitting the model 1= progressbar, 2= each epoch seperate
# batch size = the number of training examples in one forward/backward pass. It will take x samples every epoch, x = number of samples
# one epoch = one forward pass and one backward pass of all the training examples
# steps_pe_epoch = ryou can specify your own batch size. In this case, say batch_size = 20. As a result, you can set your steps_per_epoch = 100/20 = 5 because in this way you can make use of the complete training data for each epoch.


# generate the forecasts
X_ = y[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past._append(df_future).set_index('Date')

# plot the results
blue_line = mlines.Line2D([], [], color='blue', label='Current')
reds_line = mlines.Line2D([], [], color='red', label='Prediction')
plt.style.use('fivethirtyeight')
plt.figure(figsize=(16,8))
plt.title(Stock+' LSTM model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Val', fontsize=18)
plt.legend(handles=[blue_line, reds_line])
plt.plot(results)
current_absolute_path = os.path.dirname(os.path.abspath(__file__))+"\\LSTM_Stocks_out" # Figures out the absolute path for you in case your working directory moves around.
plot_file = str(Stock)+'_'+str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))+'.png'
plt.savefig(os.path.join(current_absolute_path, plot_file))
plt.show()
