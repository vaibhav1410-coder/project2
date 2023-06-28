from numpy import array
import numpy
import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import pandas_datareader as web
from keras.models import load_model
import streamlit as st


st.title('STOCK TREND PREDICTION')
user_input = st.text_input('ENTER STOCK SYMBOL', 'TITAN.NS')

yf.pdr_override()

# ds = pd.read_csv("./data.csv")
ds = pdr.get_data_yahoo(
    user_input, start=dt.datetime(2022, 1, 1), end=date.today())
# print(ds)

ds2 = ds.iloc[::-1]

# Describing data
st.subheader('Data from Past One Year')
st.write(ds2)


# closing price vs time Chart
fig0 = plt.figure(figsize=(12, 6))
st.subheader('Closing Price vs Time chart')
ds = ds[['Adj Close']]
ds.columns = ['Price']
plt.plot(ds)
st.pyplot(fig0)

scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(ds).reshape(-1, 1))

# # df1.shape
# st.write(df1)

training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size,
                            :], df1[training_size:len(df1), :1]


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  # i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# print(X_train.shape), print(y_train.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# model.load_model('final.json')
# import pickle
# model = pickle.load(open('finalized_model.sav', 'rb'))


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.1))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(
    X_test, ytest), epochs=25, batch_size=32, verbose=1)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# testing part
look_back = 10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2) +
                1:len(df1)-1, :] = test_predict

# plot baseline and predictions

fig1 = plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df1), label="original")
plt.plot(trainPredictPlot, label="train")
plt.plot(testPredictPlot, label="test")
plt.legend(loc='lower right')
# plt.show()
plt.xlabel('time')
plt.ylabel('adj close')
st.pyplot(fig1)

x_input = test_data[len(test_data)-11:].reshape(1, -1)
# x_input.shape

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for next 10 days

lst_output = []
i = 0
x_input = np.array(df1.tolist())
n_steps = 10
while (i < 10):

    if (len(temp_input) > 10):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1


# print(lst_output)

day_new = np.arange(1, 11)
day_pred = np.arange(11, 21)

i = 0
while (i < 10):
    st.write("Predicted stock price for day", i+1,  " => ",
             scaler.inverse_transform(lst_output)[i][0])
    i = i+1

df3 = df1.tolist()
df3.extend(lst_output)
# ds.tail()


fig2 = plt.figure(figsize=(12, 6))
plt.plot(day_new, scaler.inverse_transform(df1[len(df1)-10:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.xlabel('time')
plt.ylabel('adj close')
st.pyplot(fig2)

# fig3 = plt.figure(figsize = (12,6))
# df3=df1.tolist()
# df3.extend(lst_output)
# plt.plot(df3[len(df3)-20:])
# plt.xlabel('time')
# plt.ylabel('adj close')
# st.pyplot(fig3)

fig4 = plt.figure(figsize=(12, 6))
df3 = scaler.inverse_transform(df3).tolist()

plt.plot(df3)
plt.xlabel('time')
plt.ylabel('adj close')
st.pyplot(fig4)
