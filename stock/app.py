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
user_input=st.text_input('ENTER STOCK TICKER','AAPL')
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

df = pdr.get_data_yahoo(user_input, start = dt.datetime(2018, 1, 2), end = date.today())


#Describing data
st.subheader('Data from 2018')
st.write(df)