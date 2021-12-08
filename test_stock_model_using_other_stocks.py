#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dev Nambudiripad
"""

import yfinance as yf
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import silence_tensorflow.auto

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
from datetime import timedelta

model = keras.models.load_model("my_model")
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2021, 6, 1)
test_start = dt.datetime(2021, 6, 16)
test_end = dt.datetime.now()

data = web.DataReader('AMZN', 'yahoo', start, end)
test_data = web.DataReader('AMZN', 'yahoo', test_start, test_end)
prediction_days = 60
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

delta = timedelta(days=7)
data_compare = web.DataReader('AMZN', 'yahoo', dt.datetime.now() - delta, dt.datetime.now())

last_closing = data_compare['Close'].iloc[-1]
print(last_closing)

if prediction > last_closing:
    print("Higher")
