#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dev Nambudiripad
"""
### IMPORTING THINGS ###
import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from plotly import graph_objs as go
import yfinance as yf
import numpy as np
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
import silence_tensorflow.auto

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
from datetime import timedelta

model = keras.models.load_model("my_model")

option = st.sidebar.selectbox("Choose a Dashboard", ('Welcome Page', 'Stock Up/Down', 'Portfolio Optimization'))


@st.cache
def load_data():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.drop(['GICS Sub-Industry', 'CIK', 'SEC filings'], axis=1)
    return tickers


def predict_up_down(stock_input):
    predict_stock = stock_input
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2021, 6, 1)
    test_start = dt.datetime(2021, 6, 16)
    test_end = dt.datetime.now()

    data = web.DataReader(predict_stock, 'yahoo', start, end)
    test_data = web.DataReader(predict_stock, 'yahoo', test_start, test_end)
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

    return prediction


def load_ticker_data(stock):
    data = web.DataReader(stock, 'yahoo', dt.datetime(2015, 1, 3), dt.datetime.now())
    data.reset_index(inplace=True)
    return data


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def optimization(stock_1_input, stock_2_input, stock_3_input, stock_4_input, stock_5_input):
    start = dt.datetime(2021, 1, 1)
    end = dt.datetime.now()

    stock_1 = web.DataReader(stock_1_input, 'yahoo', start, end)
    stock_2 = web.DataReader(stock_2_input, 'yahoo', start, end)
    stock_3 = web.DataReader(stock_3_input, 'yahoo', start, end)
    stock_4 = web.DataReader(stock_4_input, 'yahoo', start, end)
    stock_5 = web.DataReader(stock_5_input, 'yahoo', start, end)

    stocks = pd.concat([stock_1['Close'], stock_2['Close'], stock_3['Close'], stock_4['Close'], stock_5['Close']],
                       axis=1)
    stocks.columns = [stock_1, stock_2, stock_3, stock_4, stock_5]
    st.write(stocks.head())

    returns = stocks / stocks.shift(1)

    logReturns = np.log(returns)

    noOfPortfolios = 10000
    weight = np.zeros((noOfPortfolios, 5))
    expectedReturn = np.zeros(noOfPortfolios)
    expectedVolatility = np.zeros(noOfPortfolios)
    sharpeRatio = np.zeros(noOfPortfolios)

    meanLogRet = logReturns.mean()
    Sigma = logReturns.cov()
    for k in range(noOfPortfolios):
        w = np.array(np.random.random(5))
        w = w / np.sum(w)
        weight[k, :] = w
        expectedReturn[k] = np.sum(meanLogRet * w)
        expectedVolatility[k] = np.dot(w.T, np.dot(Sigma, w))
        sharpeRatio[k] = expectedReturn[k] / expectedVolatility[k]

    maxIndex = sharpeRatio.argmax()
    weights = weight[maxIndex, :]
    return weights


def main():
    if option == 'Welcome Page':
        st.title("Investor's Toolbox-Lite")
        st.write(
            "This application would allow an individual to enter up to five stock tickers from the S&P 500 and a total "
            "amount to invest "
            " as well as a time period. Then this application would use a predictive model to" " "
            "first predict the closing price of each of the tickers and then provide an optimal share" " "
            "allocation to maximize returns")

    if option == 'Stock Up/Down':
        st.header(option)
        st.subheader("S&P 500 Companies")
        tickers = load_data()
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
                 unsafe_allow_html=True)

        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
                 unsafe_allow_html=True)

        choose = st.radio("Choose how much data to show", ("Head", "All"))
        if choose == 'Head':
            st.table(tickers.head())
        else:
            st.dataframe(tickers)

        st.subheader("Stock Ticker Selection")
        stock_input = st.selectbox("Please enter a stock tickers from the list above:", tickers["Symbol"])
        ticker_data = load_ticker_data(stock_input)
        plot_raw_data(ticker_data)

        st.markdown("**Hit Predict if you want to see how tomorrow's closing price will be!**")
        if st.button('Predict'):
            st.write('Some intense predicting in progress...')
            output = predict_up_down(stock_input)
            st.write('Predictions complete...phew!')

            delta = timedelta(days=7)
            data_compare = web.DataReader(stock_input, 'yahoo', dt.datetime.now() - delta, dt.datetime.now())

            last_closing = data_compare['Close'].iloc[-1]

            high_html = """  
              <div style="background-color:#88d8b0;padding:10px >
               <h2 style="color:white;text-align:center;"> The chosen stock ticker's closing price will be HIGHER tomorrow
               </h2>
               </div>
            """
            low_html = """  
              <div style="background-color:#ff6f69;padding:10px >
               <h2 style="color:black ;text-align:center;"> The chosen stock ticker's closing price will be LOWER 
               tomorrow</h2>
               </div>
            """

            if output > last_closing:
                st.markdown(high_html, unsafe_allow_html=True)
            else:
                st.markdown(low_html, unsafe_allow_html=True)

    ###END OF PREDICTION CODE###

    if option == 'Portfolio Optimization':
        st.header(option)
        st.write('This optimizer uses Modern Portfolio Theory to provide the optimal allocation of shares for'
                 'the HIGHEST Returns')
        st.write('You can find more information on the fundamentals of this theory here:')
        st.write('(https://www.thebalance.com/what-is-mpt-2466539')


        tickers = load_data()
        # Stock Ticker Selection
        st.subheader("Stock Ticker Multi-Selection")
        stock_inputs = st.multiselect("Please enter 5 stock tickers from the S&P 500:", tickers["Symbol"])
        stock_1_input = stock_inputs[0]
        stock_2_input = stock_inputs[1]
        stock_3_input = stock_inputs[2]
        stock_4_input = stock_inputs[3]
        stock_5_input = stock_inputs[4]

        number = st.number_input('How much money would you like to invest?')

        if st.button('Optimize'):
            st.write('Optimizing away...')
            start = dt.datetime(2021, 1, 1)
            end = dt.datetime.now()

            stock_1 = web.DataReader(stock_1_input, 'yahoo', start, end)
            stock_2 = web.DataReader(stock_2_input, 'yahoo', start, end)
            stock_3 = web.DataReader(stock_3_input, 'yahoo', start, end)
            stock_4 = web.DataReader(stock_4_input, 'yahoo', start, end)
            stock_5 = web.DataReader(stock_5_input, 'yahoo', start, end)

            stocks = pd.concat(
                [stock_1['Close'], stock_2['Close'], stock_3['Close'], stock_4['Close'], stock_5['Close']],
                axis=1)
            stocks.columns = [stock_1_input, stock_2_input, stock_3_input,stock_4_input, stock_5_input]

            returns = stocks / stocks.shift(1)

            logReturns = np.log(returns)

            noOfPortfolios = 10000
            weight = np.zeros((noOfPortfolios, 5))
            expectedReturn = np.zeros(noOfPortfolios)
            expectedVolatility = np.zeros(noOfPortfolios)
            sharpeRatio = np.zeros(noOfPortfolios)

            meanLogRet = logReturns.mean()
            Sigma = logReturns.cov()
            for k in range(noOfPortfolios):
                w = np.array(np.random.random(5))
                w = w / np.sum(w)
                weight[k, :] = w
                expectedReturn[k] = np.sum(meanLogRet * w)
                expectedVolatility[k] = np.dot(w.T, np.dot(Sigma, w))
                sharpeRatio[k] = expectedReturn[k] / expectedVolatility[k]

            maxIndex = sharpeRatio.argmax()
            weights = weight[maxIndex, :]
            st.write('I am now much improved...')

            st.write(weights)
            money_stock_1 = weights[0] * number
            money_stock_2 = weights[1] * number
            money_stock_3 = weights[2] * number
            money_stock_4 = weights[3] * number
            money_stock_5 = weights[4] * number

            st.markdown('The optimal allocation:')
            st.write(f"{stock_1_input}: $ {money_stock_1}")
            st.write(f"{stock_2_input}: $ {money_stock_2}")
            st.write(f"{stock_3_input}: $ {money_stock_3}")
            st.write(f"{stock_4_input}: $ {money_stock_4}")
            st.write(f"{stock_5_input}: $ {money_stock_5}")



# st.subheader(f"{ticker_1.info['shortName']} ({stock_1_input})")
# st.write(stock_1.tail())
# st.subheader(f"{ticker_2.info['shortName']} ({stock_2_input})")
# st.write(stock_2.tail())
# st.subheader(f"{ticker_3.info['shortName']} ({stock_3_input})")
# st.write(stock_3.tail())
# st.subheader(f"{ticker_4.info['shortName']} ({stock_4_input})")
# st.write(stock_4.tail())
# st.subheader(f"{ticker_5.info['shortName']} ({stock_5_input})")
# st.write(stock_5.tail())
if __name__ == '__main__':
    main()
