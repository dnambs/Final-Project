#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dev Nambudiripad
"""
import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import numpy as np

start = dt.datetime(2021, 1, 1)
end = dt.datetime.now()

stock_1 = web.DataReader('AMZN', 'yahoo', start, end)
stock_2 = web.DataReader('MSFT', 'yahoo', start, end)
stock_3 = web.DataReader('MMM', 'yahoo', start, end)
stock_4 = web.DataReader('CEVA', 'yahoo', start, end)
stock_5 = web.DataReader('ATVI', 'yahoo', start, end)

stocks = pd.concat([stock_1['Close'], stock_2['Close'], stock_3['Close'], stock_4['Close'], stock_5['Close']], axis=1)
print(stocks.head())
# stocks.columns = ['AMZN', 'MSFT', 'MMM', 'CEVA', 'ATVI']
#
# returns = stocks / stocks.shift(1)
#
# logReturns = np.log(returns)
#
# noOfPortfolios = 10000
# weight = np.zeros((noOfPortfolios, 5))
# expectedReturn = np.zeros(noOfPortfolios)
# expectedVolatility = np.zeros(noOfPortfolios)
# sharpeRatio = np.zeros(noOfPortfolios)
#
# meanLogRet = logReturns.mean()
# Sigma = logReturns.cov()
# for k in range(noOfPortfolios):
#     w = np.array(np.random.random(5))
#     w = w / np.sum(w)
#     weight[k, :] = w
#     expectedReturn[k] = np.sum(meanLogRet * w)
#     expectedVolatility[k] = np.dot(w.T, np.dot(Sigma, w))
#     sharpeRatio[k] = expectedReturn[k] / expectedVolatility[k]
#
# maxIndex = sharpeRatio.argmax()
# weights = weight[maxIndex, :]
# print(weights)
# print(weights[0])
