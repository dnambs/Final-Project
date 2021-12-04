#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dev Nambudiripad
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import base64
from datetime import *
from pathlib import Path
from dateutil.relativedelta import *

#
# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded
#
#
# header_html = "<img src='data:image/jpg;base64,{}' class='img-fluid'>".format(
#     img_to_bytes("stocks_image.jpg")
# )
# st.markdown(
#     header_html, unsafe_allow_html=True,
# )

st.title("Stock Portfolio Optimization")
st.write("This application would allow an individual to enter up to five stock tickers from the S&P 500 and a total "
         "amount to invest "
         " as well as a time period. Then this application would use a predictive model to" " "
         "first predict the closing price of each of the tickers and then provide an optimal share" " "
         "allocation to maximize returns")


@st.cache(persist=True)
def load_data():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = tickers.drop(['GICS Sub-Industry', 'CIK', 'SEC filings'], axis=1)
    return tickers


def run():
    st.subheader("S&P 500 descriptions loaded into a Pandas Dataframe.")
    tickers = load_data()
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
             unsafe_allow_html=True)

    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
             unsafe_allow_html=True)

    choose = st.radio("Choose how much data to show", ("Head", "All"))
    if choose == 'Head':
        st.dataframe(tickers.head())
    else:
        st.dataframe(tickers)

    # Stock Ticker Selection
    st.subheader("Stock Ticker Selection")
    stock_input = st.multiselect("Please enter 5 stock tickers from the list above:", tickers["Symbol"])
    st.markdown(f"You selected: {stock_input}")
    stock_1_input = stock_input[0]
    ticker_1 = yf.Ticker(stock_1_input)
    stock_2_input = stock_input[1]
    ticker_2 = yf.Ticker(stock_2_input)
    stock_3_input = stock_input[2]
    ticker_3 = yf.Ticker(stock_3_input)
    stock_4_input = stock_input[3]
    ticker_4 = yf.Ticker(stock_4_input)
    stock_5_input = stock_input[4]
    ticker_5 = yf.Ticker(stock_5_input)

    #  Download 5 years of data for each
    today = date.today()
    delta = relativedelta(years=5)
    start = today - delta
    end = today

    #
    stock_1 = yf.download(stock_1_input, start, end)
    stock_2 = yf.download(stock_2_input, start, end)
    stock_3 = yf.download(stock_3_input, start, end)
    stock_4 = yf.download(stock_4_input, start, end)
    stock_5 = yf.download(stock_5_input, start, end)

    st.subheader(f"{ticker_1.info['shortName']} ({stock_1_input})")
    st.write(stock_1.tail())
    st.subheader(f"{ticker_2.info['shortName']} ({stock_2_input})")
    st.write(stock_2.tail())
    st.subheader(f"{ticker_3.info['shortName']} ({stock_3_input})")
    st.write(stock_3.tail())
    st.subheader(f"{ticker_4.info['shortName']} ({stock_4_input})")
    st.write(stock_4.tail())
    st.subheader(f"{ticker_5.info['shortName']} ({stock_5_input})")
    st.write(stock_5.tail())




    # st.subheader("Reflection")
    # st.write("I'm currently in my exploration and research phase. So I am primarily determining"
    #          " " "what models, data, and calculations I'll need to make this application. As far as the model goes "
    #          "I'm currently considering one called Efficient Frontier but this may change as I do more research. As "
    #          "far as "
    #          "the stock data, I will most likely pull it from Yahoo Finance. In addition, I'm starting to thing about "
    #          "what "
    #          "what I would like my layout/selectors to be as this will inform how I do my coding. By the end of this "
    #          "weekend, I'm hoping that I'll have something a bit more tangible.")


if __name__ == '__main__':
    run()
