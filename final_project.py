#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Dev Nambudiripad
"""

import streamlit as st


st.title("Stock Portfolio Optimization")
st.write("This application would allow an individual to enter up to five stock tickers and a total amount to invest"
         " as well as a time period. Then this application would use a predictive model to" " "
         "first predict the closing price of each of the tickers and then provide an optimal share" " "
         "allocation to maximize returns")

st.subheader("Reflection")
st.write("I'm currently in my exploration and research phase. So I am primarily determining"
         " " "what models, data, and calculations I'll need to make this application. As far as the model goes " 
         "I'm currently considering one called Efficient Frontier but this may change as I do more research. As far as "
         "the stock data, I will most likely pull it from Yahoo Finance. In addition, I'm starting to thing about what "
         "what I would like my layout/selectors to be as this will inform how I do my coding. By the end of this "
         "weekend, I'm hoping that I'll have something a bit more tangible.")
