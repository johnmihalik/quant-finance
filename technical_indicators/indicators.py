# %%
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimization
import yfinance as yf
import itertools

yf.pdr_override()

# %%

def add_ema(data, close_colummn, range):
    df = data.copy()
    df["EMA" + str(range)] = df[close_colummn].ewm(span=range, min_periods=range).mean()
    return df.dropna()


# %%

def add_macd(data, close_colummn, fast=12, slow=26, signal=9):
    df = data.copy()
    df["EMA" + str(fast)] = df[close_colummn].ewm(span=fast, min_periods=fast).mean()
    df["EMA" + str(slow)] = df[close_colummn].ewm(span=slow, min_periods=slow).mean()
    df["MACD"] = df["EMA" + str(fast)] - df["EMA" + str(slow)]
    df["Signal"] = df["MACD"].ewm(span=signal, min_periods=signal).mean()

    return df.dropna()

#%%
ticker = "AAPL"

data = pdr.get_data_yahoo(ticker, datetime.date.today() - datetime.timedelta(365), datetime.date.today())

macd_df = add_macd(data, "Adj Close")

macd_df.head()

