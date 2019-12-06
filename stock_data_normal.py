import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

yf.pdr_override()

# Portfolio
stocks = ['AAPL', 'WMT' , 'TSLA', 'AMZN', 'GE' , 'DB' ]

# Start and end date for stock data to use in calculations
start = pd.to_datetime('2013-12-12')
end = pd.to_datetime('2019-11-20')

data = pdr.get_data_yahoo(stocks, start=start, end=end) ['Adj Close']

returns = np.log(data / data.shift(1))
returns.hist(bins=100)

plt.show()


