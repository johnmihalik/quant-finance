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
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
# %%

def EMA(source_data_frame, range, close_colummn="Adj Close" ):
    df = source_data_frame.copy()
    df["EMA" + str(range)] = df[close_colummn].ewm(span=range, min_periods=range).mean()
    return df.dropna()


# %%

def MACD(source_data_frame, fast=12, slow=26, signal=9, close_colummn="Adj Close"):
    df = source_data_frame.copy()
    df["EMA" + str(fast)] = df[close_colummn].ewm(span=fast, min_periods=fast).mean()
    df["EMA" + str(slow)] = df[close_colummn].ewm(span=slow, min_periods=slow).mean()
    df["MACD"] = df["EMA" + str(fast)] - df["EMA" + str(slow)]
    df["Signal"] = df["MACD"].ewm(span=signal, min_periods=signal).mean()

    return df.dropna()


#%%
def ATR(source_data_frame, ma_interval=14, ema=False, close_colummn="Adj Close"):
    "function to calculate True Range and Average True Range"
    df = source_data_frame.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df[close_colummn].shift(1))
    df['L-PC']=abs(df['Low']-df[close_colummn].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    if ema:
        df['ATR'] = df['TR'].ewm(span=ma_interval,adjust=False,min_periods=ma_interval).mean()
    else:
        df['ATR'] = df['TR'].rolling(ma_interval).mean()

    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2.dropna()

#%%
def BollingerBand(source_data_frame, n_stddev=2, sma_range=20, close_column="Adj Close"):
    "function to calculate Bollinger Band"
    df = source_data_frame.copy()
    df["MA"] = df[close_column].rolling(sma_range).mean()
    df["BB_up"] = df["MA"] +  n_stddev * df[close_column].rolling(sma_range).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] -  n_stddev * df[close_column].rolling(sma_range).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df2 = df.drop(["MA"], axis=1)
    df2.dropna(inplace=True)
    return df2

#%%
def RSI(source_data_frame, ma_interval=14, close_column="Adj Close"):
    "function to calculate RSI"
    df = source_data_frame.copy()
    df['delta']=df[close_column] - df[close_column].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < ma_interval:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == ma_interval:
            avg_gain.append(df['gain'].rolling(ma_interval).mean().tolist()[ma_interval])
            avg_loss.append(df['loss'].rolling(ma_interval).mean().tolist()[ma_interval])
        elif i > ma_interval:
            avg_gain.append(((ma_interval-1)*avg_gain[i-1] + gain[i])/ma_interval)
            avg_loss.append(((ma_interval-1)*avg_loss[i-1] + loss[i])/ma_interval)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))

    df2 = df.drop(['avg_gain','avg_loss','delta', 'RS', 'gain', 'loss'], axis=1)
    return df2.dropna()


#%%

def ADX(source_data_frame, ma_interval=14):
    "function to calculate ADX"
    #df2 = source_data_frame.copy()
    df2 = ATR(source_data_frame, ma_interval)

#    df2['TR'] = ATR(df2, ma_interval)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation


    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < ma_interval:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == ma_interval:
            TRn.append(df2['TR'].rolling(ma_interval).sum().tolist()[ma_interval])
            DMplusN.append(df2['DMplus'].rolling(ma_interval).sum().tolist()[ma_interval])
            DMminusN.append(df2['DMminus'].rolling(ma_interval).sum().tolist()[ma_interval])
        elif i > ma_interval:
            TRn.append(TRn[i-1] - (TRn[i-1]/14) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/14) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/14) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*ma_interval-1:
            ADX.append(np.NaN)
        elif j == 2*ma_interval-1:
            ADX.append(df2['DX'][j-ma_interval+1:j+1].mean())
        elif j > 2*ma_interval-1:
            ADX.append(((ma_interval-1)*ADX[j-1] + DX[j])/ma_interval)
    df2['ADX']=np.array(ADX)
    return df2.dropna()

#%%
def OBV(source_data_frame):
    """function to calculate On Balance Volume"""
    df = source_data_frame.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


#%%
ticker = "AAPL"

source = pdr.get_data_yahoo(ticker, datetime.date.today() - datetime.timedelta(365), datetime.date.today())
source_data_frame = source.copy()
ma_interval=14

#df = ATR(source_data_frame, 14)

df = ADX(source)
df.head()

