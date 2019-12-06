import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as optimization
import itertools

# SP500
#stock_list = ['LVS','NVR','CDW','LDOS','IEX','TMUS','MKTX','AMCR','CTVA','DOW','FOX','FOXA','WAB','ATO','TFX','FRC','CE','LW','MXIM','FANG','JKHY','KEYS','LIN','FTNT','ROL','WCG','ANET','CPRT','FLT','BR','HFC','TWTR','EVRG','ABMD','MSCI','TTWO','SIVB','IPGP','HII','NCLH','CDNS','SBAC','IQV','AOS','PKG','RMD','DRE','MGM','BHGE','HLT','ALGN','ANSS','RE','INFO','IT','DXC','RJF','ARE','AMD','SNPS','DISH','REG','CBOE','INCY','FTI','IDXX','MAA','COTY','COO','CHTR','MTD','FTV','ALB','LNT','FBHS','UA','TDG','AJG','LKQ','DLR','ALK','GPN','ULTA','CNC','HOLX','AWK','UDR','CXO','CFG','FRT','EXR','WLTW','CHD','nan','ILMN','SYF','HPE','VRSK','NWS','UAL','ATVI','PYPL','AAP','KHC','WRK','JBHT','QRVO','O','SLG','EQIX','HBI','AAL','HSIC','SWKS','HCA','RCL','UHS','URI','DISCK','MLM','AMG','XEC','AVGO','UAA','GOOG','ESS','TSCO','FB','MHK','ADS','ALLE','CPRI','VRTX','AME','DAL','NLSN','NWSA','ZTS','GM','KSU','MAC','REGN','PVH','ABBV','APTV','GRMN','DG','PNR','LYB','STX','MNST','LRCX','KMI','ALXN','PSX','CCI','TRIP','BWA','PRGO','DLTR','XYL','TEL','MOS','ACN','MPC','CMG','BLK','EW','FFIV','NFLX','IR','CB','KMX','CERN','OKE','DISCA','HP','NRG','ROP','ROST','V','BKNG','FMC','PWR','WDC','ORLY','ES','HRL','VTR','WELL','IRM','FLIR','RSG','XRAY','WYNN','PBCT','SJM','WEC','NDAQ','FLS','APH','PXD','LHX','CRM','FAST','CF','IVZ','DVA','MA','COG','ISRG','HCP','PM','AMT','JEC','EXPD','NBL','EXPE','ICE','MCHP','AKAM','DFS','AIZ','MDLZ','HST','CHRW','VAR','RL','AVB','CTSH','FIS','CBRE','CELG','WU','CME','JNPR','GOOGL','BXP','KIM','VRSN','EL','VIAB','AMZN','LEN','AMP','PSA','TSN','VNO','STZ','DHI','NOV','LH','TPR','GILD','VLO','MYL','ETFC','MTB','BIIB','PLD','SYMC','MKC','AIV','DGX','TRV','ANTM','UPS','GS','PRU','EA','EBAY','PFG','SPG','WAT','EQR','NVDA','ABC','ZBH','ZION','FISV','CTAS','SYK','MET','RHI','INTU','EOG','NI','MCO','DVN','TIF','SBUX','A','HOG','CTXS','XLNX','LEG','TROW','ADI','PNW','QCOM','VMC','BBY','NTAP','AFL','CMS','AGN','CTL','MCK','CCL','DHR','AES','PAYX','RF','KSS','COF','BEN','SEE','NTRS','OMC','CINF','BBT','YUM','KLAC','STT','HBAN','PGR','APA','EFX','SCHW','CAH','ADBE','TMO','AZO','AON','FITB','HIG','CMA','HUM','PPL','M','MS','FCX','ALL','DRI','L','BK','AMAT','BSX','CBS','MU','LUV','UNH','MSFT','KEY','UNM','EMN','CSCO','MAR','COST','IPG','AMGN','AEE','MRO','ADSK','ORCL','JCI','GL','NWL','ECL','NKE','STI','C','PNC','HD','AVY','CMCSA','MMC','SYY','HRB','MDT','JWN','GPS','ITW','PH','DOV','TJX','CNP','NOC','PKI','APD','NUE','BLL','HAS','LMT','HES','PHM','LOW','VZ','T','LB','CAG','OXY','AAPL','SNA','SWK','WMT','WM','ADM','MAS','GWW','ADP','FDX','PCAR','AIG','WBA','VFC','TXT','TGT','INTC','TAP','BAC','WFC','DUK','LNC','AXP','CI','DIS','NEE','IFF','JPM','WMB','HPQ','GPC','JNJ','BAX','BDX','LLY','MCD','NEM','GIS','CLX','CSX','CMI','SLB','EMR','SHW','FE','WHR','XOM','SRE','DTE','EXC','WY','AEP','PG','XRX','ETR','CVS','PPG','NSC','UTX','ROK','DD','XEL','MSI','TXN','CL','HON','PFE','SPGI','CPB','GE','GD','K','MO','GLW','F','MMM','SO','IBM','CAT','ABT','MRK','RTN','BMY','ED','UNP','KO','ARNC','KR','HSY','HAL','COP','KMB','PEG','ETN','DE','CVX','EIX','IP','BA','D','PEP' ]
#stock_list = ['AAPL','MSFT','AMZN','CAT','UA','GOOG', 'PYPL']
stock_list = ['KBE', 'KRE', 'KCE', 'KIE', 'XAR', 'XTN', 'XBI', 'XPH', 'XHE', 'XHS', 'XOP', 'XES', 'XME', 'XRT', 'XHB', 'XSD', 'XSW', 'XNTK', 'XITK', 'XTL', 'XTH' ,'XWEB']

portfolio_size = 4

min_allocation = 0.1
max_allocation = 0.5

start = '2018-01-01'
end = '2019-11-20'

def load_sp500_from_file():
    print("Loading SP500 Dataset...")
    data = pd.read_csv("/Users/john/Desktop/Stock-Data/sp500_returns.csv")
    a = data['Date'] >= start
    b = data['Date'] <= end
    data = data[ a & b ]
    return data

def download_data(stocks):
    print("Downloading stock data...")
    data = pdr.get_data_yahoo(stocks, start=start, end=end) ['Adj Close']
    data = np.log(data / data.shift(1))
    return data

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

def min_func_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(stocks, weights, returns):
    # the sum of weights is 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((min_allocation, max_allocation) for x in range( len(stocks)))

    optimum = optimization.minimize(fun=min_func_sharpe, x0=weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimum

def calculate_optimum_portfolio(stocks, start, end):
    return 0


# weights defines what stocks to include (with what portion) in the portfolio
def initialize_weights(stocks):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    return weights

# expected portfolio return,  = SUM ( weights per stock * expected return of stock )
def print_portfolio_return(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    print("Expected portfolio return:", portfolio_return)

# expected portfolio variance or RISK  =  WT * SIGMA * W
def print_portfolio_variance(returns, weights):
    portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    print("Expected variance:", portfolio_variance)

def findsubsets(s,m):
    return set(itertools.combinations(s, m))


if __name__ == "__main__":

    data = download_data(stock_list)
    #data = load_sp500_from_file()

    print("Finding Optimal Portfolios...")

    for p in findsubsets(stock_list, portfolio_size):
        #print(str(p), eof="")

        weights = initialize_weights(p)

        s = list(p)
        returns = data[s]

        optimum = optimize_portfolio(p, weights, returns)
        for i in range(0,len(s)):
            print("%6s : %4.3f   " % (s[i], optimum['x'][i]), end="" )

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        print(" Return = %4.3f , Var = %4.3f" %(portfolio_return, portfolio_variance))

	#print_portfolio_return(returns, weights)
	#print_portfolio_variance(returns, weights)


