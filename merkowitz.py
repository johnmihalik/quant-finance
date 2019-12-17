import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import scipy.optimize as optimization
from tabulate import tabulate

yf.pdr_override()

# Portfolio
#stocks = ['SPYG', 'AAPL', 'WMT' , 'TSLA', 'AMZN', 'GE' , 'DB' ,'GOOG', 'MSFT', 'COST', 'PG', 'X']

# SPY sectors
#stocks = ['XLC', 'XLP', 'XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK' , 'XLU']

# SPY Industries
#stocks = ['KBE', 'KRE', 'KCE', 'KIE', 'XAR', 'XTN', 'XBI', 'XPH', 'XHE', 'XHS', 'XOP', 'XES', 'XME', 'XRT', 'XHB', 'XSD', 'XSW', 'XNTK', 'XITK', 'XTL', 'XTH' ,'XWEB']

# SP500
#stocks = ['LVS','NVR','CDW','LDOS','IEX','TMUS','MKTX','AMCR','CTVA','DOW','FOX','FOXA','WAB','ATO','TFX','FRC','CE','LW','MXIM','FANG','JKHY','KEYS','LIN','FTNT','ROL','WCG','ANET','CPRT','FLT','BR','HFC','TWTR','EVRG','ABMD','MSCI','TTWO','SIVB','IPGP','HII','NCLH','CDNS','SBAC','IQV','AOS','PKG','RMD','DRE','MGM','BHGE','HLT','ALGN','ANSS','RE','INFO','IT','DXC','RJF','ARE','AMD','SNPS','DISH','REG','CBOE','INCY','FTI','IDXX','MAA','COTY','COO','CHTR','MTD','FTV','ALB','LNT','FBHS','UA','TDG','AJG','LKQ','DLR','ALK','GPN','ULTA','CNC','HOLX','AWK','UDR','CXO','CFG','FRT','EXR','WLTW','CHD','nan','ILMN','SYF','HPE','VRSK','NWS','UAL','ATVI','PYPL','AAP','KHC','WRK','JBHT','QRVO','O','SLG','EQIX','HBI','AAL','HSIC','SWKS','HCA','RCL','UHS','URI','DISCK','MLM','AMG','XEC','AVGO','UAA','GOOG','ESS','TSCO','FB','MHK','ADS','ALLE','CPRI','VRTX','AME','DAL','NLSN','NWSA','ZTS','GM','KSU','MAC','REGN','PVH','ABBV','APTV','GRMN','DG','PNR','LYB','STX','MNST','LRCX','KMI','ALXN','PSX','CCI','TRIP','BWA','PRGO','DLTR','XYL','TEL','MOS','ACN','MPC','CMG','BLK','EW','FFIV','NFLX','IR','CB','KMX','CERN','OKE','DISCA','HP','BRK.B','NRG','ROP','ROST','V','BKNG','FMC','PWR','WDC','ORLY','ES','HRL','VTR','WELL','IRM','FLIR','RSG','XRAY','WYNN','PBCT','SJM','WEC','NDAQ','FLS','APH','PXD','LHX','CRM','FAST','CF','IVZ','DVA','MA','COG','ISRG','HCP','PM','AMT','JEC','EXPD','NBL','EXPE','ICE','MCHP','AKAM','DFS','AIZ','MDLZ','HST','CHRW','VAR','RL','AVB','CTSH','FIS','CBRE','CELG','WU','CME','JNPR','GOOGL','BXP','KIM','VRSN','EL','VIAB','AMZN','LEN','AMP','PSA','TSN','VNO','STZ','DHI','NOV','LH','TPR','GILD','VLO','MYL','ETFC','MTB','BIIB','PLD','SYMC','MKC','AIV','DGX','TRV','ANTM','UPS','GS','PRU','EA','EBAY','PFG','SPG','WAT','EQR','NVDA','ABC','ZBH','ZION','FISV','CTAS','SYK','MET','RHI','INTU','EOG','NI','MCO','DVN','TIF','SBUX','A','HOG','CTXS','XLNX','LEG','TROW','ADI','PNW','QCOM','VMC','BBY','NTAP','AFL','CMS','AGN','CTL','MCK','CCL','DHR','AES','PAYX','RF','KSS','COF','BEN','SEE','NTRS','OMC','CINF','BBT','YUM','KLAC','STT','HBAN','PGR','APA','EFX','SCHW','CAH','ADBE','TMO','AZO','AON','FITB','HIG','CMA','HUM','PPL','M','MS','FCX','ALL','DRI','L','BK','AMAT','BSX','CBS','MU','LUV','UNH','MSFT','KEY','UNM','EMN','CSCO','MAR','COST','IPG','AMGN','AEE','MRO','ADSK','ORCL','JCI','GL','NWL','ECL','NKE','STI','C','PNC','HD','AVY','CMCSA','MMC','SYY','HRB','MDT','JWN','GPS','ITW','PH','DOV','TJX','CNP','NOC','PKI','APD','NUE','BLL','HAS','LMT','HES','PHM','LOW','VZ','T','LB','CAG','OXY','AAPL','BF.B','SNA','SWK','WMT','WM','ADM','MAS','GWW','ADP','FDX','PCAR','AIG','WBA','VFC','TXT','TGT','INTC','TAP','BAC','WFC','DUK','LNC','AXP','CI','DIS','NEE','IFF','JPM','WMB','HPQ','GPC','JNJ','BAX','BDX','LLY','MCD','NEM','GIS','CLX','CSX','CMI','SLB','EMR','SHW','FE','WHR','XOM','SRE','DTE','EXC','WY','AEP','PG','XRX','ETR','CVS','PPG','NSC','UTX','ROK','DD','XEL','MSI','TXN','CL','HON','PFE','SPGI','CPB','GE','GD','K','MO','GLW','F','MMM','SO','IBM','CAT','ABT','MRK','RTN','BMY','ED','UNP','KO','ARNC','KR','HSY','HAL','COP','KMB','PEG','ETN','DE','CVX','EIX','IP','BA','D','PEP' ]

# SPYDR Industrial ETF's AND Sectors
#stocks = ['XLC', 'XLP', 'XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK' , 'XLU', 'KBE', 'KRE', 'KCE', 'KIE', 'XAR', 'XTN', 'XBI', 'XPH', 'XHE', 'XHS', 'XOP', 'XES', 'XME', 'XRT', 'XHB', 'XSD', 'XSW', 'XNTK', 'XITK', 'XTL', 'XTH' ,'XWEB']

stocks = ['SPYG', 'AAPL', 'WMT' , 'TSLA', 'GE']

# Start and end date for stock data to use in calculations
#start = pd.to_datetime('2017-12-06')
#end = pd.to_datetime('2019-12-06')
end = pd.Timestamp.today()
start =  end -  datetime.timedelta(days=365)


min_allocation = 0.0
max_allocation = 1.0

def download_data(stocks):
    data = pdr.get_data_yahoo(stocks, start=start, end=end) ['Adj Close']
    return data

def show_data(data):
	data.plot(figsize=(10,5))
	plt.show()

def calculate_returns(data):
	returns = np.log(data/data.shift(1))
	return returns

def plot_daily_returns(returns):
	returns.plot(figsize=(10,5))
	plt.show()

#print out mean and covariance of stocks within [start_date, end_date]. There are 252 trading days within a year
def show_statistics(returns):
	print("Average Returns")
	print(returns.mean()*252)

	print("Covariance Matrix")
	print(returns.cov()*252)


# weights defines what stocks to include (with what portion) in the portfolio
def initialize_weights():
	weights = np.ones(len(stocks))
#	weights = np.random.random(len(stocks))
	weights /= np.sum(weights)
	return weights


# expected portfolio return,  = SUM ( weights per stock * expected return of stock )
def calculate_portfolio_return(returns, weights):
	portfolio_return = np.sum(returns.mean() * weights) * 252
	print("Expected portfolio return:", portfolio_return)


# expected portfolio variance or RISK  =  WT * SIGMA * W
def calculate_portfolio_variance(returns, weights):
	portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
	print("Expected variance:", portfolio_variance)

# Generate random portfolios, using random weights of the stocks
# Return the returns for each portfolio and the variance of the portfolio
def generate_portfolios(num_portfolios, stocks, returns):

	print("Generating Monte-Carlo Portfolios...")
	portfolio_returns = []
	portfolio_variances = []

	# Monte-Carlo simulation: we generate several random weights -> so random portfolios !!!
	for i in range(num_portfolios):
		weights = np.random.random(len(stocks))
		weights /= np.sum(weights)

		portfolio_returns.append(np.sum(returns.mean() * weights) * 252)
		portfolio_variances.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))

	np_returns = np.array(portfolio_returns)
	np_variances = np.array(portfolio_variances)
	return np_returns, np_variances

def plot_portfolios(returns, variances):
	plt.figure(figsize=(10, 7))
	plt.scatter(variances, returns, c=returns / variances, marker='o')
	plt.grid(True)
	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Return')
	plt.colorbar(label='Sharpe Ratio')
	plt.show()


# OK this is the result of the simulation ... we have to find the optimal portfolio with
# some optimization technique !!! scipy can optimize functions (minimum/maximum finding)
def statistics(weights, returns):
	portfolio_return = np.sum(returns.mean() * weights) * 252
	portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
	return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


def min_func_sharpe(weights, returns):
	return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):

	# the sum of weights is 1
	constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

	# the weights can be 1 at most: 1 when 100% of money is invested into a single stock
	bounds = tuple((min_allocation, max_allocation) for x in range(len(stocks)))

	optimum = optimization.minimize(fun=min_func_sharpe, x0=weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

	return optimum


# optimal portfolio according to weights: 0 means no shares of that given company
def print_optimal_portfolio(optimum, returns):

	expected_return, volatility, sharpe = statistics(optimum['x'].round(3), returns)

	print("Portfolio Allocation:")
	for i in range(0,len(stocks)):
		if optimum['x'][i] > 0.00001:
			print(" %6s : % 4.3f" % (returns.columns[i], optimum['x'][i]) )

	print("\n-------------------------------------")
	print("Expected Return  : ", expected_return.round(3))
	print("Volatility       : ", volatility.round(3))
	print("Sharpe Ratio     : ", sharpe.round(3))


def show_optimal_portfolio(optimum, returns, preturns, pvariances):
	plt.figure(figsize=(10, 7))
	plt.scatter(pvariances, preturns, c=preturns / pvariances, marker='o')
	plt.grid(True)
	plt.xlabel('Expected Volatility')
	plt.ylabel('Expected Return')
	plt.colorbar(label='Sharpe Ratio')
	plt.plot(statistics(optimum['x'], returns)[1], statistics(optimum['x'], returns)[0], 'g*', markersize=20.0)
	plt.show()


if __name__ == "__main__":

	data = download_data(stocks)
#	show_data(data)
	returns = calculate_returns(data)
#	plot_daily_returns(returns)
	show_statistics(returns)
	weights=initialize_weights()
#	calculate_portfolio_return(returns,weights)
#	calculate_portfolio_variance(returns,weights)
	preturns, pvariances = generate_portfolios(10000, stocks, returns)
	plot_portfolios(preturns, pvariances)
	optimum = optimize_portfolio(weights,returns)
	print_optimal_portfolio(optimum, returns)
#	show_optimal_portfolio(optimum, returns, preturns, pvariances)


