import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as pdr
import yfinance as yf
yf.pdr_override()


def getSummary(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix, stockData

def get_latest_price(ticker):
    start = dt.datetime(2023, 1, 1)
    end = dt.datetime.now()
    stockData = pdr.get_data_yahoo(ticker, start, end)
    latest_price = stockData['Close'].iloc[-1]
    return latest_price

def options_data(symbol):

    ticker = yf.Ticker(symbol)
    expirations = ticker.options
    options = pd.DataFrame()

    for e in expirations:
        opt = ticker.option_chain(e)
        opt = pd.concat([opt.calls, opt.puts])
        opt['expirationDate'] = e
        options = pd.concat([options, opt], ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + pd.DateOffset(days=1)
    options['DTE'] = (options['expirationDate'] - pd.to_datetime('today')).dt.days / 365

    options['Call'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    options['Mark Price'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask

    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)

    options = options.drop(columns=['change', 'percentChange', 'lastTradeDate', 'contractSize', 'currency', 'volume', 'inTheMoney', 'openInterest', 'lastPrice'])

    return options

# for dividend paying stocks, convert and use s_x = s/(1+q) where q is the compounded dividend yield --> (i.e. NVDA, MSFT, TSM)
stockList = ['AI', 'GOOGL', 'META', 'AMD', 'NOW']
startDate = dt.datetime(2023, 1, 1)
endDate = dt.datetime.now()

meanReturns, covMatrix, stockData = getSummary(stockList, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Simulation for regular AI stock equity portfolio 
sims = 100
T_days = 365 

meanMatrix = np.full(shape=(T_days, len(weights)), fill_value=meanReturns)
meanMatrix = meanMatrix.T

portfolio_sims = np.full(shape=(T_days, sims), fill_value=0.0)
starting_balance = 10000

for m in range(0, sims):
    Z = np.random.normal(size=(T_days, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanMatrix + np.inner(L, Z)
    portfolio_sims[:,m] = starting_balance*np.cumprod(np.inner(weights, dailyReturns.T)+1)

"""
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of AI Stock Portfolio')
plt.show()
"""

def getVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def getCVaR(returns, alpha=5):
    varLoss = returns <= getVaR(returns, alpha=alpha)
    return returns[varLoss].mean()
    
portfolio_end = pd.Series(portfolio_sims[-1,:])

VaR = starting_balance - getVaR(portfolio_end, alpha=5)
CVaR = starting_balance - getCVaR(portfolio_end, alpha=5)