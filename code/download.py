# For reading stock data from yahoo
import yfinance as yf
import pandas as pd
import datetime as dt
import time
#from pandas_datareader.data import DataReader

#from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN','TSLA','NVDA','META','AMD','BABA','INTC','PYPL','EA']

# Set up End and Start times for data grab
#end = datetime.now()
#start = datetime(end.year - 5, end.month, end.day)

for stock in tech_list:
    print(stock)
    globals()[stock] = yf.download(stock, end="2024-01-01")


company_list = [globals()[stock] for stock in tech_list]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON","TESLA","NVIDIA","META","AMD","ALIBABA","INTEL","PAYPAL","ELECTRONIC ARTS"]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name
