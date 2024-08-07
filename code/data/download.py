#############################################
# DOWNLOAD DE CERTAS AÇÕES DO YAHOO FINANCE #
#############################################

# For reading stock data from yahoo
import yfinance as yf
import pandas as pd
import datetime as dt
import time

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
tech_list = ['AAPL',]

# Set up End and Start times for data grab
#end = datetime.now()
#start = datetime(end.year - 5, end.month, end.day)

for stock in tech_list:
    print(stock)
    globals()[stock] = yf.download(stock, start="2024-01-01")


company_list = [globals()[stock] for stock in tech_list]
company_name = ["APPLE", ]

for company, com_name in zip(company_list, company_name):
    company["company_name"] = com_name


