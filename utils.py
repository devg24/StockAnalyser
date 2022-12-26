import yfinance as yf
import pandas as pd
import numpy as np

google = yf.Ticker("GOOG")
google.info
print(google.price)

