
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


# get all companies from dataset
df_companies = pd.read_excel("Company Names and Ticker Symbols/yahootickers.xlsx",sheet_name= "Stock",skiprows=3,usecols=["Ticker","Name","Exchange","Category Name"])
df_companies.dropna(inplace=True)
start = "2010-01-01"
end = "2011-01-01"
stocks = []
for i in range(0,1,100):
    df_stock = yf.download(df_companies["Ticker"].tolist()[i:i+100],start=start,end=end,group_by="ticker")
    stocks.append(df_stock)
    # save to csv
    # df_stock.to_csv("Stocks/{}.csv".format(i))
df_stock = pd.concat(stocks)
print(df_stock.head())


# perform pca on stocks
# pca = PCA(n_components=10)
# pca.fit(df_stock)
# df_pca = pd.DataFrame(pca.transform(df_stock),columns=["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"])
# print(df_pca.head())








