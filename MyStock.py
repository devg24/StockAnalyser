
import yfinance as yf
import pandas as pd
import numpy as np
import utils
from sklearn.decomposition import PCA


dataset = "Company Names and Ticker Symbols/yahootickers.xlsx"
cols = ["Ticker","Name","Exchange","Category Name"]
def get_data(dataset, start, end, cols):
    '''
    @param dataset: path to dataset
    @param start: start date
    @param end: end date
    @param cols: columns to read from dataset

    @return df_companies: dataframe of companies
    @return df_stocks: dataframe of stocks
    get all companies from dataset

    '''
    df_companies = pd.read_excel(dataset,sheet_name= "Stock",skiprows=3,usecols=cols)
    df_companies.dropna(inplace=True)
    stocks = []
    for i in range(0,len(df_companies),100):
        df_stock = yf.download(df_companies["Ticker"].tolist()[i:i+100],start=start,end=end)
        stocks.append(df_stock)
        # save to csv
        # df_stock.to_csv("Stocks/{}.csv".format(i))
    df_stock = pd.concat(stocks)
    df_stock = df_stock[["Adj Close"]]
    df_stock.columns = df_stock.columns.droplevel(0)
    print(df_stock.head())
    return df_companies, df_stock

def perform_pca(X_train, Y_train, n_components):
    '''
    @param X_train: training features
    @param Y_train: training labels

    @return pca: pca object
    @return X_train_pca: training features after pca
    @return Y_train_pca: training labels after pca
    perform pca on training data

    '''
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    Y_train_pca = pca.fit_transform(Y_train)
    return pca, X_train_pca, Y_train_pca

df_companies, df_stocks = get_data(dataset, "2019-01-01", "2020-01-01", cols)









