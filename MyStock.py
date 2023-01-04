import utils

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf


import requests
import time
from functools import reduce

LIMIT = 20
OFFSET = 0

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
    # e = len(df_companies)
    e = 5
    for i in range(0,e//5, 5):
        for j in range(i, i+5):
            symbol = df_companies['Ticker'].to_list()[j]
            print(symbol)
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey=S1T4GAWWNLWYKAMF'
            r = requests.get(url)
            data = r.json()
            df = pd.DataFrame.from_dict(data['Monthly Time Series'], orient='index')
            df.rename(columns={"4. close": symbol}, inplace=True)
            df.drop(columns=["1. open","2. high","3. low","5. volume"], inplace=True)
            df['date'] = df.index
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start) & (df['date'] <= end)]
            stocks.append(df)

            
        # time.sleep(65)
        # save to csv
        # df_stock.to_csv("Stocks/{}.csv".format(i))
    df_stock = reduce(lambda left,right: pd.merge(left,right,on='date'), stocks)
    

    return df_companies, df_stock

def perform_pca(market_data):
    '''
    @param market_data: dataframe of market data

    @return x: Dataframe of scaled and pca transformed data
    '''
    print("Scalling data...")
    scaler = StandardScaler()

    x_data = market_data.drop(columns=['date'])
    x = market_data.drop(columns=['date'])

    x = scaler.fit_transform(x)
    print("Performing PCA...")
    pca = PCA(0.90)
    x = pca.fit_transform(x)

    pca_components = abs(pca.components_)
    # # print top relevant features
    for row in range(pca_components.shape[0]):
        # get the indices of the top 4 values in each row
        temp = np.argpartition(-(pca_components[row]), 4)
        
        # sort the indices in descending order
        indices = temp[np.argsort((-pca_components[row])[temp])][:4]
        
        # print the top 4 feature names
        print(f'Component {row}: {[utils.get_name_from_id(i) for i in x_data.columns[indices].to_list()]}')

    to_return = pd.DataFrame(x)
    return to_return

    


def main():
    global OFFSET
    global LIMIT

    start_date = "2010-01-01"
    end_date = "2020-01-01"
    df_companies, df_stocks = get_data(dataset, start=start_date, end=end_date, cols=cols)
    dfs = []
    for i in range(3):
        df_mei = utils.get_mei(start=start_date, end=end_date, limit=LIMIT, offset=OFFSET)
        OFFSET += LIMIT
        dfs.append(utils.combine_datasets(df_mei))
        print("Done with {} datasets".format(20*(i+1)))
    

    df = reduce(lambda left,right: pd.merge(left,right,on="date"), dfs)

    print(df.shape)

    datasets = [df, df_stocks]
    datasets = utils.remove_day_from_date(datasets)

    df_mei, df_stocks = datasets[0], datasets[1]

    df_mei_reduced = perform_pca(df_mei)

    X_train, X_test, y_train, y_test = train_test_split(df_mei_reduced, df_stocks, test_size=0.2, random_state=42 ,shuffle=True)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    





if __name__ == "__main__":
    main()












