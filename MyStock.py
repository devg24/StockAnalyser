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

LIMIT = 1000
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
    e = len(df_companies)

    try:
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
            time.sleep(65)
        # save to csv
        # df_stock.to_csv("Stocks/{}.csv".format(i))
    except:
        stocks = utils.remove_day_from_date(stocks)
        df_stock = reduce(lambda left,right: pd.merge(left,right,on='date'), stocks)
    

    return df_companies, df_stock

def perform_pca(market_data, print_top_features=True):
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
    if print_top_features:
        for row in range(pca_components.shape[0]):
            # get the indices of the top 4 values in each row
            temp = np.argpartition(-(pca_components[row]), 4)
            
            # sort the indices in descending order
            indices = temp[np.argsort((-pca_components[row])[temp])][:4]
            
            # print the top 4 feature names
            print(f'Component {row}: {[utils.get_name_from_id(i) for i in x_data.columns[indices].to_list()]}')

    to_return = pd.DataFrame(x)

    # add date column back
    to_return['date'] = market_data['date']
    return to_return

    
def get_mei_data(start_date, end_date):
    
    global OFFSET
    global LIMIT
    dfs = []
    df_mei = utils.get_mei(start=start_date, end=end_date, limit=LIMIT, offset=OFFSET)
    i = 0
    while df_mei is not None and i < 20:
        OFFSET += LIMIT
        dfs.append(utils.combine_datasets(df_mei))
        df_mei = utils.get_mei(start=start_date, end=end_date, limit=LIMIT, offset=OFFSET)
        print("Done with {} datasets".format(LIMIT*(i+1)))
        i += 1
    df = reduce(lambda left,right: pd.merge(left,right,on="date"), dfs)
    return df

def main():

    start_date = "2010-01-01"
    end_date = "2020-01-01"

    df_companies, df_stocks = get_data(dataset, start=start_date, end=end_date, cols=cols)

    df = get_mei_data(start_date, end_date)

    df_mei_reduced = perform_pca(df,False)

    df_mei_reduced = df_mei_reduced[df_mei_reduced['date'] >= df_stocks['date'].min()]

    df_mei_reduced = df_mei_reduced[df_mei_reduced['date'] <= df_stocks['date'].max()]

    # cast all columns of df_stocks to float except date 
    df_stocks = df_stocks.astype({col: 'float64' for col in df_stocks.columns if col != 'date'})

    print(df_stocks.dtypes)


    X_train, X_test, y_train, y_test = train_test_split(df_mei_reduced, df_stocks, test_size=0.2, random_state=42 ,shuffle=True)


    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    y_train = y_train.drop(columns=['date'])
    y_test = y_test.drop(columns=['date'])

    print(X_train.shape, type(X_train))
    print(X_test.shape, type(X_test))
    print(y_train.shape, type(y_train))
    print(y_test.shape, type(y_test))

    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    # add input layer
    model.add(tf.keras.layers.Dense(100, input_dim=X_train.shape[1], activation='relu'))

    # add hidden layers
    # model.add(tf.keras.layers.Dense(100, activation='relu'))
    # model.add(tf.keras.layers.Dense(100, activation='relu'))

    # add output layer
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)
    model.evaluate(X_test, y_test)

    # temp , df_stocks_test = get_data(dataset, start="2020-01-01", end="2022-01-01", cols=cols)
    # df_mei_test = get_mei_data("2020-01-01", "2022-01-01")
    # df_mei_test = perform_pca(df_mei_test, False)
    # df_mei_test = df_mei_test[df_mei_test['date'] >= df_stocks_test['date'].min()]
    # df_mei_test = df_mei_test[df_mei_test['date'] <= df_stocks_test['date'].max()]
    # df_mei_test = df_mei_test.drop(columns=['date'])
    # df_stocks_test = df_stocks_test.astype({col: 'float64' for col in df_stocks_test.columns if col != 'date'})
    # df_stocks_test = df_stocks_test.drop(columns=['date'])

    

    # model.evaluate(df_mei_test, df_stocks_test, verbose=1)

    





if __name__ == "__main__":
    main()

