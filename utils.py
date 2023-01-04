import pandas as pd
from full_fred.fred import Fred
import numpy as np
from functools import reduce
import datetime as dt



LIMIT = 10
fred = Fred("./fred_api_key.txt")
OFFSET = 0

def get_mei(start=None,end=None, limit=LIMIT, offset=OFFSET):
    '''
    @param start: start date
    @param end: end date

    @return to_return: list of dataframes of mei
    get Main Economic Indicators from fred
    '''

    if start:
        fred.observation_start = start
    if end:
        fred.observation_end = end
    to_return=dict()
    r = fred.get_series_matching_tags(["mei","monthly","usa", "all items"], limit=limit, offset=offset)
    datasets = r['seriess']
    to_return  = to_return.update({i['id'] : fred.get_series_df(i['id']) for i in datasets})

    #remove keys with same title
    titles = {i["title"] for i in datasets}

    to_return = {key: to_return[key] for key in to_return if get_name_from_id(key) not in titles}

    # print(to_return.keys())
    return to_return

def combine_datasets(data, on='date', exclude=["realtime_start","realtime_end"]):
    '''
    @param data: list of dataframes
    
    @return df: dataframe of all dataframes
    '''
    datasets = []
    for key in data:
        if "date" not in data[key].columns:
            continue
        data[key].rename(columns={"value": key}, inplace=True)
        data[key].drop(columns=exclude, inplace=True, errors = "ignore")
        datasets.append(data[key])
    
    
    df = reduce(lambda left,right: pd.merge(left,right,on=on), datasets)
    return df

def remove_day_from_date(datasets):
    '''
    @param datasets: list of dataframes
    @return datasets: list of dataframes with date column without day
    '''
    for key in datasets:
        key["date"] = pd.to_datetime(key["date"])
        key["date"] = key["date"].dt.strftime("%Y-%m")
    return datasets



def get_name_from_id(id):
    '''
    @param id: id of fred dataset
    @return name: name of fred dataset
    '''
    return fred.get_a_series(id)["seriess"][0]["title"]



# print(fred.get_all_tags(search_text="seasonally adjusted"))

test = get_mei()


    



