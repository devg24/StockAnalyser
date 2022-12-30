import pandas as pd
from full_fred.fred import Fred
import numpy as np
from functools import reduce



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
    r = fred.get_series_matching_tags(["mei","monthly","usa"], limit=limit, offset=offset)
    datasets = r['seriess']
    to_return  = {i['id'] : fred.get_series_df(i['id']) for i in datasets}
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










    



