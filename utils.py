import pandas as pd
from full_fred.fred import Fred
import numpy as np


LIMIT = 2
fred = Fred("./fred_api_key.txt")

def get_mei(start=None,end=None):
    '''
    @param start: start date
    @param end: end date

    @return to_return: list of dataframes of mei
    get Main Economic Indicators from fred
    '''
    if start and end:
        fred.observation_start = start
        fred.observation_end = end
    r = fred.get_series_matching_tags(["mei"], limit=LIMIT)
    datasets = r['seriess']
    to_return  = [fred.get_series_df(i['id']) for i in datasets]
    return to_return


# get_mei()







    



