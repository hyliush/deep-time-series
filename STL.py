import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# statsmodels == 0.10.2
# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)


def get_statsmodels_df():
    """Return packaged data in a pandas.DataFrame"""
    # some hijinks to get around outdated statsmodels code
    dataset = sm.datasets.co2.load()
    data = pd.DataFrame(dataset.data)
    # start = data['index'][0].decode('utf-8')
    data = data.rename(columns={"index":'date'})
    index = pd.date_range(start=data.index[0], periods=len(data), freq='W-SAT')
    obs = pd.DataFrame(data['co2'].values, index=index, columns=['co2'])
    return obs

obs = get_statsmodels_df()

obs = (obs
       .resample('D')
       .mean()
       .interpolate('linear'))


decomp = decompose(obs, period=365)

decomp