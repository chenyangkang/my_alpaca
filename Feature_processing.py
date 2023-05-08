import pandas as pd
import numpy as np
import os, sys, pytz
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import filterwarnings
filterwarnings('ignore')

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import wandb
import random
import pickle

import pandas as pd
import numpy as np
import alpaca
from alpaca_trade_api import REST
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import pytz

### get stock data
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient

import sys
sys.path = sys.path + ['../']

import my_secrets
import pickle

from warnings import filterwarnings
filterwarnings('ignore')

stock_code='TQQQ'



path = f'./historical_price_data/{stock_code}.csv'
data = pd.read_csv(path,index_col=0)
data.index = pd.Series([i.replace(tzinfo=None) for i in pd.to_datetime(data.index)]).dt.tz_localize(pytz.timezone('US/Eastern'))

a = data.resample('1min').mean().dropna()
a['change_rate'] = (a['close'] - a['open'])/a['open']
a['change_bi'] = np.where((a['close'] - a['open'])>0,1,0)
a['date'] = pd.to_datetime([i.date() for i in a.index])
a['year'] = a.date.dt.year
a['WOY'] = a.date.dt.week
a['WOY_year'] = [str(a)+str(b) for a,b in zip(a['WOY'],a['year'])]
a['MOY'] = a.date.dt.month
a['MOY_year'] = [str(a)+str(b) for a,b in zip(a['MOY'],a['year'])]
a['DOW'] = a.date.dt.day_of_week
a['DOY'] = a.date.dt.day_of_year
a['DOY_year'] = [str(a)+str(b) for a,b in zip(a['DOY'],a['year'])]
a['MOD'] = [i.hour*60+i.minute for i in a.index]
a['HOD'] = [i.hour for i in a.index]
a = a[(a.MOD>=9.5*60) & (a.MOD<=16*60)]
def calc_one_day(df):
    start = df.iloc[0,:]['close']
    df['change_from_today_start'] = (df['close'] - start)/start
    return df
a = a.groupby('date').apply(calc_one_day)
a['code'] = stock_code



#####
a = a.merge(
    a[['DOY_year','change_rate']].groupby('DOY_year').sum().reset_index(drop=False).rename(columns={'change_rate':'change_rate_DOY_year'}),
    left_on='DOY_year', right_on='DOY_year',how='left'
).merge(
    a[['WOY_year','change_rate']].groupby('WOY_year').sum().reset_index(drop=False).rename(columns={'change_rate':'change_rate_WOY_year'}),
    left_on='WOY_year', right_on='WOY_year',how='left'
).merge(
    a[['MOY_year','change_rate']].groupby('MOY_year').sum().reset_index(drop=False).rename(columns={'change_rate':'change_rate_MOY_year'}),
    left_on='MOY_year', right_on='MOY_year',how='left'
)





### predict based on past 15 days. 5min interval, 2h+30min+4h = 6.5h. ((6.5*60)/1) * 15 = 5850 block
# each interval being predicted using the past 1170 block

### predict based on past 15 weeks. 5min interval, 2h+30min+4h = 6.5h. ((6.5*60)/1) * 15 * 7 = 40950 block

### predict based on past 15 month. 5min interval, 2h+30min+4h = 6.5h. ((6.5*60)/1) * 15 * 30= 175500 block

feature_dim_minute = 5850
feature_dim_day = 30
feature_dim_week = 15
feature_dim_month = 15

mi_value = [i for i in a['change_rate'].values]
day_value = [i for i in a['change_rate_DOY_year'].values]
uniq_day_value = list(a['change_rate_DOY_year'].unique())
week_value = [i for i in a['change_rate_WOY_year'].values]
uniq_week_value = list(a['change_rate_WOY_year'].unique())
month_value = [i for i in a['change_rate_MOY_year'].values]
uniq_month_value = list(a['change_rate_MOY_year'].unique())

df = []
batch_count = 0
for index,v in tqdm(enumerate(zip(mi_value,day_value,week_value,month_value)), total=len(mi_value)):
    if index<feature_dim_month*(6.5*60)*22+10:
        continue
    
    ### month level, 15 month
    this_month_index = uniq_month_value.index(month_value[index])
    month_list = uniq_month_value[int(this_month_index-feature_dim_month):this_month_index][::-1] ### arround 25 days per month for trading
    
    ### week level, 15 week
    this_week_index = uniq_week_value.index(week_value[index])
    week_list = uniq_week_value[int(this_week_index-feature_dim_week):this_week_index][::-1] ### arround 5 days per week for trading
    
    ### day level, 30 days
    this_day_index = uniq_day_value.index(day_value[index])
    day_list = uniq_day_value[int(this_day_index-feature_dim_day):this_day_index][::-1]

    ### minitue level, 5850 minutes
    minute_list = mi_value[int(index-feature_dim_minute):index][::-1]
    
    df.append({
        **{'y':v[0]},
        **{f'm{i}':j for i,j in zip(list(range(feature_dim_month)),month_list)},
        **{f'w{i}':j for i,j in zip(list(range(feature_dim_week)),week_list)},
        **{f'd{i}':j for i,j in zip(list(range(feature_dim_day)),day_list)},
        **{f'mi{i}':j for i,j in zip(list(range(feature_dim_minute)),minute_list)}
    })
    
    if len(df)>=5120:
        df = pd.DataFrame(df)
        with open(f'./training_data/{stock_code}_processed_data_{batch_count}.pkl','wb') as f:
            pickle.dump(df,f)
        batch_count+=1
        df=[]



