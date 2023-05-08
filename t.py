import pandas as pd
import numpy as np
import os, sys, pytz
import matplotlib.pyplot as plt
import pickle
from warnings import filterwarnings
filterwarnings('ignore')

code_list = [i.split('.csv')[0] for i in os.listdir('./historical_price_data/')]



data_list = []
for code in code_list:
    path = f'./historical_price_data/{code}.csv'
    data = pd.read_csv(path,index_col=0)
    data.index = pd.Series([i.replace(tzinfo=None) for i in pd.to_datetime(data.index)]).dt.tz_localize(pytz.timezone('US/Eastern'))
    
    a = data.resample('1min').mean().dropna()
    a['date'] = [i.date() for i in a.index]
    a['MOD'] = [i.hour*60+i.minute for i in a.index]
    a['HOD'] = [i.hour for i in a.index]
    a = a[(a.MOD>=9.5*60) & (a.MOD<=16*60)]
    def calc_one_day(df):
        start = df.iloc[0,:]['close']
        df['change_from_today_start'] = (df['close'] - start)/start
        return df
    a = a.groupby('date').apply(calc_one_day)
    a.groupby('MOD').mean()['change_from_today_start'].plot()
#    plt.title(code)
#    plt.show()
    a['code'] = code
    data_list.append(a)

    
    

data_all = pd.concat(data_list)
with open('data_all.pkl','wb') as f:
    pickle.dump(data_all, f)


