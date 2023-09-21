#%%
from __future__ import annotations
from abc        import ABC, abstractmethod
from typing     import List

import sys
sys.path.insert(0, '../../..')

import warnings
warnings.filterwarnings("ignore")
import joblib
import requests
import configparser
import asyncio
from binance.client import Client as BinanceClient
from binance.client import AsyncClient
from binance        import BinanceSocketManager
import pandas as pd
import numpy  as np

from ta import momentum, trend, volatility
from ta import volume as tavolume

import mlfinlab as fml

from cgi import test
from multiprocessing.sharedctypes import Value
from yaml import parse
import glob
import warnings
import requests
import dateutil
import pytz
import traceback
import pandas            as pd
import pandas_ta         as ta
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import mlfinlab          as fml
from   mlfinlab          import sample_weights
import pyfolio           as pf
from scipy.stats         import norm
from backtesting         import Backtest, Strategy
from backtesting.lib     import crossover

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (20,12)



#%%


#%%
df = pd.read_csv("/home/sharav/src/binance-history-downloader/data/klines/BTCUSDT-1m-spot.csv", parse_dates=True, index_col="timestamp")

df

#%%


#%%


#%%
def get_daily_volatility(close, lookback=100):
    # daily vol re-indexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0.array].array - 1  # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0


#%%
class MLInterface:
    def __init__(self):
        self.primary_threshold = 0.65

        self.side_features = []
        self.meta_features = []
        with open("../../../asset_btcusdt/meta-ml/model/features_side.txt", "r") as f:
            self.side_features = f.readlines()[0].strip().split()
        with open("../../../asset_btcusdt/meta-ml/model/features_meta.txt", "r") as f:
            self.meta_features = f.readlines()[0].strip().split()
        print(f"side features : {self.side_features}")
        print(f"meta features : {self.meta_features}")
        self.side_rf = joblib.load("../../../asset_btcusdt/meta-ml/model/btcusdt_rf_side.save")
        self.meta_rf = joblib.load("../../../asset_btcusdt/meta-ml/model/btcusdt_rf_meta.save")
        print(self.side_rf)
        print(self.meta_rf)

    def do_inference(self, df):
        df["volatility_tpsl"] = get_daily_volatility(close=df['Close'], lookback=600)

        # primary features : ['ma_440', 'daily_volatility_30', 'vm_eom_25', 'vl_atr_180', 'vm_cmf_180']
        df["sma_440"            ] = trend.sma_indicator(df['Close'], window=440)
        df["ma_440"             ] = df['Close']/df[f"sma_440"]-1.0
        df["daily_volatility_30"] = fml.util.get_daily_vol(close=df['Close'], lookback = 30)
        df["vm_eom_25"          ] = tavolume.ease_of_movement(df['High'], df['Low'], df['Volume'], window=25)
        df["vl_atr_180"         ] = volatility.average_true_range(df['High'], df['Low'], df['Close'], window=180)
        df["vm_cmf_180"         ] = tavolume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=180)
        # meta features : ['vl_bbp', 't_adx_90', 'close_plus_minus_40']
        df["vl_bbp"             ] = volatility.bollinger_pband(df['Close'])
        df["t_adx_90"           ] = trend.adx(df['High'], df['Low'], df['Close'], window=90)
        df["momentum_1"         ] = df['Close'].pct_change(periods=1)
        df["close_sign"         ] = df['momentum_1'].apply(np.sign)
        df["close_plus_minus_40"] = df['close_sign'].rolling(40).apply(lambda x: x.sum())

        df = df.fillna(0.0)

        df['side'] = self.side_rf.predict(df[self.side_features]) 
        df.loc[df['side']==-1, 'prob'] = self.side_rf.predict_proba(df[df['side']==-1][self.side_features])[:,0]
        df.loc[df['side']== 1, 'prob'] = self.side_rf.predict_proba(df[df['side']== 1][self.side_features])[:,1]
        df['is_signal'] = False
        df.loc[(df['prob']>=self.primary_threshold)|(df['prob']>=self.primary_threshold), 'is_signal'] = True

        signal_indexes = df[df['is_signal']==True].index
        df['act'] = 0
        df.loc[signal_indexes, 'act'     ] = self.meta_rf.predict(df.loc[signal_indexes][self.meta_features])
        df.loc[signal_indexes, 'act_prob'] = self.meta_rf.predict_proba(df.loc[signal_indexes][self.meta_features])[:,1]

        return df


#%%


#%%
ml_inference = MLInterface()

#%%
df = ml_inference.do_inference(df)

df

#%%


#%%
df['is_signal'].value_counts()


#%%
df['position'] = df['side']

#%%


#%% 
df['position'].value_counts()

#%%
df['prob'].hist(bins=400)

#%%
df['volatility_tpsl'] = get_daily_volatility(close=df['Close'], lookback=840)

#%%

#%%


#%%


#%%
# Model performance evaluation on OOS data

import datetime
from backtesting     import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.lib import resample_apply

def SMA(array, n):
    return pd.Series(array).rolling(n).mean()

def kelly(prob_win, payout_perc):
    return (prob_win * payout_perc - (1 - prob_win)) / payout_perc

binance_commission = 0.0004  # Taker Fee rate is 0.04%

RR         = 1.2
sl_target  = 1.0
long_ptsl  = [round(sl_target*RR, 2), sl_target]
short_ptsl = [round(sl_target*RR, 2), sl_target]
ptsl       = [round(sl_target*RR, 2), round(sl_target*RR, 2)]


#%%


#%%
act_threshold = 0.75

class SideStrategy(Strategy):
    def init(self):
        super().init()
        self.last_position_dt = None

    def next(self):
        super().next()

        current_dt = self.data.index[-1]

        available_to_trade = True
        if len(self.trades)>=1:
            available_to_trade = False
        if not available_to_trade:
            if self.last_position_dt:
                delta_time = current_dt - self.last_position_dt
                if delta_time>datetime.timedelta(days=2):
                    self.position.close()
            return

        close_price     = self.data.Close[-1]
        volatility_tpsl = self.data.volatility_tpsl[-1]
        position        = self.data.position[-1]
        prob            = self.data.prob[-1]

        size=0.1

        if position==1 and prob>=act_threshold:
            ret_upper = np.exp(round((volatility_tpsl*long_ptsl[0]), 6))-1.0
            ret_lower = np.exp(round((volatility_tpsl*long_ptsl[1]), 6))-1.0
            price_upper = (ret_upper+1.0)*close_price
            price_lower = (ret_lower+1.0)*close_price
            delta_upper = abs(close_price-price_upper)
            delta_lower = abs(close_price-price_lower)
            price_tp = close_price+delta_upper
            price_sl = close_price-delta_lower
            self.buy(size=size, sl=price_sl, tp=price_tp)
            self.last_position_dt = current_dt

        if position==-1 and prob>=act_threshold:
            ret_upper = np.exp(round((volatility_tpsl*short_ptsl[1]), 6))-1.0
            ret_lower = np.exp(round((volatility_tpsl*short_ptsl[0]), 6))-1.0
            price_upper = (ret_upper+1.0)*close_price
            price_lower = (ret_lower+1.0)*close_price
            delta_upper = abs(close_price-price_upper)
            delta_lower = abs(close_price-price_lower)
            price_sl = close_price+delta_upper
            price_tp = close_price-delta_lower
            self.sell(size=size, sl=price_sl, tp=price_tp)
            self.last_position_dt = current_dt

bt = Backtest(
    df["2022-01-24":], 
    SideStrategy, 
    cash             = 100000000, 
    #commission       = binance_commission, 
    exclusive_orders = True
    )

stats = bt.run()

stats


#%%
stats_df = stats['_trades'][['ReturnPct', 'EntryTime']]
stats_df = stats_df.set_index('EntryTime')

pf.create_simple_tear_sheet(stats_df['ReturnPct'])


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%

