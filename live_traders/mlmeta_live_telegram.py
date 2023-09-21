#%%
from __future__ import annotations
from abc        import ABC, abstractmethod
from typing     import List

import sys
sys.path.insert(0, '..')

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


#%%


#%%
# Global Variables 

CONFIG_NAMESPACE = "SHARAV"
CONFIG_FILE      = "../binance_config.ini"
SYMBOL           = "BTCUSDT"

#%%


#%%
# Classes for observer design pattern
#   - https://refactoring.guru/design-patterns/observer/python/example
#

class Observer(ABC):
    @abstractmethod
    def update(self, subject: Subject) -> None:
        pass

class Subject(ABC):
    @abstractmethod
    def attach(self, observer: Observer) -> None:
        pass
    @abstractmethod
    def deatach(self, observer: Observer) -> None:
        pass
    @abstractmethod
    def notify(self) -> None:
        pass


#%%


#%%
# If there is a need to interact Binance, we can use this class

class BinanceInterface:
    def __init__(self) -> None:
        conf = configparser.ConfigParser() 
        conf.read(CONFIG_FILE)
        self.api_key    = conf[CONFIG_NAMESPACE]['api'   ]
        self.api_secret = conf[CONFIG_NAMESPACE]['secret']
        self.bclient    = BinanceClient(self.api_key, self.api_secret)
        print("BinanceInterface : Initialized.")

    def load_history(self, symbol="BTCUSDT", interval="1 week ago UTC"):
        klines = self.bclient.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1MINUTE, interval)
        df = pd.DataFrame(klines, columns='DateTime Open High Low Close Volume a b c d e f'.split())
        df = df.astype({'DateTime':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
        df = df.set_index('DateTime')
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    

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

class MLInterface:
    def __init__(self):
        self.primary_threshold = 0.65

        self.side_features = []
        self.meta_features = []
        with open("../asset_btcusdt/meta-ml/model/features_side.txt", "r") as f:
            self.side_features = f.readlines()[0].strip().split()
        with open("../asset_btcusdt/meta-ml/model/features_meta.txt", "r") as f:
            self.meta_features = f.readlines()[0].strip().split()
        print(f"side features : {self.side_features}")
        print(f"meta features : {self.meta_features}")
        self.side_rf = joblib.load("../asset_btcusdt/meta-ml/model/btcusdt_rf_side.save")
        self.meta_rf = joblib.load("../asset_btcusdt/meta-ml/model/btcusdt_rf_meta.save")
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
def notify_to_telegram(message):
    token   = ""
    clients = ['']
    def send_message(clientid, bot_message):
        send_text = 'https://api.telegram.org/bot' + token + '/sendMessage?chat_id=' + clientid + '&parse_mode=Markdown&text=' + bot_message
        response = requests.get(send_text)
        return response.json()
    for clientid in clients:
        send_message(clientid, message)


#%%


#%%
RR         = 2.0
sl_target  = 1.0
long_ptsl  = [round(sl_target*RR, 2), sl_target]
short_ptsl = [round(sl_target*RR, 2), sl_target]

#%%


#%%

#%%
# responsible for detecting new events

class BinanceLive(Subject):
    _observers : List[Observer] = []
    df_           = None 
    df            = None
    bar_count_1m  = 0

    def __init__(self) -> None:
        super().__init__()
        self.binterface  = BinanceInterface()
        self.mlinterface = MLInterface()

    def attach(self, observer: Observer):
        self._observers.append(observer)
    def deatach(self, observer: Observer):
        self._observers.remove(observer)
    def notify(self):
        for observer in self._observers:
            observer.update(self)

    def run(self):

        async def klines_listener():
            async_bclient = await AsyncClient.create(self.binterface.api_key, self.binterface.api_secret)
            bm = BinanceSocketManager(async_bclient)

            ts = bm.kline_socket(SYMBOL, interval=BinanceClient.KLINE_INTERVAL_1MINUTE)

            async with ts as tscm:
                while True:
                    res = await tscm.recv()
                    if res:
                        k_line = res["k"]
                        new_df = pd.DataFrame(
                            [[k_line["t"], k_line["o"], k_line["h"], k_line["l"], k_line["c"], k_line["v"]]], 
                            columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
                        new_df = new_df.astype({'DateTime':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
                        new_df = new_df.set_index('DateTime')

                        self.integrate_1m_df(new_df)

                        if self.df is not None:
                            if (len(self.df)>self.bar_count_1m):
                                self.bar_count_1m = len(self.df)
                                self.update_and_inference()
                                self.notify()

            await async_bclient.close_connection()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(klines_listener())

        pass

    def initialize(self):
        self.df_ = self.binterface.load_history(interval="3 days ago UTC")
        print("BLive : history is loaded...")
        print(self.df_)

    def integrate_1m_df(self, new_df):
        self.df_    = pd.concat([self.df_, new_df])
        self.df_    = self.df_[~self.df_.index.duplicated(keep='last')]
        self.df     = self.df_.iloc[:-1]
    
    def update_and_inference(self):
        print("Updating...")
        print(self.df)
        self.df = self.mlinterface.do_inference(df=self.df)
        print(self.df)
        if self.df.iloc[-1]['is_signal'] == True:
            close_price     = float(self.df.iloc[-1]['Close'])
            position        = self.df.iloc[-1]['side']
            probability     = float(self.df.iloc[-1]['act_prob'])
            volatility_tpsl = self.df.iloc[-1]['volatility_tpsl']

            #size = 0.5
            #if probability>=0.6:
            #    size = 1.0

            if position==1 and probability>=0.55:
                ret_upper = np.exp(round((volatility_tpsl*long_ptsl[0]), 6))-1.0
                ret_lower = np.exp(round((volatility_tpsl*long_ptsl[1]), 6))-1.0
                price_upper = (ret_upper+1.0)*close_price
                price_lower = (ret_lower+1.0)*close_price
                delta_upper = abs(close_price-price_upper)
                delta_lower = abs(close_price-price_lower)
                price_tp = close_price+delta_upper
                price_sl = close_price-delta_lower
                notify_to_telegram(message=f"Long at price={round(close_price,5)}, TP={round(price_tp, 5)}, SL={round(price_sl, 5)}")

            if position==-1 and probability>=0.55:
                ret_upper = np.exp(round((volatility_tpsl*short_ptsl[1]), 6))-1.0
                ret_lower = np.exp(round((volatility_tpsl*short_ptsl[0]), 6))-1.0
                price_upper = (ret_upper+1.0)*close_price
                price_lower = (ret_lower+1.0)*close_price
                delta_upper = abs(close_price-price_upper)
                delta_lower = abs(close_price-price_lower)
                price_sl = close_price+delta_upper
                price_tp = close_price-delta_lower
                notify_to_telegram(message=f"Short at price={round(close_price,5)}, TP={round(price_tp, 5)}, SL={round(price_sl, 5)}")

        pass



#%%


#%%
class TelegramNotifier(Observer):
    def update(self, subject: Subject) -> None:
        pass


#%%


#%%
if __name__=="__main__":
    notify_to_telegram(message="NevMLMetaRR2 service is started.")

    blive = BinanceLive()
    blive.initialize()

    telegram_notifier = TelegramNotifier()
    blive.attach(telegram_notifier)

    try:
        blive.run()
    except KeyboardInterrupt as e:
        print("Interrupted...")
        sys.exit(0)


#%%


#%%


#%%


#%%