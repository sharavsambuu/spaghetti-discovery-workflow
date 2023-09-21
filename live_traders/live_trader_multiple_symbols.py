#%%
# References:
#   - https://github.com/sammchardy/python-binance/issues/868
#

#%%
from __future__ import annotations
from abc        import ABC, abstractmethod
from socket import timeout
from typing     import List

import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
import os
import requests
import dateutil
import pytz
import pickle
import joblib
import websocket
import json
import time
import configparser
import asyncio
import concurrent.futures
import datetime
from datetime import timezone
from binance.client import Client as BinanceClient
from binance.client import AsyncClient
from binance        import BinanceSocketManager
from binance        import ThreadedWebsocketManager
import pandas   as pd
import numpy    as np
import mlfinlab as fml
import talib
from ta import momentum, trend, volatility
from ta import volume as tavolume




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
# Global Variables 

CONFIG_NAMESPACE           = "SHARAV" #"SHARAV_TESTNET"
CONFIG_FILE                = "../binance_config.ini"

RISK                       = 2.0 # by percentage
LEVERAGE                   = 2.0 # by times

BASE_ASSET                 = "USDT"
SYMBOL                     = "BTCUSDT"

ORDER_TOLERANCE_PERCENTAGE = 60.0 # by percentage
ORDER_TOLERANCE_PIPS       = 60.0 # by price change




#%%


#%%
def cci_indicator(df_, length=40):
    hlc3 = (df_['High']+df_['Low']+df_['Close'])/3
    sma  = hlc3.rolling(length).mean()
    mad  = hlc3.rolling(length).apply(lambda x: pd.Series(x).mad())
    cci  = (hlc3-sma)/(0.015*mad)
    return cci


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


#%%
# If there is a need to interact Binance, we can use this class

class BinanceInterface:

    def __init__(self) -> None:
        conf = configparser.ConfigParser() 
        conf.read(CONFIG_FILE)
        print(f"BinanceInterface : using config namespace = {CONFIG_NAMESPACE}")
        self.api_key    = conf[CONFIG_NAMESPACE]['api'   ]
        self.api_secret = conf[CONFIG_NAMESPACE]['secret']
        self.bclient    = BinanceClient(self.api_key, self.api_secret)
        print("BinanceInterface : Initialized.")

    def inquire_futureacc_balance(self, asset="USDT"):
        balance = self.bclient.futures_account_balance(0)
        amount  = 0.0
        for item in balance:
            if item["asset"]==asset:
                amount = float(item["balance"])
                break
        return amount

    def get_lot_size(self, price, sl_price, asset="USDT", leverage=1.0, risk=2.0):
        balance_by_asset = self.inquire_futureacc_balance(asset=asset)
        if balance_by_asset<=0.0:
            print("BinanceInterface : There is no enough asset to trade!")
            return None
        print(f"BinanceInterface : balance by {asset} : {balance_by_asset}")
        sl_price_diff  =  abs(float(price)-float(sl_price))
        risk_by_dollar = (balance_by_asset*leverage)*risk/100.0
        risk_with_sl   = price*risk_by_dollar/sl_price_diff
        lot_size       = risk_with_sl/price
        return lot_size

    def check_in_position(self, symbol="BTCUSDT"):
        in_position = False
        positions = self.bclient.futures_position_information()
        for item in positions:
            if item["symbol"]==symbol:
                if abs(float(item["positionAmt"]))>0.0:
                    in_position = True
                    break
        side  = None
        amount = None
        if in_position==True:
            if float(item["positionAmt"])>0.0:
                side="LONG"
            else:
                side="SHORT"

            amount = float(item["positionAmt"])
        return in_position, side, amount

    def check_order_criterion(self, signal_price, tp_price, ticker_price, tolerance_percentage=30.0, tolerance_pips=60.0):
        can_open_order = False
        price_diff     = abs(float(signal_price)-float(tp_price))
        tolerable_pips = price_diff*tolerance_percentage/100.0
        ticker_diff    = abs(float(signal_price)-float(ticker_price))
        if tolerable_pips>ticker_diff:
            can_open_order = True
        return can_open_order

    def open_long_position(self, asset, symbol, probability, risk, leverage, signal_price, sl_price, tp_price):
        position, side, amount = self.check_in_position(symbol=symbol)
        if position:
            print("There is a position already created, so cannot create this LONG position.")
            return False

        self.bclient.futures_cancel_all_open_orders(symbol=symbol)

        position_risk = risk
        if probability<=0.6:
            position_risk = position_risk/2.0

        lot_size = self.get_lot_size(signal_price, sl_price, asset, leverage, position_risk)
        print(f"lot size for risk={position_risk} diff={abs(signal_price-sl_price)}: ", lot_size)

        self.bclient.futures_create_order(
            symbol   = symbol,
            side     = self.bclient.SIDE_BUY,
            type     = self.bclient.FUTURE_ORDER_TYPE_MARKET,
            quantity = round(lot_size, 3)
        )
        self.bclient.futures_create_order(
            symbol        = symbol,
            type          = self.bclient.FUTURE_ORDER_TYPE_STOP_MARKET,
            side          = self.bclient.SIDE_SELL,
            stopPrice     = sl_price,
            closePosition = True
        )
        self.bclient.futures_create_order(
            symbol        = symbol,
            type          = self.bclient.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            side          = self.bclient.SIDE_SELL,
            stopPrice     = tp_price,
            closePosition = True
        )

        return True


    def open_short_position(self, asset, symbol, probability, risk, leverage, signal_price, sl_price, tp_price):
        position, side, amount = self.check_in_position(symbol=symbol)
        if position:
            print("There is a position already created, so cannot create this SHORT position.")
            return False

        self.bclient.futures_cancel_all_open_orders(symbol=symbol)

        position_risk = risk
        if probability<=0.6:
            position_risk = position_risk/2.0

        lot_size = self.get_lot_size(signal_price, sl_price, asset, leverage, position_risk)
        print(f"lot size for risk={position_risk} diff={abs(signal_price-sl_price)}: ", lot_size)

        self.bclient.futures_create_order(
            symbol   = symbol,
            side     = self.bclient.SIDE_SELL,
            type     = self.bclient.FUTURE_ORDER_TYPE_MARKET,
            quantity = round(lot_size, 3)
        )
        self.bclient.futures_create_order(
            symbol        = symbol,
            type          = self.bclient.FUTURE_ORDER_TYPE_STOP_MARKET,
            side          = self.bclient.SIDE_BUY,
            stopPrice     = sl_price,
            closePosition = True
        )
        self.bclient.futures_create_order(
            symbol        = symbol,
            type          = self.bclient.FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
            side          = self.bclient.SIDE_BUY,
            stopPrice     = tp_price,
            closePosition = True
        )

        return True

    def market_liquidate_positions(self, symbol):
        position, side, amount = self.check_in_position(symbol=symbol)
        if position:
            print(f"Current position : side={side}, amount={amount}")
            if side=="LONG":
                print("Closing LONG position...")
                self.bclient.futures_cancel_all_open_orders(symbol=symbol)
                self.bclient.futures_create_order(
                    symbol   = symbol,
                    side     = self.bclient.SIDE_SELL,
                    type     = self.bclient.FUTURE_ORDER_TYPE_MARKET,
                    quantity = abs(amount)
                )
            elif side=="SHORT":
                print("Closing SHORT position...")
                self.bclient.futures_cancel_all_open_orders(symbol=symbol)
                self.bclient.futures_create_order(
                    symbol   = symbol,
                    side     = self.bclient.SIDE_BUY,
                    type     = self.bclient.FUTURE_ORDER_TYPE_MARKET,
                    quantity = abs(amount)
                )
                pass
        else:
            print("There is no position so no need to liquidate.")

    def load_history(self, symbol="BTCUSDT", interval="1 week ago UTC"):
        klines = self.bclient.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1MINUTE, interval)
        df = pd.DataFrame(klines, columns='DateTime Open High Low Close Volume a b c d e f'.split())
        df = df.astype({'DateTime':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
        df = df.set_index('DateTime')
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    

#%%


#%%


#%%


#%%
# responsible for detecting new events

class BinanceLive(Subject):

    _observers : List[Observer] = []

    symbols        = []
    df_            = {}
    df             = {}
    df_xm_         = {}
    df_xm          = {}
    bar_count_1m   = {}
    bar_count_xm   = {}
    position_count = {}

    def __init__(self, symbols) -> None:
        super().__init__()
        self.binterface = BinanceInterface()
        self.symbols    = symbols

    def attach(self , observer: Observer):
        self._observers.append(observer)
    def deatach(self, observer: Observer):
        self._observers.remove(observer)
    def notify(self, symbol):
        for observer in self._observers:
            if observer.symbol==symbol:
                observer.update(self)

    def run(self):
        async def get_all_klines(bsm):
            streams = ["%s@kline_1m"%str(symbol).lower() for symbol in self.symbols]
            async with bsm.multiplex_socket(streams) as stream:
                while True:
                    res = await stream.recv()
                    if res:
                        k_line = res['data']["k"]
                        new_df = pd.DataFrame(
                            [[k_line["t"], k_line["o"], k_line["h"], k_line["l"], k_line["c"], k_line["v"]]], 
                            columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
                        new_df = new_df.astype({'DateTime':'datetime64[ms]', 'Open':float, 'High':float, 'Low':float, 'Close':float, 'Volume':float})
                        new_df = new_df.set_index('DateTime')

                        self.integrate_1m_df(str(res['data']['s']).strip().lower(), new_df)

                        for symbol in self.symbols:
                            if symbol in self.df.keys():
                                if (len(self.df[symbol])>self.bar_count_1m[symbol]):
                                    self.bar_count_1m[symbol] = len(self.df[symbol])
                                    self.update_indicators(symbol)
                                    if (len(self.df_xm[symbol])>self.bar_count_xm[symbol]):
                                        self.bar_count_xm[symbol] = len(self.df_xm[symbol])
                                        new_position_count = len(self.df[symbol][(self.df[symbol]['position']==1)|(self.df[symbol]['position']==-1)])
                                        if (self.position_count[symbol]>new_position_count):
                                            self.position_count[symbol] = new_position_count
                                            self.notify(symbol)
                                    print(f"1m dataframe of {symbol} is renewed...")
                                    #self.notify(symbol)

        async def klines_listener():
            async_bclient = await AsyncClient.create(self.binterface.api_key, self.binterface.api_secret)
            bsm = BinanceSocketManager(async_bclient)
            await asyncio.gather(get_all_klines(bsm))

        loop = asyncio.get_event_loop()
        loop.run_until_complete(klines_listener())

    def initialize(self, interval="2 days ago UTC"):
        for symbol in self.symbols:
            self.bar_count_1m  [symbol] = 0
            self.bar_count_xm  [symbol] = 0
            self.position_count[symbol] = 0
        for symbol in self.symbols:
            self.df_[symbol] = self.binterface.load_history(symbol=symbol.upper(), interval=interval)
            self.bar_count_1m[symbol] = len(self.df_[symbol])
            self.df_xm_[symbol] = self.df_[symbol].resample("5Min").agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
            self.bar_count_xm[symbol] = len(self.df_xm_[symbol])
            print(f"{symbol} history is loaded...")

        print(f"histories are loaded...")

    def integrate_1m_df(self, symbol, new_df):
        self.df_[symbol] = pd.concat([self.df_[symbol], new_df])
        self.df_[symbol] = self.df_[symbol][~self.df_[symbol].index.duplicated(keep='last')]
        self.df [symbol] = self.df_[symbol].iloc[:-1]

        self.df_xm_[symbol] = self.df_[symbol].resample("5Min").agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        self.df_xm [symbol]  = self.df_xm_[symbol].iloc[:-1]

    def update_indicators(self, symbol):
        self.df_xm[symbol]['cci'            ] = cci_indicator(self.df_xm[symbol])
        self.df   [symbol]['volatility_tpsl'] = fml.util.get_daily_vol(close=self.df[symbol]['Close'], lookback=840)

        upper_threshold =  200.0
        lower_threshold = -200.0
        self.df_xm[symbol]['position'] = 0
        self.df_xm[symbol].loc[(self.df_xm[symbol]['cci'].shift(1)<upper_threshold)&(self.df_xm[symbol]['cci']>=upper_threshold), 'position'] = -1
        self.df_xm[symbol].loc[(self.df_xm[symbol]['cci'].shift(1)>lower_threshold)&(self.df_xm[symbol]['cci']<=lower_threshold), 'position'] =  1

        self.df[symbol]['position'] = 0
        for idx, row in self.df_xm[symbol][(self.df_xm[symbol]['position']==1)|(self.df_xm[symbol]['position']==-1)].iterrows():
            if idx in self.df[symbol].index:
                self.df[symbol].loc[idx, 'position'] = row.position

        self.df[symbol]['position'] = self.df[symbol]['position'].shift(5)


#%%


#%%
class BtcusdtCCIMetaTrigger(Observer):
    symbol = ""
    def __init__(self, symbol) -> None:
        super().__init__()
        self.inference_tolerance_by_minutes = 8
        self.symbol = symbol
        self.rf     = joblib.load("../asset_btcusdt/meta-cci/model/btcusdt_rf.save")
        print(f"{symbol} observer is initialized and associated ML model is loaded...")
    def update(self, subject: Subject) -> None:
        print("New signal is created and considering some positions ")

        df = subject.df[self.symbol].copy()

        features = [
            'm_rsi_12', 'HT_TRENDMODE', 't_trix_400', 't_macdd', 't_adx_25', 't_adx_12', 't_adx_6',
            't_trix', 't_dpo', 'DEMA', 'close_plus_minus_20', 'log_lag_3', 'STOCHRSI_fastk', 't_trix_180',
            'vm_fi_25', 'LINEARREG_ANGLE', 'vm_cmf_90', 'vl_atr_300', 'm_rsi_400', 'LINEARREG', 't_dpo_50',
            'BBANDS_upperband', 'BBANDS_lowerband', 'HT_PHASOR_inphase', 'log_lag_7',
            'HT_DCPHASE', 'vm_mfi_400', 'TRANGE', 'vm_mfi_90', 't_dpo_180', 'volume', 'position'
        ]

        df['m_rsi_12'           ] = momentum.rsi(df['Close'], window=12)
        df['HT_TRENDMODE'       ] = talib.HT_TRENDMODE(df['Close'])
        df['t_trix_400'         ] = trend.trix(df['Close'], window=400)
        df['t_macdd'            ] = trend.MACD(df['Close']).macd_diff()
        df['t_adx_25'           ] = trend.adx(df['High'], df['Low'], df['Close'], window=25)
        df['t_adx_12'           ] = trend.adx(df['High'], df['Low'], df['Close'], window=12)
        df['t_adx_6'            ] = trend.adx(df['High'], df['Low'], df['Close'], window=6)
        df['t_trix'             ] = trend.trix(df['Close'])
        df['t_dpo'              ] = trend.dpo(df['Close'])
        df['hilo'               ] = (df['High']+df['Low'])/2
        df['DEMA'               ] = talib.DEMA(df['Close'], timeperiod=30)-df['hilo']
        df['momentum_1'         ] = df['Close'].pct_change(periods=1)
        df['close_sign'         ] = df['momentum_1'].apply(np.sign)
        df['close_plus_minus_20'] = df['close_sign'].rolling(20).apply(lambda x: x.sum())
        df['log_ret'            ] = np.log(df['Close'])-np.log(df['Close'].shift(1))
        df['log_lag_3'          ] = df['log_ret'].shift(3)
        df['STOCHRSI_fastk'     ], _ = talib.STOCHRSI(df['Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df['t_trix_180'         ] = trend.trix(df['Close'], window=180)
        df['vm_fi_25'           ] = tavolume.force_index(df['Close'], df['Volume'], window=25)
        df['LINEARREG_ANGLE'    ] = talib.LINEARREG_ANGLE(df['Close'], timeperiod=14)
        df['vm_cmf_90'          ] = tavolume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=90)
        df['vl_atr_300'         ] = volatility.average_true_range(df['High'], df['Low'], df['Close'], window=300)
        df['m_rsi_400'          ] = momentum.rsi(df['Close'], window=400)
        df['LINEARREG'          ] = talib.LINEARREG(df['Close'], timeperiod=14)-df['Close']
        df['t_dpo_50'           ] = trend.dpo(df['Close'], window=50)
        df['BBANDS_upperband'   ], _, df['BBANDS_lowerband'] = talib.BBANDS(df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df['HT_PHASOR_inphase'  ], _ = talib.HT_PHASOR(df['Close'])
        df['log_lag_7'          ] = df['log_ret'].shift(7)
        df['HT_DCPHASE'         ] = talib.HT_DCPHASE(df['Close'])
        df['vm_mfi_400'         ] = tavolume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=400)
        df['TRANGE'             ] = talib.TRANGE(df['High'], df['Low'], df['Close'])
        df['vm_mfi_90'          ] = tavolume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=90)
        df['t_dpo_180'          ] = trend.dpo(df['Close'], window=180)
        df['volume'             ] = df['Volume']

        inferencable_df = df[(df['position']==1)|(df['position']==-1)][features]
        #print(inferencable_df.columns[inferencable_df.isna().any()].tolist())
        inferencable_df.dropna(inplace=True)

        if len(inferencable_df)>1:
            dt         = datetime.datetime.now(timezone.utc)
            utc_now    = dt.replace(tzinfo=timezone.utc)
            last_record_dt  = pd.to_datetime(inferencable_df.index[-1], unit='s', utc=True).to_pydatetime()
            difference      = utc_now - last_record_dt
            seconds_in_day  = 24*60*60
            diff_by_minutes = divmod(difference.days*seconds_in_day+difference.seconds, 60)
            elapsed_minutes = diff_by_minutes[0]
            if (elapsed_minutes<self.inference_tolerance_by_minutes):
                probability = round(float(self.rf.predict_proba(inferencable_df.iloc[-1:])[0][1]), 3)
                print(f"Probability : {probability}")
                print(inferencable_df)
                position        = int  (inferencable_df.iloc[-1]['position'       ])
                price           = float(inferencable_df.iloc[-1]['Close'          ])
                volatility_tpsl = float(inferencable_df.iloc[-1]['volatility_tpsl'])
                pt_sl           = [0.7, 0.7]
                risk            = 2.5 # by percentage
                leverage        = 1.0
                act_threshold   = 0.55

                if position==1 and probability>=act_threshold:
                    print(f"Opening long position with probability={probability} : ")
                    tp_price = np.exp(np.log(price)+volatility_tpsl*pt_sl[0])
                    sl_price = np.exp(np.log(price)-volatility_tpsl*pt_sl[1])
                    try:
                        result = subject.binterface.open_long_position(
                            asset        = BASE_ASSET,
                            symbol       = self.symbol.upper(),
                            probability  = probability,
                            risk         = risk,
                            leverage     = leverage,
                            signal_price = price,
                            sl_price     = sl_price,
                            tp_price     = tp_price
                        )
                        if result:
                            print(f"successfully created long position at price={price}, tp={tp_price}, sl={sl_price}")
                        else:
                            print("couldn't create long position.")
                    except Exception as ex:
                        print("Exception at opening long position :")
                        print(f"{ex}")
                        pass
                    pass

                if position==-1 and probability>=act_threshold:
                    print(f"Opening short position with probability={probability} : ")
                    sl_price = np.exp(np.log(price)+volatility_tpsl*pt_sl[1])
                    tp_price = np.exp(np.log(price)-volatility_tpsl*pt_sl[0])
                    try:
                        result = subject.binterface.open_short_position(
                            asset        = BASE_ASSET,
                            symbol       = self.symbol.upper(),
                            probability  = probability,
                            risk         = risk,
                            leverage     = leverage,
                            signal_price = price,
                            sl_price     = sl_price,
                            tp_price     = tp_price
                            )
                        if result:
                            print(f"successfully created short position at price={price}, sl={sl_price}, tp={tp_price}")
                        else:
                            print("couldn't create short position.")

                    except Exception as ex:
                        print("Exception at opening short position :")
                        print(f"{ex}")
                        pass
                    pass

                pass


#%%


#%%
class EthusdtCCIMetaTrigger(Observer):
    symbol = ""
    def __init__(self, symbol) -> None:
        super().__init__()
        self.symbol = symbol
        print(f"{symbol} observer is initialized.")
    def update(self, subject: Subject) -> None:
        #print(f"update for {self.symbol} ...")
        pass


#%%


#%%


#%%
if __name__=="__main__":

    symbols = [
        "BTCUSDT", "BNBUSDT", "ETHUSDT",
        #"MATICUSDT", "SOLUSDT", "XRPUSDT", "LTCUSDT", "LINKUSDT" , "ADAUSDT", "DASHUSDT"
        ]

    blive = BinanceLive([str(symbol).lower() for symbol in symbols])
    blive.initialize(interval="2 days ago UTC")

    btcusdt_ccimeta = BtcusdtCCIMetaTrigger("btcusdt")
    blive.attach(btcusdt_ccimeta)

    ethusdt_ccimeta = EthusdtCCIMetaTrigger("ethusdt")
    blive.attach(ethusdt_ccimeta)

    try:
        blive.run()
    except KeyboardInterrupt as e:
        print("Interrupted...")
        sys.exit(0)


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


#%%

