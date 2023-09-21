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
# Global Variables 

CONFIG_NAMESPACE           = "SHARAV_TESTNET"
CONFIG_FILE                = "../binance_config.ini"

RISK                       = 2.0 # by percentage
LEVERAGE                   = 2.0 # by times

BASE_ASSET                 = "USDT"
SYMBOL                     = "BTCUSDT"

ORDER_TOLERANCE_PERCENTAGE = 60.0 # by percentage
ORDER_TOLERANCE_PIPS       = 60.0 # by price change


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
        self.bclient    = BinanceClient(self.api_key, self.api_secret, testnet=True)
        print("BinanceInterface : Initialized.")

    def inquire_futureacc_balance(self, asset):
        balance = self.bclient.futures_account_balance()
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

    def check_in_position(self, symbol):
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
binterface = BinanceInterface()


#%%
df = binterface.load_history(interval="1 day ago UTC")

print(df)


#%%
asset   = "USDT"

balance = binterface.inquire_futureacc_balance(asset=asset)
print(f"balance : {balance}")

assert balance>0.0


#%%
in_position, _, _ = binterface.check_in_position(symbol="BTCUSDT")
print(f"in position with BTCUSDT : {in_position}")

in_position, _, _ = binterface.check_in_position(symbol="ETHUSDT")
print(f"in position with ETHUSDT : {in_position}")


#%%


#%%
# Long case

asset         = "USDT"
current_price = 23756.0
sl_price      = 23450.0
leverage      = 1.0
risk          = 0.05

lot_size = binterface.get_lot_size(price=current_price, sl_price=sl_price, asset=asset, leverage=leverage, risk=risk)

print(f"lot size for leverage={leverage}, diff={abs(current_price-sl_price)}, risk={risk} => {lot_size}")



#%%
# Open Long Position
asset         = "USDT"
symbol        = "BTCUSDT"
current_price = 25200.0
sl_price      = 22050.0
tp_price      = 26450.0
leverage      = 1.0
risk          = 5.0 # 5%

try:
    result = binterface.open_long_position(
        asset        = asset, 
        symbol       = symbol, 
        probability  = 0.65, 
        risk         = risk, 
        leverage     = leverage, 
        signal_price = current_price, 
        sl_price     = sl_price, 
        tp_price     = tp_price
        )
    assert result==True
except Exception as ex:
    print("Exception at creating long position :")
    print(f"{ex}")
    pass

time.sleep(1)

in_position, _, _ = binterface.check_in_position(symbol="BTCUSDT")
assert in_position==True

time.sleep(2)


#%%
# liquidate current position
symbol = "BTCUSDT"
try:
    binterface.market_liquidate_positions(symbol=symbol)
except Exception as ex:
    print("Exception at liquidating positions")
    print(f"{ex}")
    pass

time.sleep(2)

in_position, _, _ = binterface.check_in_position(symbol="BTCUSDT")
assert in_position==False


#%%
# Open Short Position
asset         = "USDT"
symbol        = "BTCUSDT"
current_price = 23756.0
sl_price      = 24250.0
tp_price      = 22005.0
leverage      = 1.0
risk          = 5.0 # 5%

try:
    result = binterface.open_short_position(
        asset        = asset, 
        symbol       = symbol, 
        probability  = 0.65, 
        risk         = risk, 
        leverage     = leverage, 
        signal_price = current_price, 
        sl_price     = sl_price, 
        tp_price     = tp_price
        )
    assert result==True
except Exception as ex:
    print("Exception at creating short position :")
    print(f"{ex}")
    pass


time.sleep(1)

in_position, _, _ = binterface.check_in_position(symbol="BTCUSDT")
assert in_position==True

time.sleep(2)


#%%
# liquidate current position
symbol = "BTCUSDT"
try:
    binterface.market_liquidate_positions(symbol=symbol)
except Exception as ex:
    print("Exception at liquidating positions :")
    print(f"{ex}")
    pass

time.sleep(1)

in_position, _, _ = binterface.check_in_position(symbol="BTCUSDT")
assert in_position==False

time.sleep(2)


#%%
print("Done.")

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

