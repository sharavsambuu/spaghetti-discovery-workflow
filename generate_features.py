#%%
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy  as np
import talib
from ta import momentum, trend, volatility
from ta import volume as tavolume
import mlfinlab as fml


#%%
asset = str(sys.argv[1]).upper()


#%%
df = pd.read_csv(f"./data/{asset}/{asset}-1m.csv", parse_dates=True, index_col="timestamp")

df

#%%


#%%
# feature generation
# https://github.com/sharavsambuu/FXCM_currency/blob/master/Automated%20Algo.ipynb
# https://github.com/sharavsambuu/Alternative-Bars/blob/master/strategy/meta-labeling.ipynb
# https://github.com/romachandr/FML_lib/blob/master/features_creating.py
# https://www.kaggle.com/bapireddy/trend-following-strategy-on-btc
# https://github.com/sharavsambuu/mlbot_tutorial_richmanbtc_wannabebotter/blob/master/work/tutorial.ipynb


def create_features(input_df):
    df = input_df.copy(deep=True)
    opens  = df['op'    ]
    high   = df['hi'    ]
    low    = df['lo'    ]
    close  = df['cl'    ]
    volume = df['volume']

    print('step 1...')

    df['m_rsi'  ] = momentum.rsi(close)
    df['m_roc'  ] = momentum.roc(close)
    df['m_wr'   ] = momentum.williams_r(high, low, close)
    df['vm_cmf' ] = tavolume.chaikin_money_flow(high, low, close, volume)
    df['vm_mfi' ] = tavolume.money_flow_index(high, low, close, volume)
    df['vm_fi'  ] = tavolume.force_index(close, volume)
    df['vm_eom' ] = tavolume.ease_of_movement(high, low, volume)
    df['vl_bbp' ] = volatility.bollinger_pband(close)
    df['vl_atr' ] = volatility.average_true_range(high, low, close)
    df['t_macdd'] = trend.MACD(close).macd_diff()
    df['t_trix' ] = trend.trix(close)
    df['t_cci'  ] = trend.cci(high, low, close)
    df['t_dpo'  ] = trend.dpo(close)
    df['t_kst'  ] = trend.kst(close)
    df['t_adx'  ] = trend.adx(high, low, close)

    momentums    = [1, 2, 3, 4, 5, 6]
    volatilities = [3, 6, 10, 25, 30, 45]
    autocorrs    = [1, 2, 3, 4, 5]
    log_rets     = [1, 2, 3, 4, 5, 6, 7]
    df['log_ret'] = np.log(close) - np.log(close.shift(1))

    print('step 2...')

    for period in momentums:
        df[f"momentum_{period}"] = close.pct_change(periods=period)

    print('step 3...')

    for period in volatilities:
        df[f"volatility_{period}"] = df['log_ret'].rolling(
            window      = period,
            min_periods = period, 
            center      = False
            ).std()

    print('step 4...')

    autocorr_window = 10
    for corr_lag in autocorrs:
        df[f"autocorr_{corr_lag}"] = df['log_ret'].rolling(
            window      = autocorr_window, 
            min_periods = autocorr_window, 
            center      = False
            ).apply(lambda x: x.autocorr(lag=corr_lag), raw=False)

    print('step 5...')

    for lag in log_rets:
        df[f"log_lag_{lag}"] = df['log_ret'].shift(lag)

    print('step 6...')

    for w in [6, 12, 25, 50, 90, 120, 180, 300, 400]:
        df[f"m_rsi_{w}"        ] = momentum.rsi(close, window=w)
        df[f"m_roc_{w}"        ] = momentum.roc(close, window=w)
        df[f"m_wr_{w}"         ] = momentum.williams_r(high, low, close, lbp=w)
        df[f"vm_cmf_{w}"       ] = tavolume.chaikin_money_flow(high, low, close, volume, window=w)
        df[f"vm_mfi_{w}"       ] = tavolume.money_flow_index(high, low, close, volume, window=w)
        df[f"vm_fi_{w}"        ] = tavolume.force_index(close, volume, window=w)
        df[f"vm_eom_{w}"       ] = tavolume.ease_of_movement(high, low, volume, window=w)
        df[f"vl_bbp_{w}"       ] = volatility.bollinger_pband(close, window=w)
        df[f"vl_atr_{w}"       ] = volatility.average_true_range(high, low, close, window=w)
        df[f"t_macd_{w*2}_{w}" ] = trend.MACD(close, window_fast=w, window_slow=w*2).macd_diff()
        df[f"t_trix_{w}"       ] = trend.trix(close, window=w)
        df[f"t_cci_{w}"        ] = trend.cci(high, low, close, window=w)
        df[f"t_dpo_{w}"        ] = trend.dpo(close, window=w)
        df[f"t_adx_{w}"        ] = trend.adx(high, low, close, window=w)

    print('step 7...')

    for w in [15, 30, 50, 120, 240]:
        df[f"daily_volatility_{w}"] = fml.util.get_daily_vol(
            close    = close,
            lookback = w
        )

    print('step 8...')

    df[f"stochastic_k"] = momentum.stochrsi_k(close)
    df[f"stochastic_d"] = momentum.stochrsi_d(close)

    # signing
    df[f"close_sign"  ] = df['momentum_1'].apply(np.sign)

    # plus-minus
    df[f"close_plus_minus_5" ] = df['close_sign'].rolling(5 ).apply(lambda x: x.sum())
    df[f"close_plus_minus_20"] = df['close_sign'].rolling(20).apply(lambda x: x.sum())
    df[f"close_plus_minus_40"] = df['close_sign'].rolling(40).apply(lambda x: x.sum())

    # logs
    df[f"high_log"  ] = high.apply(np.log)
    df[f"close_log" ] = close.apply(np.log)

    df[f"volume_log"     ] = volume.apply(np.log)
    df[f"volume_norm"    ] = volume/1000.0
    df[f"volume_norm_log"] = df['volume_norm'].apply(np.log)

    # volatility normalized prices, might be useful
    df['price_volatility_norm'    ] = (close/df['daily_volatility_50'])
    df['price_volatility_norm_log'] = df['price_volatility_norm'].apply(np.log)

    print('step 9...')

    # moving averages
    for w in [6, 12, 25, 50, 110, 220, 440]:
        df[f"sma_{w}"    ] = trend.sma_indicator(close, window=w)
        df[f"sma_{w}_log"] = df[f"sma_{w}"].apply(np.log)
        df[f"ma_{w}"     ] = close/df[f"sma_{w}"]-1.0

    print('step 10...')

    # richmanbtc's features
    hilo = (df['hi'] + df['lo']) / 2
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] -= hilo
    df['BBANDS_middleband'] -= hilo
    df['BBANDS_lowerband'] -= hilo
    df['DEMA'] = talib.DEMA(close, timeperiod=30) - hilo
    df['EMA'] = talib.EMA(close, timeperiod=30) - hilo
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(close) - hilo
    df['KAMA'] = talib.KAMA(close, timeperiod=30) - hilo
    df['MA'] = talib.MA(close, timeperiod=30, matype=0) - hilo
    df['MIDPOINT'] = talib.MIDPOINT(close, timeperiod=14) - hilo
    df['SMA'] = talib.SMA(close, timeperiod=30) - hilo
    df['T3'] = talib.T3(close, timeperiod=5, vfactor=0) - hilo
    df['TEMA'] = talib.TEMA(close, timeperiod=30) - hilo
    df['TRIMA'] = talib.TRIMA(close, timeperiod=30) - hilo
    df['WMA'] = talib.WMA(close, timeperiod=30) - hilo

    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    df['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high, low, timeperiod=14)
    df['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    df['BOP'] = talib.BOP(opens, high, low, close)
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['DX'] = talib.DX(high, low, close, timeperiod=14)
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['TRIX'] = talib.TRIX(close, timeperiod=30)
    df['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    df['AD'] = talib.AD(high, low, close, volume)
    df['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['OBV'] = talib.OBV(close, volume)

    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    df['NATR'] = talib.NATR(high, low, close, timeperiod=14)
    df['TRANGE'] = talib.TRANGE(high, low, close)

    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close)
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close)
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close)
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    df['BETA'] = talib.BETA(high, low, timeperiod=5)
    df['CORREL'] = talib.CORREL(high, low, timeperiod=30)
    df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14) - close
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)
    df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14)
    df['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1)

    #df = df.dropna()

    return df


#%%


#%%

df = create_features(df)

df

#%%


#%%
df = df.dropna()

df

#%%
df.to_csv(f"./data/{asset}/{asset}-features-1m.csv", index=True, header=True)

#%%


#%%
#loaded_df = pd.read_csv(f"./data/{asset}/{asset}-features-1m.csv", parse_dates=True, index_col="timestamp")

#loaded_df

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

