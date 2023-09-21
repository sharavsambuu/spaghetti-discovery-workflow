#%%
import sys
sys.path.insert(0, '../../..')
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


#%%
# loading history with features...
print("reading dataset...")

df = pd.read_csv("../../../data/BTCUSDT/BTCUSDT-features-1m.csv", parse_dates=True, index_col="timestamp")

df

#%%


#%%


#%%
# Helper functions

def get_daily_volatility(close, lookback=100):
    # daily vol re-indexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))
    df0 = close.loc[df0.index] / close.loc[df0.array].array - 1  # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0

def cci_indicator(df_, length=40):
    hlc3 = (df_['hi']+df_['lo']+df_['cl'])/3
    sma  = hlc3.rolling(length).mean()
    mad  = hlc3.rolling(length).apply(lambda x: pd.Series(x).mad())
    cci  = (hlc3-sma)/(0.015*mad)
    cci_smooth = cci.ewm(span=5, min_periods=0, adjust=False, ignore_na=False).mean()
    return cci, cci_smooth


#%%


#%%


#%%
# CCI signal extraction

timeframe = 5

d = {
    'op'    : 'first', 
    'hi'    : 'max'  ,
    'lo'    : 'min'  ,
    'cl'    : 'last' ,
    'volume': 'sum'
    }
df_xm = df.resample(f"{timeframe}Min").agg(d)

df_xm['cci'], _ = cci_indicator(df_xm)
df_xm.dropna(inplace=True)

upper_threshold =  232.4135
lower_threshold = -232.4135

df_xm['position'] = 0
df_xm.loc[(df_xm['cci'].shift(1)<upper_threshold)&(df_xm['cci']>=upper_threshold), 'position'] = -1
df_xm.loc[(df_xm['cci'].shift(1)>lower_threshold)&(df_xm['cci']<=lower_threshold), 'position'] =  1

# saving signals to market state
df['position'] = 0
for idx, row in df_xm[(df_xm['position']==1)|(df_xm['position']==-1)].iterrows():
    if idx in df.index:
        df.loc[idx, 'position'] = row.position

# removing look-ahead bias by lagging signal
df['position'] = df['position'].shift(timeframe+1)

df.dropna(inplace=True)


#%%
df['position'].value_counts()


#%%
import joblib

short_features = open("../model/features_short.txt", 'r').read().strip().split()
long_features  = open("../model/features_long.txt" , "r").read().strip().split()

short_rf = joblib.load("../model/btcusdt_rf_short.save")
long_rf  = joblib.load("../model/btcusdt_rf_long.save" )


#%%
feature_columns = list(dict.fromkeys(short_features+long_features+["position", "op", "hi", "lo", "cl", "volume"]))
print(feature_columns, len(feature_columns))

#%%
len(feature_columns)

#%%
len(df.columns)

#%%
len(df[feature_columns].columns)

#%%
filtered_df = df[feature_columns]

len(filtered_df.columns)


#%%
# from here we starting to show symptom of the columns duplication
print(list(filtered_df.columns))

#%%
print(len(filtered_df.columns))

#%%


#%%


#%%


#%%


#%%
train_df = filtered_df[:"2022-02-02"].copy(deep=True)
test_df  = filtered_df["2022-02-04":].copy(deep=True)


#%%
# Daily volatility calculation which is helpful for defining tp and sl levels

w = 840 # 14 hours
train_df[f'volatility_tpsl'] = fml.util.get_daily_vol(close=train_df['cl'], lookback=w)
test_df [f'volatility_tpsl'] = fml.util.get_daily_vol(close=test_df ['cl'], lookback=w)


#%%
# remove nans
train_df.dropna(inplace=True)
test_df.dropna (inplace=True)

#%%


#%%
len(train_df.columns), len(list(dict.fromkeys(train_df.columns)))

#%%
len(test_df.columns), len(list(dict.fromkeys(test_df.columns)))

#%%
train_df['m_roc_300']


#%%
short_features

#%%
long_features

#%%


#%%
from backtesting     import Backtest, Strategy
from backtesting.lib import crossover


#%%
eval_feature_columns = list(dict.fromkeys(short_features+long_features+["position", "volatility_tpsl", "Open", "High", "Low", "Close", "Volume"]))

list(eval_feature_columns), len(eval_feature_columns)


#%%
test_df = test_df.rename(columns={
    "op"    : "Open" ,
    "hi"    : "High" ,
    "lo"    : "Low"  ,
    "cl"    : "Close",
    "volume": "Volume"
    })

test_df = test_df[eval_feature_columns]

test_df

#%%
test_df['m_roc_300']


#%%
train_df = train_df.rename(columns={
    "op"    : "Open" ,
    "hi"    : "High" ,
    "lo"    : "Low"  ,
    "cl"    : "Close",
    "volume": "Volume"
    })

train_df = train_df[eval_feature_columns]

train_df

#%%
train_df['m_roc_300']


#%%
len(train_df.columns), len(test_df.columns)

#%%
# Model performance evaluation on OOS data

pt_sl = [0.7, 0.7]

binance_commission = 0.001

long_trigger_threshold  = 0.55
long_green_threshold    = 0.6

short_trigger_threshold = 0.55
short_green_threshold   = 0.6

class CCIMetaStrategy(Strategy):
    long_model  = None
    short_model = None
    def init(self):
        super().init()
    def next(self):
        super().next()

        if self.long_model is None:
            print("please supply long side model...")
            return
        if self.short_model is None:
            print("please supply short side model...")
            return

        available_to_trade = True
        if len(self.trades)>=1:
            available_to_trade = False
        if not available_to_trade:
            return

        price      = self.data.Close[-1]
        volatility = self.data.volatility_tpsl[-1]

        short_state = []
        for feature in short_features:
            short_state.append(self.data[feature][-1])

        long_state = []
        for feature in long_features:
            long_state.append(self.data[feature][-1])

        # Long case
        if self.long_model is not None and self.data.position[-1]==1:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[0])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[1])
            probability = self.long_model.predict_proba([long_state])[0,1]
            size = 1.0
            if probability>=long_trigger_threshold and probability<long_green_threshold:
                size = 0.5
            if probability>=long_green_threshold:
                size=1.0
            if probability>=long_trigger_threshold:
                self.buy(size=size, sl=price_1, tp=price_0)

        # Short case
        if self.short_model is not None and self.data.position[-1]==-1:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[1])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[0])

            probability = self.short_model.predict_proba([short_state])[0,1]
            size = 1.0
            if probability>=short_trigger_threshold and probability<short_green_threshold:
                size = 0.5
            if probability>=short_green_threshold:
                size=1.0
            if probability>=short_trigger_threshold:
                self.sell(size=size, sl=price_0, tp=price_1)

bt = Backtest(
    test_df, 
    CCIMetaStrategy, 
    cash             = 100000000, 
    commission       = binance_commission, 
    exclusive_orders = True
    )

stats = bt.run(long_model=long_rf, short_model=short_rf)

print(stats)


#%%
stat_df = stats['_trades'][['ReturnPct', 'EntryTime']]
stat_df = stat_df.set_index('EntryTime')

pf.create_simple_tear_sheet(stat_df['ReturnPct'])


#%%
from pyfolio.plotting import plot_rolling_sharpe

plt.figure(figsize=(20, 14))
plot_rolling_sharpe(stat_df['ReturnPct'], rolling_window=21*3)


#%%


#%%


#%%


#%%
# Attempt to filter with moving average trend filter
# Model performance evaluation on OOS data

from backtesting.lib import resample_apply

pt_sl = [0.7, 0.7]

binance_commission = 0.001

def SMA(array, n):
    return pd.Series(array).rolling(n).mean()

class CCIMetaStrategy(Strategy):
    long_model  = None
    short_model = None

    def init(self):
        super().init()
        self.sma_fast = resample_apply('30min', SMA, self.data.Close, 50 )
        self.sma_slow = resample_apply('30min', SMA, self.data.Close, 200)

    def next(self):
        super().next()

        if self.long_model is None:
            print("Please supply long side model...")
            return
        if self.short_model is None:
            print("Please supply short side model...")
            return

        is_up_trend   = False
        is_down_trend = False
        if self.sma_fast[-1]>self.sma_slow[-1]:
            is_up_trend = True
        if self.sma_fast[-1]<self.sma_slow[-1]:
            is_down_trend = True

        available_to_trade = True
        if len(self.trades)>=1:
            available_to_trade = False
        if not available_to_trade:
            return

        price      = self.data.Close[-1]
        volatility = self.data.volatility_tpsl[-1]

        short_state = []
        for feature in short_features:
            short_state.append(self.data[feature][-1])

        long_state = []
        for feature in long_features:
            long_state.append(self.data[feature][-1])

        # Long case
        if self.long_model is not None and self.data.position[-1]==1 and is_up_trend:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[0])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[1])
            probability = self.long_model.predict_proba([long_state])[0,1]
            size = 1.0
            if probability>=long_trigger_threshold and probability<long_green_threshold:
                size = 0.5
            if probability>=long_green_threshold:
                size=1.0
            if probability>=long_trigger_threshold:
                self.buy(size=size, sl=price_1, tp=price_0)

        # Short case
        if self.short_model is not None and self.data.position[-1]==-1 and is_down_trend:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[1])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[0])

            probability = self.short_model.predict_proba([short_state])[0,1]
            size = 1.0
            if probability>=short_trigger_threshold and probability<short_green_threshold:
                size = 0.5
            if probability>=short_green_threshold:
                size=1.0
            if probability>=short_trigger_threshold:
                self.sell(size=size, sl=price_0, tp=price_1)


bt = Backtest(
    test_df, 
    CCIMetaStrategy, 
    cash             = 100000000, 
    commission       = binance_commission, 
    exclusive_orders = True
    )

stats = bt.run(long_model=long_rf, short_model=short_rf)

print(stats)


#%%
stat_df = stats['_trades'][['ReturnPct', 'EntryTime']]
stat_df = stat_df.set_index('EntryTime')

pf.create_simple_tear_sheet(stat_df['ReturnPct'])


#%%
from pyfolio.plotting import plot_rolling_sharpe

plt.figure(figsize=(20, 14))
plot_rolling_sharpe(stat_df['ReturnPct'], rolling_window=21*3)


#%%


#%%


#%%


#%%
train_df


#%%
train_sides             = train_df[(train_df['position']==1)|(train_df['position']==-1)]['position']
train_signal_indexes    = train_df[(train_df['position']==1)|(train_df['position']==-1)].index
train_vertical_barriers = fml.labeling.add_vertical_barrier(t_events=train_signal_indexes, close=train_df['Close'], num_days=2)

pt_sl   = [0.7, 0.7]
min_ret = 0.001
train_triple_barriers = fml.labeling.get_events(
    close                  = train_df['Close'],
    t_events               = train_signal_indexes,
    pt_sl                  = pt_sl,
    target                 = train_df['volatility_tpsl'],
    min_ret                = min_ret,
    num_threads            = 1,
    vertical_barrier_times = train_vertical_barriers,
    side_prediction        = train_sides
    )

train_labels = fml.labeling.get_bins(train_triple_barriers, train_df['Close'])


#%%
train_triple_barriers


#%%
train_labels


#%%


#%%
# combinatorial purged cross validation and warm them up

X_train = df.loc[train_labels.index]
y_train = train_labels['bin']

n_split      = 10
n_test_split = 2

info_sets = train_triple_barriers.loc[X_train.index].t1

cv_gen = fml.cross_validation.CombinatorialPurgedKFold(
    n_splits          = n_split,
    n_test_splits     = n_test_split,
    samples_info_sets = info_sets,
    pct_embargo       = 0.01
)

# warmup for path information
for train_idxs, test_idxs in cv_gen.split(X=X_train, y=y_train):
    pass

print(f"total paths : {len(cv_gen.backtest_paths)}")


#%%


#%%
# Analyzing average testing periods by weeks

dt_list = []

for cv_path_set in cv_gen.backtest_paths:
    for cv_path in cv_path_set:
        test_dates = []
        for test_idx in cv_path['test']:
            test_dates.append(info_sets.iloc[test_idx])
        test_min_date = pd.to_datetime(min(test_dates))
        test_max_date = pd.to_datetime(max(test_dates))

        delta_time = pd.Timedelta(test_max_date - test_min_date)
        dt_list.append(int(delta_time.days/7))
        print(f"{delta_time} {int(delta_time.days/7)} weeks")

dt_list = np.array(dt_list)
dt_df   = pd.DataFrame()
dt_df['dt'] = dt_list


#%%
dt_df['dt'].plot()


#%%


#%%
X_train.loc[info_sets.index]

#%%
y_train[info_sets.index]


#%%

#%%


#%%
# Annualized Sharpe Ratio
def annualized_sharpe_ratio(returns, periods=252):
    estimated_sharpe_ratio = returns.mean()/returns.std(ddof=1)
    return estimated_sharpe_ratio*np.sqrt(periods)


#%%


#%%
print(len(long_features), " ===> ", long_features)

#%%
print(len(short_features), " ===> ", short_features)

#%%


#%%


#%%


#%%
for cv_path_set in cv_gen.backtest_paths:
    for cv_path in cv_path_set:
        temp_train_df = X_train.loc[info_sets.index[cv_path['train']]][long_features]
        print(temp_train_df.values.shape)

        temp_train_df = X_train.loc[info_sets.index[cv_path['train']]][short_features]
        print(temp_train_df.values.shape)

        #for col_name in temp_train_df.columns:
        #    print(temp_train_df[col_name])
        #print(temp_train_df.columns)
        #print(temp_train_df['t_adx_400'])
        #temp_df = pd.DataFrame()
        #print(temp_train_df)
        break
    break

#for cv_path_set in cv_gen.backtest_paths:
#    for cv_path in cv_path_set:
#        print(X_train.loc[info_sets.index[cv_path['train']]][short_features].columns)
#        break


#%%


#%%
import random
from mlfinlab.sample_weights import get_weights_by_return
from sklearn.ensemble        import RandomForestClassifier


#%%
print(long_features , len(long_features))
print(short_features, len(short_features))

#%%
print(train_df.columns)
print(train_df[long_features ].values.shape)
print(train_df[short_features].values.shape)

#%%


#%%
test_df['m_roc_300']


#%%


#%%
# Sharpe Ratio collections from testing paths

pt_sl = [0.7, 0.7]

binance_commission = 0.001

long_trigger_threshold  = 0.52
long_green_threshold    = 0.6

short_trigger_threshold = 0.52
short_green_threshold   = 0.6

class CCIMetaStrategy(Strategy):
    model_long  = None
    model_short = None
    def init(self):
        super().init()
    def next(self):
        super().next()

        if self.model_long is None:
            print("Need to supply long side Random Forest model...")
            return
        if self.model_short is None:
            print("Need to supply short side Random Forest model...")
            return

        available_to_trade = True
        if len(self.trades)>=1:
            available_to_trade = False
        if not available_to_trade:
            return

        price      = self.data.Close[-1]
        volatility = self.data.volatility_tpsl[-1]

        short_state = []
        for feature in short_features:
            short_state.append(self.data[feature][-1])

        long_state = []
        for feature in long_features:
            long_state.append(self.data[feature][-1])

        # Long case
        if self.model_long is not None and self.data.position[-1]==1:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[0])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[1])
            probability = self.model_long.predict_proba([long_state])[0,1]
            size = 1.0
            if probability>=long_trigger_threshold and probability<long_green_threshold:
                size = 0.5
            if probability>=long_green_threshold:
                size=1.0
            if probability>=long_trigger_threshold:
                self.buy(size=size, sl=price_1, tp=price_0)

        # Short case
        if self.model_short is not None and self.data.position[-1]==-1:
            price_0 = np.exp(np.log(price)+volatility*pt_sl[1])
            price_1 = np.exp(np.log(price)-volatility*pt_sl[0])

            probability = self.model_short.predict_proba([short_state])[0,1]
            size = 1.0
            if probability>=short_trigger_threshold and probability<short_green_threshold:
                size = 0.5
            if probability>=short_green_threshold:
                size=1.0
            if probability>=short_trigger_threshold:
                self.sell(size=size, sl=price_0, tp=price_1)


ret_sr_list = []

for cv_path_set in cv_gen.backtest_paths:
    for cv_path in cv_path_set:

        #sample_weights = get_weights_by_return(
        #    train_triple_barriers.loc[X_train.index], 
        #    train_df.loc[X_train.index, 'Close'],
        #    num_threads=1)

        # train long side model ...
        print("training long sided model...")
        X_train_long = X_train.loc[info_sets.index[cv_path['train']]]
        X_train_long = X_train_long[X_train_long['position']==1][long_features]
        y_train_long = y_train[X_train_long.index]

        rf_long = RandomForestClassifier(
            max_depth     = 80                  , 
            n_estimators  = 2500                ,
            criterion     = 'entropy'           , 
            class_weight  = 'balanced_subsample',
            random_state  = 42                  ,
            n_jobs        = -1
        )
        rf_long.fit(
            X_train_long[long_features], 
            y_train_long,
        )

        # train short side model ...
        print("training short sided model...")
        X_train_short = X_train.loc[info_sets.index[cv_path['train']]]
        X_train_short = X_train_short[X_train_short['position']==-1][short_features]
        y_train_short = y_train[X_train_short.index]

        rf_short = RandomForestClassifier(
            max_depth     = 80                  , 
            n_estimators  = 2500                ,
            criterion     = 'entropy'           , 
            class_weight  = 'balanced_subsample',
            random_state  = 42                  ,
            n_jobs        = -1
        )
        rf_short.fit(
            X_train_short[short_features], 
            y_train_short,
        )

        # Testing
        test_dates = []
        for test_idx in cv_path['test']:
            test_dates.append(info_sets.iloc[test_idx])
        test_min_date = min(test_dates)
        test_max_date = max(test_dates)
        print(f"Testing ...")

        bt = Backtest(
            train_df[test_min_date:test_max_date], 
            CCIMetaStrategy, 
            cash             = 100000000, 
            commission       = binance_commission, 
            exclusive_orders = True
        )
        stats = bt.run(model_long=rf_long, model_short=rf_short)

        stats_df = stats['_trades'][['ReturnPct', 'EntryTime']]
        stats_df = stats_df.set_index('EntryTime')
        estimated_sr = annualized_sharpe_ratio(stats_df["ReturnPct"])

        ret_sr_list.append((stats_df['ReturnPct'], estimated_sr))

        print(f"{test_min_date} : {test_max_date}, SR={estimated_sr}")



#%%


#%%
ret_sr_list

#%%


#%%
# Different paths and their Sharpe Ratio calculation took about 2.5 Hours
# so needs to save it as pickle for later analyze


#%%
import pickle

with open('ret_sr_list.pickle', 'wb') as f:
    pickle.dump(ret_sr_list, f, pickle.HIGHEST_PROTOCOL)


#%%
with open('ret_sr_list.pickle', 'rb') as f:
    ret_sr_loaded = pickle.load(f)

ret_sr_loaded


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

