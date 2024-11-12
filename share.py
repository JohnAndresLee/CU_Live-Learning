import os
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',500)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from pyomo.environ import *
from IPython.display import display
from joblib import Parallel, delayed

def load_df():
    files = os.listdir('cu2023/')
    files.sort()

    column_names = pd.read_csv('column_names.txt')

    dfs = []
    for file in tqdm(files):
        df = pd.read_csv('cu2023/'+file,encoding='gb2312',names=column_names.iloc[0,:].values)
        df['ExchangeID'] = 'SHFE'

        Timestamp = df['TradingDay'].astype(str) + ' ' + df['UpdateTime'] + '.' + df['UpdateMillisec'].astype(str).str.zfill(3)

        Timestamp = pd.to_datetime(Timestamp, format='%Y%m%d %H:%M:%S.%f')

        df.index = Timestamp

        dfs.append(df)

    df = pd.concat(dfs)
    df['TradingDay'] = pd.to_datetime(df['TradingDay'], format='%Y%m%d')

    resampled = df.resample('500ms').last()
    night = resampled[(resampled['UpdateTime']<='01:00:00')|(resampled['UpdateTime']>='21:00:00')]
    night['TradingDay'][night['UpdateTime']<='01:00:00'] = night['TradingDay'][night['UpdateTime']<='01:00:00'] - pd.Timedelta('1D')
    day = resampled[((resampled['UpdateTime']<='11:30:00')&(resampled['UpdateTime']>='09:00:00'))|((resampled['UpdateTime']<='15:00:00')&(resampled['UpdateTime']>='13:30:00'))]

    day[['Volume','Turnover']].fillna(0,inplace=True)
    day[[f'BidVolume{i}' for i in range(1,6)]].fillna(0,inplace=True)
    day[[f'AskVolume{i}' for i in range(1,6)]].fillna(0,inplace=True)

    night[['Volume','Turnover']].fillna(0,inplace=True)
    night[[f'BidVolume{i}' for i in range(1,6)]].fillna(0,inplace=True)
    night[[f'AskVolume{i}' for i in range(1,6)]].fillna(0,inplace=True)

    night['TradingDay'] = night['TradingDay'].dt.strftime('%Y%m%d')
    day['TradingDay'] = day['TradingDay'].dt.strftime('%Y%m%d')

    day.index.name = 'TimeStamp'
    night.index.name = 'TimeStamp'

    day = day[~day.index.duplicated(keep='last')]
    night = night[~night.index.duplicated(keep='last')]
    
    day = cal_diff(day)
    night = cal_diff(night)

    day = cal_assit_columns(day)
    night = cal_assit_columns(night)

    return day,night

def backtest(alpha,target):
    res = pd.DataFrame()
    res['pearson'] = alpha.corrwith(target)
    res['clipIC'] = alpha.corrwith(np.clip(target,-1e-3,1e-3))
    res['spearman'] = alpha.corrwith(target,method='spearman')
    res['long95'] = [(target.values * (alpha>alpha.quantile(0.95)).loc[:,col].values).mean() * 10000 for col in alpha.columns]
    res['short05'] = [(target.values * (alpha<alpha.quantile(0.05)).loc[:,col].values).mean() * 10000 for col in alpha.columns]
    res['long99'] = [(target.values * (alpha>alpha.quantile(0.99)).loc[:,col].values).mean() * 10000 for col in alpha.columns]
    res['short01'] = [(target.values * (alpha<alpha.quantile(0.01)).loc[:,col].values).mean() * 10000 for col in alpha.columns]
    return res.T

def cal_diff(data):
    data['deltaVolume'] = data['Volume'].diff().fillna(0)
    data['deltaVolume'] = np.where(data['TradingDay']==data['TradingDay'].shift(1), data['deltaVolume'], data['Volume'])

    data['deltaTurnover'] = data['Turnover'].diff().fillna(0)
    data['deltaTurnover'] = np.where(data['TradingDay']==data['TradingDay'].shift(1), data['deltaTurnover'], data['Turnover'])

    data['deltaTurnover'] = data['deltaTurnover']/5.0 #  五倍乘数

    return data

def cal_assit_columns(data):
    data['mp'] = (data['BidPrice1']+data['AskPrice1']) / 2
    data['mpRtn'] = (data['mp'] / data['mp'].shift(1)-1).fillna(0)

    data['vwap'] = data['deltaTurnover'] / data['deltaVolume']
    data['vwap'].fillna(data['mp'],inplace=True)
    data['vwapRtn'] = (data['vwap'] / data['vwap'].shift(1)-1).fillna(0)

    data['swmp'] = (data['AskPrice1'] * data['BidVolume1'] + data['BidPrice1'] * data['AskVolume1']) / (data['BidVolume1'] + data['AskVolume1'])
    data['imbalance'] = (data['AskVolume1'] - data['BidVolume1']) / (data['AskVolume1'] + data['BidVolume1']+1e-5)

    data['AskTotalVolume'] = data[['AskVolume1','AskVolume2','AskVolume3','AskVolume4','AskVolume5']].sum(axis=1)
    data['BidTotalVolume'] = data[['BidVolume1','BidVolume2','BidVolume3','BidVolume4','BidVolume5']].sum(axis=1)

    data['AskTotalTurnover'] = data['AskVolume1']*data['AskPrice1'] + data['AskVolume1']*data['AskPrice1'] + data['AskVolume2']*data['AskPrice2'] + data['AskVolume3']*data['AskPrice3'] + data['AskVolume4']*data['AskPrice4'] + data['AskVolume5']*data['AskPrice5']
    data['BidTotalTurnover'] = data['BidVolume1']*data['BidPrice1'] + data['BidVolume1']*data['BidPrice1'] + data['BidVolume2']*data['BidPrice2'] + data['BidVolume3']*data['BidPrice3'] + data['BidVolume4']*data['BidPrice4'] + data['BidVolume5']*data['BidPrice5']

    data['AskVWAP'] = data['AskTotalTurnover'] / (data['AskTotalVolume'])
    data['BidVWAP'] = data['BidTotalTurnover'] / (data['BidTotalVolume'])
    data['AskVWAP'].fillna(data['AskPrice1'],inplace=True)
    data['BidVWAP'].fillna(data['BidPrice1'],inplace=True)

    data['VWAPmp'] = (data['BidVWAP']+data['AskVWAP']) / 2
    data['Totalswmp'] = (data['AskTotalTurnover']+data['BidTotalTurnover']) / (data['BidTotalVolume'] + data['AskTotalVolume'] )

    data['imbalanceTotal'] = (data['AskTotalVolume'] - data['BidTotalVolume']) / (data['AskTotalVolume'] + data['BidTotalVolume']+1e-5)

    data['spread'] = data['AskPrice1'] - data['BidPrice1']

    data['vwapPosition'] = ((data['vwap'] - data['BidPrice1']) / (data['AskPrice1'] - data['BidPrice1'])).fillna(0)
    data['TotalswmpPosition'] = ((data['Totalswmp'] - data['BidPrice1']) / (data['AskPrice1'] - data['BidPrice1'])).fillna(0)
    data['LastPricePosition'] = ((data['LastPrice'] - data['BidPrice1']) / (data['AskPrice1'] - data['BidPrice1'])).fillna(0)

    return data

def minite_apply(df):
    minubar = pd.Series()

    if len(df)==0:
        return minubar

    mp_change = (df['mp'].diff()!=0).astype(int).cumsum()
    mp_change_max = mp_change.iloc[-1]
    mp_last_change = df[mp_change==mp_change_max]

    last_row = df.iloc[-1]

    minubar['TradingDay'] = df['TradingDay'].iloc[0]
    minubar['mp'] = last_row['mp']
    minubar['swmp'] = last_row['swmp']

    BidVolume,AskVolume,Spread,last_change_direct = last_row['BidVolume1'],last_row['AskVolume1'],last_row['AskPrice1']-last_row['BidPrice1'],last_row['last_mp_change_direction']

    BidExhaustedRatio, AskExhaustedRatio = monte_carlo(BidVolume,AskVolume,Spread,last_change_direct,100)

    bid_rho = (last_row['bid_insert_lambda']) / (last_row['bid_trade_lambda']+last_row['bid_cancel_lambda'])
    ask_rho = (last_row['ask_insert_lambda']) / (last_row['ask_trade_lambda']+last_row['ask_cancel_lambda'])

    bid_rho_hawkes = (last_row['bid_insert_lambda']) / (last_row['bid_trade_lambda_hawkes_to_last_mp_change']+last_row['bid_cancel_lambda'])
    ask_rho_hawkes = (last_row['ask_insert_lambda']) / (last_row['ask_trade_lambda_hawkes_to_last_mp_change']+last_row['ask_cancel_lambda'])

    minubar['imbalance'] = -df['imbalance'].iloc[-1]

    minubar['AskBidExhaustedRatio'] = - BidExhaustedRatio # +AskExhaustedRatio

    minubar['bid_rho_minus_ask_rho'] = ask_rho-bid_rho
    minubar['bid_rho_minus_ask_rho_hawkes'] = ask_rho_hawkes-bid_rho_hawkes

    minubar['last_mp_change'] = -df['last_mp_change'].iloc[-1]

    minubar['mp_last_change_NetOrderedOnBids'] = (mp_last_change['NetOrderedOnBids'].sum() - (mp_last_change['TradedOnBids'].sum()+mp_last_change['CanceledOnBids'].sum()+1)) -\
                                                 (mp_last_change['NetOrderedOnAsks'].sum() - (mp_last_change['TradedOnAsks'].sum()+mp_last_change['CanceledOnAsks'].sum()+1))

    minubar['HeavyOrder'] = ((df['TradedOnBids']-df['WeightedTradedOnBids']-1)/\
                             (df['TradedOnAsks']-df['WeightedTradedOnAsks']-1)).ewm(alpha=0.9).sum().iloc[-1]

    minubar['WeightedTradedtoTraded'] = ((df['WeightedTradedOnAsks']-df['WeightedTradedOnBids']) / \
                                        (df['TradedOnBids']+df['TradedOnAsks'])).ewm(alpha=0.1).sum().iloc[-1]

    minubar['TradeTimesImbalance'] = (df['TradedOnBids'] * (1-df['imbalance'])**3).ewm(alpha=0.2).sum().iloc[-1] - \
                                     (df['TradedOnAsks'] * (1+df['imbalance'])**3).ewm(alpha=0.2).sum().iloc[-1]

    minubar['ewm0.1TradedOnAskMinusBid'] = (df['TradedOnAsks']-df['TradedOnBids']).ewm(alpha=0.1).sum().iloc[-1]

    minubar['CanceledOnBids2CanceledOnAsks'] = (df['CanceledOnAsks']-df['CanceledOnBids']).ewm(alpha=0.95).sum().iloc[-1]

    return minubar

def daily_apply(df):
    minubar = df.groupby(df['minute'], group_keys=False).apply(minite_apply)
    minubar['mp_target'] = (minubar['mp'].shift(-1) / minubar['mp'] -1).fillna(0)
    minubar['swmp_target'] = (minubar['swmp'].shift(-1) / minubar['swmp'] -1).fillna(0)
    return minubar.reset_index()

def main():
    assigned = pd.read_parquet('../assigned.parquet')
    df = cal_assit_columns(assigned)
    df = df[df['UpdateTime']>='09:00:01']
    df['minute'] = df['UpdateTime'].apply(lambda x: x[:5])
    minubar = pd.concat(Parallel(n_jobs=100)(delayed(daily_apply)(df) for df in tqdm(dfs[118:])))
    minubar.index = pd.to_datetime(minubar['TradingDay'] + ' ' + minubar['minute'])
    mp_target = minubar.pop('mp_target')
    swmp_target = minubar.pop('swmp_target')
    minubar.drop(['minute','TradingDay','mp','swmp'],axis=1,inplace=True)

    backtest(minubar,mp_target)