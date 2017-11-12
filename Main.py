import os
import numpy as np
import pandas as pd
import pickle
import quandl
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import time
from datetime import datetime
from datetime import date

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

#Helper Functions
def get_quandl_data(quandl_id):
    '''Download and cache Quandl dataseries'''
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df


def merge_dfs_on_column(dataframes, labels, col):
    '''Merge a single column of each dataframe into a new combined dataframe'''
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)

def get_json_data(json_url, cache_path):
    '''Download and cache JSON data, return as a dataframe.'''
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df

def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    '''Generate a scatter plot of the entire dataframe'''
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))

    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels=not seperate_y_axis,
            type=scale
        )
    )

    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale)

    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'

    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index,
            y=series,
            name=label_arr[index],
            visible=visibility
        )

        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    py.plot(fig,filename='graph.html')


#SETUP FOR get_crypto_data function()
base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2017-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
pediod = 1800 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    '''Retrieve cryptocurrency data from poloniex'''
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df


def correlation_heatmap(df, title, absolute_bounds=True):
    '''Plot a correlation heatmap for the entire dataframe'''
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient'),
        colorscale='Jet',
    )

    print(df.columns.map(str))
    layout = go.Layout(title=title)

    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0

    fig = go.Figure(data=[heatmap], layout=layout, )
    py.plot(fig, filename='heatmap.html')


#-----------------MAIN-----------------------------------
# # Pull Kraken BTC price exchange data
# btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')
#
# # Pull pricing data for 3 more BTC exchanges
# exchanges = ['COINBASE','BITSTAMP','ITBIT']
#
# exchange_data = {}
#
# exchange_data['KRAKEN'] = btc_usd_price_kraken
#
# for exchange in exchanges:
#     exchange_code = 'BCHARTS/{}USD'.format(exchange)
#     btc_exchange_df = get_quandl_data(exchange_code)
#     exchange_data[exchange] = btc_exchange_df
#
# # Merge the BTC price dataseries' into a single dataframe
# btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
#
# # # Chart the BTC pricing data
# # btc_trace = go.Scatter(x=btc_usd_price_kraken.index, y=btc_usd_price_kraken['Weighted Price'])
# # py.plot([btc_trace], filename='graph1.html')
# #
# # print(btc_usd_datasets.tail())
#
# # Remove "0" values
# btc_usd_datasets.replace(0, np.nan, inplace=True)

# Calculate the average BTC price as a new column



#btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)


#---------------------------- MAIN ---------------------------------

#Highly inefficient lines of code: (Though does the job atleast)
#Get all the cryptocurrency data available on poloniex
all_tickers_df =  get_json_data('https://poloniex.com/public?command=returnTicker', 'TICKERS')
#delete the unwanted data
all_tickers_df = all_tickers_df.drop(all_tickers_df.index[[0,2,3,4,5,6,7,8,9]])
#Delete the unwanted columns. ie, columns without BTC as reference
all_tickers_df = all_tickers_df.loc[:, all_tickers_df.columns.str.startswith('BTC')]
#Sort dataframe by highest 24 hour price
all_tickers_df = all_tickers_df.iloc[:,np.argsort(-1*all_tickers_df.loc['high24hr'])]
#get input from user to determine how many cryptocurencies to use for analysis
cryptoCurrencyNumber = input('Enter the top n currencies to be used in analysis: ')
#Delete the unwanted columns again. Only keep the top ones entered by the user
all_tickers_df = all_tickers_df.iloc[:,0:int(cryptoCurrencyNumber)]

#Generate a list of altcoin pairs
altcoinsPair = list(all_tickers_df.columns.values)
#print the pairs being used
print(list(all_tickers_df.columns.values))

#Get the price information on the trading pairs
altcoin_data = {}
for altcoin in altcoinsPair:
    #coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(altcoin)
    altcoin_data[altcoin] = crypto_price_df





# # Calculate USD Price as a new column in each altcoin dataframe
# for altcoin in altcoin_data.keys():
#     altcoin_data[altcoin]['price_usd'] = altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']


# Merge USD price of each altcoin into single dataframe
#Merge all altcoins in one dataset
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'weightedAverage')
#generate a correleation matrix
print(combined_df.pct_change().corr(method='pearson'))
#Generate a heatmap for the correlation matrix for visualization
#correlation_heatmap(combined_df.pct_change(), "Cryptocurrency Correlations in 2017")
#print(combined_df.pct_change())

# # Add BTC price to the dataframe
# combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']

# Calculate the pearson correlation coefficients for cryptocurrencies in 2016
# combined_df_2016 = combined_df[combined_df.index.year == 2017]
#print(combined_df_2016.pct_change().corr(method='pearson'))

#correlation_heatmap(combined_df_2016.pct_change(), "Cryptocurrency Correlations in 2017")


# Chart all of the altocoin prices
#Plot the cryptocurrency values
#df_scatter(combined_df, 'Cryptocurrency Prices (BTC)', seperate_y_axis=False, y_axis_label='Coin Value (BTC)', scale='linear')



# # Plot the average BTC price
# btc_trace = go.Scatter(x=combined_df.index, y=combined_df)
# py.plot([btc_trace], filename='graph4.html')

# Plot all of the BTC exchange prices
#df_scatter(btc_usd_datasets, 'Bitcoin Price (USD) By Exchange')