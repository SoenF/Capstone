from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def prep_data(data, covariates, data_start, train = True):
    time_len = data.shape[0]
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 1
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)
    print(x_input)

def replacement(number):
    calendar_df['event_type_'+str(number)] = calendar_df['event_type_'+str(number)].fillna(0)
    for i,elem in enumerate(calendar_df['event_type_'+str(number)]):
        if calendar_df['event_type_'+str(number)][i]!=0:
            calendar_df['event_type_'+str(number)][i]= dict_type[elem]

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = calendar_df['wday'][input_time]
        covariates[i, 2] = calendar_df['month'][input_time]
        covariates[i, 3] = calendar_df['event_type_1'][input_time]
        covariates[i, 4] = calendar_df['event_type_2'][input_time]
        covariates[i, 5] = calendar_df['snap_CA'][input_time]
        covariates[i, 6] = calendar_df['snap_TX'][input_time]
        covariates[i, 7] = calendar_df['snap_WI'][input_time]
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def visualize(data, week_start):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()


if __name__ == '__main__':

    global save_path
    save_name = 'elect'
    window_size = 84
    stride_size = 28
    num_covariates = 8
    train_start = 'd_1'
    train_end = 'd_1913'
    test_start = 'd_1858'  # need additional 56 days as given info
    test_end = 'd_1941'
    pred_days = 28
    given_days = 56
    sales_name = 'sales_train_evaluation.csv'
    calendar_name = 'calendar.csv'
    price_name = 'sell_prices.csv'
    dict_type={'Sporting':1, 'Cultural':2, 'National':3, 'Religious':4}

    save_path = os.path.join('data', save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sales_path = os.path.join(save_path, sales_name)
    calendar_path = os.path.join(save_path, calendar_name)
    price_path = os.path.join(save_path, price_name)

    
    sales_df = pd.read_csv(sales_path).T
    sales_df_val = sales_df['d_1':]
 
    
    calendar_df = pd.read_csv(calendar_path)
    calendar_df.set_index(['d'],inplace=True)
    price_df = pd.read_csv(price_path)
    replacement(1)
    replacement(2)
    covariates = gen_covariates(sales_df_val[train_start:test_end].index, num_covariates)
    train_data = sales_df[train_start:train_end].values
    test_data = sales_df[test_start:test_end].values
    data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
    total_time = sales_df_val.shape[0] #1941
    num_series = sales_df_val.shape[1] #30490
    prep_data(train_data, covariates, data_start)
    prep_data(test_data, covariates, data_start, train=False)

