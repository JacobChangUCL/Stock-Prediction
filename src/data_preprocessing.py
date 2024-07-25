import warnings
warnings.filterwarnings("ignore")
"""
Data Preprocessing for Financial Data.

This module provides functionalities to preprocess financial data, specifically:
    - AAPL stock data
    - DFF (Federal Funds Rate) data
    - T10Y2Y (10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity) data
It includes functions for:
    - Anomaly detection and replacement using z-scores
    - Interpolation for missing dates
    - Data transformation and aggregation

Usage:
    To preprocess the AAPL stock data, simply run this module:
    $ python <filename>.py

Note:
    Make sure to have the raw data files in the './data/' directory before running the preprocessing.

Author:
    Jingbo Zhang
    Date: 2023-9-09
"""
import random
import pandas as pd

def replace_anomalies_with_normal(dataframe:pd.DataFrame, column_name:str, window_size=6, threshold=2):
    """
    Replaces anomalies in a dataframe with normally distributed values.
    The anomalies are detected using z-scores>threshold.
    The anomalies are replaced with normally distributed values 
    with the same mean and standard deviation as the values in the window 
    around the anomaly.
    """
    rolling_mean = dataframe[column_name].rolling(window=window_size).mean()
    rolling_std = dataframe[column_name].rolling(window=window_size).std()
    dataframe['z_score'] = (dataframe[column_name] - rolling_mean) / rolling_std
    dataframe['anomaly'] = abs(dataframe['z_score']) > threshold
    anomaly_indices = dataframe[dataframe['anomaly']].index
    for index in anomaly_indices:
        mean = rolling_mean.loc[index]
        std = rolling_std.loc[index]
        dataframe.at[index, column_name] = random.normalvariate(mean, std)
    return dataframe.drop(columns=['z_score', 'anomaly'])

def data_preprocessing():
    """
    preprocess all of the data 
    """
    data_preprocessing_AAPL()
    data_preprocessing_dff()
    data_preprocessing_T10Y2Y()
    print("data preprocessing is done")
    # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity


def data_preprocessing_AAPL():
    """
    preprocess the AAPL data
    """
    AAPL=pd.read_csv("./data/AAPL.csv", parse_dates=['Date'])
    AAPL.set_index('Date', inplace=True)
    AAPL=AAPL.resample('D').interpolate(method='linear')
    
    for column in AAPL.columns:
        if column == 'Date':
            continue
        AAPL = replace_anomalies_with_normal(AAPL, column)
    AAPL.to_csv("./data/AAPL_preprocessed.csv")

def data_preprocessing_dff():
    DFF=pd.read_csv("./data/DFF.csv", parse_dates=["DATE"])
    #select the data between "2017-04-01" and "2022-04-01"
    DFF=DFF[("2017-04-01"<=DFF["DATE"])&(DFF["DATE"]<="2022-04-01")]
    DFF.to_csv("./data/DFFnew.csv",index=False)

def data_preprocessing_T10Y2Y():
    T10Y2Y=pd.read_csv("./data/T10Y2Y.csv", parse_dates=["DATE"])
    #select the data between "2017-04-01" and "2022-04-01"
    T10Y2Y=T10Y2Y[("2017-04-01"<=T10Y2Y["DATE"])&(T10Y2Y["DATE"]<="2022-04-01")]
    T10Y2Y["T10Y2Y"]=T10Y2Y["T10Y2Y"].replace('.',pd.NA).fillna(method='ffill',inplace=False)
    T10Y2Y.to_csv("./data/T10Y2Ynew.csv",index=False)


def data_integration():
    """
    integrate the data"""
    #read the data
    AAPL=pd.read_csv("./data/AAPL_preprocessed.csv", parse_dates=['Date'])
    DFF=pd.read_csv("./data/DFFnew.csv", parse_dates=["DATE"])
    T10Y2Y=pd.read_csv("./data/T10Y2Ynew.csv", parse_dates=["DATE"])

    #merge the data
    all=pd.merge(AAPL,DFF,left_on="Date",right_on="DATE",how="right")
    all=pd.merge(all,T10Y2Y,left_on="DATE",right_on="DATE",how="left")
    all=all.drop(columns=["Date"])
    #fill the NA with the 
    all=all.fillna(method='bfill')
    all.set_index('DATE', inplace=True)
    #save the data
    all.to_csv("./data/ALL.csv",index=True)
    print("data integration is done")


if __name__ == '__main__':
    data_preprocessing()
    data_integration()

