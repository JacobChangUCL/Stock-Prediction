#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

def visualization():
    ALL=pd.read_csv('./data/ALL.csv', parse_dates=["DATE"])
    ALL.set_index('DATE', inplace=True)

    print("All.head():\n",ALL.head(), "\n")

    #candle chart
    one_month=ALL.loc['2020-01':'2020-02']
    mpf.plot(one_month, type='candle', style='charles',
             title='          AAPL Candlestick Chart 2020-01-01~2020-02-29',
             ylabel='Price ($)', savefig='./pictures/Candlestick_Chart.png')

    #line chart
    plt.title("DFF - Federal Reserve Interest Rate")
    plt.xlabel("DATE")  # 根据实际情况更改
    plt.ylabel("DFF Value(%)")
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(100))
    plt.plot(ALL.index,ALL["DFF"])
    plt.savefig("./pictures/DFF.jpg")
    plt.clf()
    #histogram

    plt.hist(ALL["DFF"])
    plt.ylabel("Days")
    plt.title("DFF-histogram")
    plt.xlabel("Interest Rate(%)")
    plt.savefig("./pictures/DFF_hist.jpg")
    plt.clf()

    #line chart
    plt.title("AAPL Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.plot(ALL.index,ALL['Close'])
    plt.tick_params(axis='x', labelsize=8)
    plt.savefig("./pictures/AAPLDataLineChart.jpg")
    plt.clf()

    #histogram
    plt.hist(ALL['Close'], bins=25, edgecolor='black')
    plt.ylabel("frequency")
    plt.title("AAPL Closing Price Histogram")
    plt.xlabel("Closing Price")
    plt.savefig("./pictures/AAPLDataHist.jpg")
    plt.clf()

    plt.clf()
    plt.title("T10Y2Y:10-Year Treasury Constant Maturity \nMinus 2-Year Treasury Constant Maturity")
    plt.xlabel("Date")
    plt.ylabel("T10Y2Y(%)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.plot(ALL.index,ALL["T10Y2Y"])
    plt.savefig("./pictures/T10Y2Y_lineChart.jpg")
    plt.clf()

    plt.title("T10Y2Y-histogram")
    plt.xlabel("T10Y2Y(%)")
    plt.ylabel("frequency")
    plt.hist(ALL["T10Y2Y"], bins=25, edgecolor='black')
    plt.savefig("./pictures/T10Y2Y_hist.jpg")
    plt.clf()

    print("Visualization is done")


