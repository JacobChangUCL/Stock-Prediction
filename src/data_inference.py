#!/usr/bin/env python
# coding: utf-8
#This is the code for building a time series prediction model using the Prophet library.

import pandas as pd
#The original name was fbprophet
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings("ignore")

#train_test_split
def train_test_split(df_inner,test_size=0.2):
    """
    Split the data into training and testing sets. default test_size=0.2
    """
    train_size = int(len(df_inner)*(1-test_size))
    train = df_inner.iloc[:train_size]
    test = df_inner.iloc[train_size:]
    return train,test

def calculate_RMSE(df:pd.DataFrame)->float:
    """
    Calculate the root mean square error of the model
    """
    return ((df['y']-df['yhat'])**2).mean()**0.5

# use the 'make_predictions_df' function to make predictions on our forecasted data
def include_all_data(forecast, train,test):
    """
    Include all the data in the prediction dataframe,which includes the true price,
    the predicted price, and the confidence interval.
    "y" is the true price, "yhat" is the predicted price, "yhat_lower" is the lower bound of the confidence interval, and "yhat_upper" is the upper bound of the confidence interval.
    """
    all=pd.concat([train,test])
    # create a dataframe to store the predictions
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    # merge the predictions with all the data
    prediction['y']=all['y']
    return prediction

def visualization(df:pd.DataFrame,test,path=None,figsize=(10,6)):
    plt.figure(figsize=figsize)
    plt.plot(df['ds'],df['y'],label='True Price',color='r')
    plt.plot(df['ds'],df['yhat'],label='Predicted Price',color='b')
    plt.fill_between(df['ds'],df['yhat_lower'],df['yhat_upper'],color='k',alpha=0.2)
    plt.title(f' Price Prediction')
    # add a vertical dashed line to mark the separation point between training and testing data
    plt.axvline(test.iloc[0,0], color='k', ls='--', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    if path is not None:
        plt.savefig(path)

def create_joint_plot(forecast, x='yhat', y='y', path=None, title=None):
    # create a joint plot with 'yhat' as the x-axis and 'y' as the y-axis
    g = sns.jointplot(x='yhat', y='y', data=forecast, kind="reg", color="b")

    # To plot the image with a fair scale, the range of the x-axis appearing on the image and the range of the y-axis should be the same. To ensure no points are missed, I need to find the minimum and maximum values and set them as the range for the image.
    total_max = max(forecast["y"].max(), forecast["yhat"].max())
    total_min = min(forecast["y"].min(), forecast["yhat"].min())
    g.ax_joint.set_xlim(total_min, total_max)
    g.ax_joint.set_ylim(total_min, total_max)

    # set the width and height of the figure
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    # access the second subplot in the figure (scatter plot) and set its title if provided
    ax = g.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    # access the first subplot in the figure (histograms) and display the correlation coefficient
    ax = g.fig.axes[0]
    fig_center = (total_max + total_min) / 2  # the center location of the figure
    ax.text(fig_center, fig_center, "R = {:+4.2f}".format(forecast.loc[:, ['y', 'yhat']].corr().iloc[0, 1]),
            fontsize=16)

    # set labels, limits, and grid lines for the x and y axes
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Observations', fontsize=15)

    ax.grid(ls=':')
    if path is not None:
        g.savefig(path)
    # set the font size for the x-axis and y-axis tick labels
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    # add gridlines
    ax.grid(ls=':')

def data_inference():#main function

    #load data
    df = pd.read_csv('./data/ALL_cleaned.csv', parse_dates=['DATE'])

    #we use logPrice rather than Close because it is more linear, and we can convert it back to price later
    #create a prophet model
    m = Prophet(seasonality_mode='multiplicative',
                growth='linear',
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True)
    #because the minimum interval of time is 1 day, so we don't need to set the daily_seasonality

    #fit the format of prophet model; the column names should be 'ds' and 'y'
    df_with_logPrice_DATA = df[["DATE","logPrice"]]
    df_with_logPrice_DATA.columns= ["ds","y"]
    train,test=train_test_split(df_with_logPrice_DATA)

    m.fit(train)

    df_cv = cross_validation(m, initial='1095 days', period='15 days', horizon = '30 days')
    #1095 days~3 years

    df_p = performance_metrics(df_cv,rolling_window=1)
    print("single_datasource_model performance\n",df_p,'\n')

    #an dataframe with only one column ds, which is the date we want to predict.
    #include_history=False means we don't need to predict the history data.the default value is True
    prediction = m.make_future_dataframe(periods=366,freq='1D',include_history=True)

    forecast = m.predict(prediction)
    # display the forecasted results

    fig=m.plot_components(forecast)
    fig.savefig("./pictures/single_source_data_prediction_decomposition")

    result = include_all_data(forecast, train, test)
    print("single_datasource_prediction_from_2017\n",result.head(10),'\n')

    visualization(result, test, "./pictures/single_source_data_prediction.png")

    result_predict= result[result['ds'] >=test.iloc[0,0]]
    print("predict_result_in_testing_period\n",result_predict.head(10),'\n')
    #RMSE for one month
    single_source_RMSE=calculate_RMSE(result_predict.iloc[0:30,:])
    print(f"single_datasource_model RMSE for one month : {single_source_RMSE}\n")

    #the prediction is higher than the true price, which means the model is not good enough.

    create_joint_plot(result_predict, title='Test Set Predictions', path="./pictures/single_source_data_Test_set_predictions.png")
    #这个要保证x与y是一个方块

    create_joint_plot(result[result['ds'] <test.iloc[0,0]], title='Train Set Predictions', path="./pictures/single_source_data_Train_set_predictions.png")
    #不对，这里把test set的结果也加上了。
    #very very good

    #use multiple sauce data to predict
    #create a new model
    m2=Prophet( seasonality_mode='multiplicative',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False)

    #add 2 data sources,mode=multiplicative
    m2.add_regressor('T10Y2Y', mode='multiplicative')
    m2.add_regressor('DFF', mode='multiplicative')

    df2=df.drop("Close",axis=1)
    df2.rename(columns={"DATE":"ds","logPrice":"y"},inplace=True)
    print("multiple_datasource_training_data\n",df2.head(),'\n')

    train2,test2=train_test_split(df2)

    m2.fit(train2)

    #cross validation
    df_cv2 = cross_validation(m2, initial='1095 days', period='15 days', horizon = '30 days')
    #1095 days~3 years

    df_p2 = performance_metrics(df_cv,rolling_window=1)
    print("mutiple_datasource_model_performance\n",df_p2,'\n')

    future2=m2.make_future_dataframe(periods=len(test2), freq='1D')

    future2=pd.concat([future2,df2[["DFF","T10Y2Y"]]],axis=1)

    result2=m2.predict(future2)

    fig2=m.plot_components(result2)
    fig2.savefig("./pictures/multiple_source_data_prediction_decomposition.png")

    all_data2=include_all_data(result2,train2,test2)

    print("multiple_datasource_prediction\n",all_data2.tail(),'\n')

    visualization(all_data2, test, path="./pictures/multiple_source_data_prediction.png")

    create_joint_plot(all_data2[all_data2['ds'] <test2.iloc[0,0]], title='Train Set prediction', path="./pictures/multiple_source_data_Train_set_predictions.png")

    #prediction part
    prediction2=all_data2[all_data2['ds'] >=test2.iloc[0,0]]
    create_joint_plot(prediction2, title='Test Set Predictions', path="./pictures/multiple_source_data_Test_set_predictions.png")
    mutiple_source_RMSE=calculate_RMSE(prediction2.iloc[0:30,:])
    print(f"multiple_datasource_model RMSE for one month: {mutiple_source_RMSE}")

if __name__ == "__main__":
    data_inference()
