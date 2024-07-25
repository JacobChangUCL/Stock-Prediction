"""
This module provides functionalities to download stock data 
using yfinance and generic data using HTTP requests..
"""
import yfinance as yf
import requests


def download_data(ticker: str, start: str, end: str) -> True:
    """
    ticker: the ticker of the stock, e.g. AAPL (Apple),the ticker is the unique name of the stock.
    start: the start date of the data, e.g. '2020-01-01'
    end: the end date of the data, e.g. '2023-01-01'
    """
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        print(f"yfinance doesn't work now.")
        return False
    # maybe violate the rule of naming file
    data.to_csv("./data/" + ticker + ".csv")
    return True


def download_using_http(url: str, name: str, download_folder="./data/") -> True:
    """
    Downloads a file from the given URL and saves it with the specified name.

    Args:
        url (str): The URL to download the file from.
        name (str): The name to save the downloaded file as.
    """
    response = requests.get(url)
    with open(download_folder + f'{name}.csv', 'wb') as file:
        file.write(response.content)
    return True


def download_all():
    """
    download all the data we need
    """
    download_data('AAPL', '2017-04-01', '2022-04-02')
    URL_DFF = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1138&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DFF&scale=left&cosd=2017-04-01&coed=2022-04-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily%2C%207-Day&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-08-07&revision_date=2023-08-07&nd=1954-07-01"
    URL_T10Y2Y = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1138&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10Y2Y&scale=left&cosd=2017-04-01&coed=2022-04-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2023-09-09&revision_date=2023-09-09&nd=1976-06-01"
    download_using_http(URL_DFF, "DFF")
    download_using_http(URL_T10Y2Y, "T10Y2Y")
    print("downloading is done")


if __name__ == "__main__":
    download_all()
