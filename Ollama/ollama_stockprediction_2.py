import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from langchain_ollama import OllamaLLM
from io import StringIO


llm = OllamaLLM(model="llama3.1")

def pull_stocks(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    stock_data = yf.Ticker(ticker)
    stock_df = stock_data.history(start=start_date, end=end_date)
    stock_df.index = stock_df.index.tz_localize(None)  # Ensure stock data is timezone-naive
    stock_df = stock_df.reset_index()
    stock_df['Date'] = stock_df['Date'].dt.strftime('%Y-%m-%d')
    stock_df['pct_change'] = stock_df['Close'].pct_change()
    stock_df = stock_df[stock_df['pct_change'].notna()==True]
    stock_df = stock_df[['Date','pct_change']]
    actual_final = stock_df.tail(1)
    # stock_df = stock_df.iloc[:-1]
    return stock_df, actual_final

def arima(timeseries_df):
    # Ensure 'Date' is the index and in datetime format
    timeseries_df.set_index('Date', inplace=True)
    timeseries_df.index = pd.to_datetime(timeseries_df.index)

    # Remove the last row (assumed to be NaN)
    timeseries_df = timeseries_df[:-1]

    # Convert percentage strings to float if necessary
    if timeseries_df['pct_change'].dtype == 'object':
        timeseries_df['pct_change'] = timeseries_df['pct_change'].str.rstrip('%').astype('float') / 100.0

    # Fit ARIMA model
    model = ARIMA(timeseries_df['pct_change'].dropna(), order=(1, 1, 1))
    results = model.fit()

    # Predict the next day's percentage change
    forecast = results.forecast(steps=1)
    predicted_pct_change = forecast.values[0]

    print(f"Predicted percentage change for next day: {predicted_pct_change:.6f}")

def convert_to_csv_string(timeseries):

    timeseries = timeseries.reset_index()

    timeseries['pct_change'] = np.round(timeseries['pct_change'], 6)

    # Remove final row
    timeseries = timeseries.iloc[:-1]

    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    timeseries.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()

    return csv_string

## Note: Change the date of the prediction/forecast for your own use (here, 2024-12-02)

def predict_timeseries(timeseries):
    output = llm.invoke(f"""
        You are a large language model with time series forecasting capabilities.
        Predict the percent change for the day immediately after the end of the provided time series (2024-06-28).
        Use only your model capabilities, not any other method.
        The data is in the format of a csv file.
        The dataset includes:
        - Date
        - Percent change in the cryptocurrency from the previous day
        Provide only the forecasted percent change for 2024-12-02 as a point estimate. 
        Do not include any other text or context, just the one value:
        {timeseries}
    """)
    return output.strip()


btc, btc_final = pull_stocks('BTC-USD')
eth, eth_final = pull_stocks('ETH-USD')
xrp, xrp_final = pull_stocks('XRP-USD')

arima(btc)
arima(eth)
arima(xrp)

btc_for_llm = convert_to_csv_string(btc)
eth_for_llm = convert_to_csv_string(eth)
xrp_for_llm = convert_to_csv_string(xrp)
print(btc_for_llm)

print(predict_timeseries(btc_for_llm))
print(predict_timeseries(eth_for_llm))
print(predict_timeseries(xrp_for_llm))

print(btc_final)
print(eth_final)
print(xrp_final)