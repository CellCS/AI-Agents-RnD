##############
#### Deep Charts Youtube Channel: https://www.youtube.com/@DeepCharts
#### Subscribe for more AI/Machine Learning/Quant Finance Tutorials
##############


##############
### PART 1 ###
# LIBRARIES ##
# & OLLAMA  ##
##############

## streamlit run ollama_stockprediction.py

# Data Importing Libraries
import yfinance as yf
from finvizfinance.quote import finvizfinance

# Data Modeling Library
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Charts
import plotly.graph_objects as go

# Data Manipulation
import pandas as pd
import numpy as np

# Avoid Forecasting on Holidays
import holidays

# Create Local LLM Server Connection
from langchain_ollama import OllamaLLM

# Interactive Web App UI
import streamlit as st


# Connect to local Ollama server
llm = OllamaLLM(model="llama3.1")


##############
### PART 2 ###
# FUNCTIONS ##
##############


# Function to classify sentiment
def classify_sentiment(title):
    output = llm.invoke(
        f"Classify the sentiment as 'POSITIVE' or 'NEGATIVE' or 'NEUTRAL' with just that one word only, no additional words or reasoning: {title}"
    )
    return output.strip()  # Ensure the response is clean and without extra spaces


# Function to get and process news data
def get_news_data(ticker):

    # Data Pull
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()

    # Preprocess before putting into LLM
    news_df["Title"] = news_df["Title"].str.lower()

    # Classify Sentiment function applied to each row of news_df
    news_df["sentiment"] = news_df["Title"].apply(classify_sentiment)

    # Postprocess after putting into LLM
    news_df["sentiment"] = news_df["sentiment"].str.upper()
    news_df = news_df[news_df["sentiment"] != "NEUTRAL"]
    news_df["Date"] = pd.to_datetime(news_df["Date"])
    news_df["DateOnly"] = news_df["Date"].dt.date

    return news_df


# Function to group and process sentiment data
def process_sentiment_data(news_df):

    # Reshape data to have df with columns: Date, # of positive Articles, # of negative Articles
    grouped = news_df.groupby(["DateOnly", "sentiment"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=["POSITIVE", "NEGATIVE"], fill_value=0)

    # Create rolling averages that count number of positive and negative sentiment articles within past t days
    grouped["7day_avg_positive"] = (
        grouped["POSITIVE"].rolling(window=7, min_periods=1).sum()
    )
    grouped["7day_avg_negative"] = (
        grouped["NEGATIVE"].rolling(window=7, min_periods=1).sum()
    )

    # Create "Percent Positive" by creating percentage measure
    grouped["7day_pct_positive"] = grouped["POSITIVE"] / (
        grouped["POSITIVE"] + grouped["NEGATIVE"]
    )
    result_df = grouped.reset_index()

    return result_df


# Function to fetch and process stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)  # Pull ticker data
    stock_data["Pct_Change"] = (
        stock_data["Close"].pct_change() * 100
    )  # Transform closing value to percent change in closing value since previous day
    return stock_data


# Function to combine sentiment and stock data
def combine_data(result_df, stock_data):
    combined_df = result_df.set_index("DateOnly").join(
        stock_data[["Pct_Change"]], how="inner"
    )
    combined_df["lagged_7day_pct_positive"] = combined_df["7day_pct_positive"].shift(
        1
    )  # Lag sentiment feature by 1 day for temporal alignment
    return combined_df


# Function to calculate Pearson correlation
def calculate_correlation(combined_df):
    correlation_pct_change = (
        combined_df[["lagged_7day_pct_positive", "Pct_Change"]].corr().iloc[0, 1]
    )
    return correlation_pct_change


# Function to get future dates excluding weekends and holidays
def get_future_dates(start_date, num_days):
    us_holidays = holidays.US()
    future_dates = []
    current_date = start_date
    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)
    return future_dates


# Function to fit ARIMAX model and forecast
def fit_and_forecast(combined_df, forecast_steps=3):
    endog = combined_df["Pct_Change"].dropna()  # Dependent variable
    exog = combined_df["lagged_7day_pct_positive"].dropna()  # Predictor variable
    endog = endog.loc[exog.index]  # Align variables
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))  # ARIMAX model
    fit = model.fit(disp=False)  # Fit model

    future_dates = get_future_dates(
        combined_df.index[-1], forecast_steps
    )  # Future dates
    future_exog = combined_df["lagged_7day_pct_positive"][
        -forecast_steps:
    ].values.reshape(
        -1, 1
    )  # Future exogenous values

    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)  # Get forecast
    forecast_mean = forecast.predicted_mean  # Predicted mean
    forecast_ci = forecast.conf_int()  # Confidence intervals

    return forecast_mean, forecast_ci, future_dates  # Return results


# Function to create and display plot
def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):
    # Standardize the sentiment proportion
    sentiment_std = (
        combined_df["7day_pct_positive"] - combined_df["7day_pct_positive"].mean()
    ) / combined_df["7day_pct_positive"].std()

    fig = go.Figure()

    # Add standardized sentiment proportion
    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=sentiment_std,
            name="Standardized Sentiment Proportion",
            line=dict(color="blue"),
            mode="lines",
        )
    )

    # Add stock percentage change
    fig.add_trace(
        go.Scatter(
            x=combined_df.index,
            y=combined_df["Pct_Change"],
            name="Stock Pct Change",
            line=dict(color="green"),
            yaxis="y2",
            mode="lines",
        )
    )

    # Add forecasted stock percentage change
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_mean,
            name="Forecasted Pct Change",
            line=dict(color="red"),
            mode="lines",
        )
    )

    # Add confidence intervals for the forecast
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([forecast_index, forecast_index[::-1]]),
            y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Update layout with appropriate y-axis ranges
    fig.update_layout(
        title="Sentiment Proportion and Stock Percentage Change with Forecast",
        xaxis_title="Date",
        yaxis=dict(
            title="Standardized Sentiment Proportion", titlefont=dict(color="blue")
        ),
        yaxis2=dict(
            title="Stock Pct Change",
            titlefont=dict(color="green"),
            overlaying="y",
            side="right",
        ),
        template="plotly_dark",
    )
    st.plotly_chart(fig)


##############
### PART 3 ###
# STREAMLIT ##
##############

# Streamlit app
st.sidebar.title("Predicting Stock Prices by News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., SBUX):", value="SBUX")
run_button = st.sidebar.button("Run Analysis")

if run_button:
    news_df = get_news_data(ticker)
    result_df = process_sentiment_data(news_df)
    start_date = result_df["DateOnly"].min().strftime("%Y-%m-%d")
    end_date = result_df["DateOnly"].max().strftime("%Y-%m-%d")
    stock_data = get_stock_data(ticker, start_date, end_date)
    combined_df = combine_data(result_df, stock_data)
    correlation_pct_change = calculate_correlation(combined_df)
    st.write(
        f"Pearson correlation between lagged sentiment score and stock percentage change: {correlation_pct_change}"
    )
    forecast_mean, forecast_ci, forecast_index = fit_and_forecast(combined_df)
    create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)

####
## for fintech
"""
streamlit
yfinance==0.2.40
pandas
plotly
kaleido
watchdog
holidays
finvizfinance
statsmodels
pandas_ta
scikit-learn
"""
