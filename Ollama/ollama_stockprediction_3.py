import yfinance as yf
import pandas_ta as ta
# need fix for issue: "from numpy import NaN as npNaN".
# pip install -U git+https://github.com/twopirllc/pandas-ta.git@development
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go


print("Load in Stock Ticker Price with yfinance Library")
df = yf.download('NVDA', start="2022-12-31", end="2024-12-31")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['Previous_Close'] = df['Close'].shift(1)  # Add previous day's close as a feature
df['Close_shifted'] = df['Close'].shift(1)
df['Open_shifted'] = df['Open'].shift(1)
df['High_shifted'] = df['High'].shift(1)
df['Low_shifted'] = df['Low'].shift(1)
df

# Calculate technical indicators on the shifted data

# Simple Moving Average (SMA): Average price over the last 50 periods
df['SMA_50'] = ta.sma(df['Close_shifted'], length=50)

# Exponential Moving Average (EMA): Weighted average that reacts faster to recent price changes, using 50 periods
df['EMA_50'] = ta.ema(df['Close_shifted'], length=50)

# Relative Strength Index (RSI): Momentum indicator that measures the magnitude of recent price changes to evaluate overbought/oversold conditions, using a 14-period lookback
df['RSI'] = ta.rsi(df['Close_shifted'], length=14)

# Moving Average Convergence Divergence (MACD): Trend-following momentum indicator, using 12 and 26 periods for the fast and slow EMAs and a 9-period signal line
macd = ta.macd(df['Close_shifted'], fast=12, slow=26, signal=9)
df['MACD'] = macd['MACD_12_26_9']        # MACD line
df['Signal_Line'] = macd['MACDs_12_26_9'] # Signal line

# Bollinger Bands: Volatility indicator using a 20-period moving average and 2 standard deviations
bollinger = ta.bbands(df['Close_shifted'], length=20, std=2)
df['BB_Upper'] = bollinger['BBU_20_2.0']  # Upper Bollinger Band
df['BB_Middle'] = bollinger['BBM_20_2.0'] # Middle Band (20-period SMA)
df['BB_Lower'] = bollinger['BBL_20_2.0']  # Lower Bollinger Band

# Stochastic Oscillator: Momentum indicator comparing closing prices to price ranges over 14 periods with a 3-period %D moving average
stoch = ta.stoch(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], k=14, d=3)
df['%K'] = stoch['STOCHk_14_3_3'] # %K line (main line)
df['%D'] = stoch['STOCHd_14_3_3'] # %D line (3-period moving average of %K)

# Average True Range (ATR): Volatility indicator measuring the average range of price movement over the last 14 periods
df['ATR'] = ta.atr(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], length=14)
print("==========================")
# Drop rows with missing values due to shifting and indicator calculation
df.dropna(inplace=True)
df

print("Choose # of days for rolling training data window and Choose Technical Indicators")
# Parameters
window_size = 20  # 4 weeks of trading days (5 days per week * 4)

# List of indicators to test, including Previous_Close
indicators = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR', 'Close_shifted', 'Previous_Close']

# Initialize a dictionary to store predictions, actuals, daily MAE for each indicator
results = {indicator: {'predictions': [], 'actual': [], 'daily_mae': []} for indicator in indicators}

print("Loop over multiple 20-Day Train Datasets for Model Building and Next Day Test Datasets for Model Evaluatio")
# Sequentially predict the actual close price using a rolling 4 weeks window, set by window_size
for i in range(window_size, len(df) - 1):
    train_df = df.iloc[i - window_size:i]  # Training window 
    test_index = i + 1  # Index of next day's prediction
    actual_close_price = df['Close'].iloc[test_index]  # Next day's actual closing price

    # Individual indicators as predictors (plus Previous_Close)
    for indicator in indicators[:-1]:  # Exclude Previous_Close from standalone tests
        X_train = train_df[[indicator, 'Previous_Close']]
        y_train = train_df['Close']
        X_train = sm.add_constant(X_train)  # Add constant for intercept

        model = sm.OLS(y_train, X_train).fit()
        X_test = pd.DataFrame({indicator: [df[indicator].iloc[test_index]], 'Previous_Close': [df['Previous_Close'].iloc[test_index]]})
        X_test = sm.add_constant(X_test, has_constant='add')  # Add constant for prediction

        prediction = model.predict(X_test)[0]
        results[indicator]['predictions'].append(prediction)
        results[indicator]['actual'].append(actual_close_price)
        
        daily_mae = mean_absolute_error([actual_close_price], [prediction])
        results[indicator]['daily_mae'].append(daily_mae)
        
# Calculate accuracy metrics (MAE, MSE) for each individual indicator and the combined model
accuracy_data = {
    'Indicator': [],
    'MAE': [],
    'MSE': []
}

for indicator in indicators[:-1]:  # Exclude Previous_Close from standalone tests in accuracy table
    if results[indicator]['actual']:  # Check if there are results for this indicator
        mae = mean_absolute_error(results[indicator]['actual'], results[indicator]['predictions'])
        mse = mean_squared_error(results[indicator]['actual'], results[indicator]['predictions'])
        accuracy_data['Indicator'].append(indicator)
        accuracy_data['MAE'].append(mae)
        accuracy_data['MSE'].append(mse)


# Create accuracy DataFrame
accuracy_df = pd.DataFrame(accuracy_data).sort_values(by='MAE').reset_index(drop=True)
accuracy_df

print("4. Plotting the Results")
# Create faceted plot with each indicator's daily MAE
fig = make_subplots(rows=len(indicators), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    subplot_titles=[f"{indicator} Daily MAE" for indicator in indicators[:-1]])

# Find the global y-axis range across all indicators
y_values = [results[indicator]['daily_mae'] for indicator in indicators[:-1]]
y_min = min(min(y) for y in y_values)
y_max = max(max(y) for y in y_values)

# Add each individual indicator's daily MAE
for idx, indicator in enumerate(indicators[:-1]):
    fig.add_trace(
        go.Scatter(
            x=df.index[window_size + 1:],  # Start date after the initial window
            y=results[indicator]['daily_mae'],
            mode='lines',
            name=f'{indicator} Daily MAE'
        ),
        row=idx + 1, col=1
    )

# Update layout with shared y-axis range and individual x-axis labels
fig.update_yaxes(range=[y_min, y_max])  # Apply the common y-axis range across all subplots
fig.update_xaxes(title_text="Date", row=len(indicators), col=1)  # Add x-axis label for the last row

# Final layout adjustments
fig.update_layout(
    height=150 * (len(indicators)),  # Adjust height for the combined model
    title="Daily MAE of Each Technical Indicator on NVDA Closing Price",
    yaxis_title="Daily MAE",
    showlegend=False,
    template="plotly_white"
)

fig.show()


# Create the figure
fig = go.Figure()

# Add Close price
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='white', width=1)))

# Add SMA, EMA
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='yellow', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='orange', width=1)))

# Add Bollinger Bands
fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='blue', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='blue', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='blue', width=1)))

# Add MACD and Signal Line
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan', width=1)))
fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='purple', width=1)))

# Configure layout
fig.update_layout(
    title="Overlay of Technical Indicators on NVDA Close Price",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color="white"),
    width=800,  # Width of the slide, adjust as needed
    height=600   # Height of the slide, adjust as needed
)

fig.show()