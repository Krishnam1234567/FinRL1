import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_automated_data(ticker, start, end):
    # Fetch data directly from Yahoo Finance
    df = yf.download(ticker, start=start, end=end, interval="1d")

    # Handle MultiIndex columns common in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate basic Technical Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Price_Diff'] = df['Close'].diff()


    df = df.dropna()

    # Scaling for the Neural Network
    scaler = StandardScaler()
    features = ['Close', 'SMA_20', 'SMA_50']
    df[features] = scaler.fit_transform(df[features])

    return df[['Close', 'SMA_20', 'SMA_50', 'Price_Diff']]