import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_automated_data(ticker, start, end):
    # Fetch data
    df = yf.download(ticker, start=start, end=end, interval="1d")

    # Handle yfinance MultiIndex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. PRESERVE ORIGINAL OHLC (Needed for backtesting.py)
    # We create a copy so scaling doesn't overwrite these original values
    ohlc_data = df[['Open', 'High', 'Low', 'Close']].copy()

    # 2. Add Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Price_Diff'] = df['Close'].diff()
    df = df.dropna()

    # 3. Scaling (Only for the RL features)
    scaler = StandardScaler()
    features_to_scale = ['Close', 'SMA_20', 'SMA_50']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 4. Final Merge
    # We return the scaled features + original OHLC for backtesting.py
    # Note: 'Close' in backtesting will be the unscaled one if we aren't careful.
    # It is safer to give backtesting.py its own clean DataFrame.
    return df