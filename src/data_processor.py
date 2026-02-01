import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_clean_data(file_path):
    # Load raw data
    df = pd.read_csv(file_path)

    # ADJUSTED: Only 6 names to match your file structure
    # Based on Investing.com standard 6-column exports:
    df.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Change_Pct']

    # 1. Clean Numeric Columns (Remove commas and convert to float)
    # Note: We skip 'Date' and 'Change_Pct' here as they are strings/dates
    cols_to_fix = ['Close', 'Open', 'High', 'Low']
    for col in cols_to_fix:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)

    # 2. Convert Date (Format: 01-02-2026)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # 3. Sort newest to oldest -> oldest to newest
    df = df.sort_values('Date').reset_index(drop=True).set_index('Date')

    # 4. Feature Engineering (Technical Indicators)
    # We use a smaller window for SMA if your dataset is small
    df['SMA_50'] = df['Close'].rolling(window=min(len(df), 50)).mean()
    df['SMA_200'] = df['Close'].rolling(window=min(len(df), 200)).mean()
    df['Vol_20'] = df['Close'].pct_change().rolling(window=20).std()
    df['Price_Diff'] = df['Close'].diff()

    df = df.dropna()

    # 5. Scaling features for the AI
    scaler = StandardScaler()
    features = ['Close', 'SMA_50', 'SMA_200', 'Vol_20']
    df[features] = scaler.fit_transform(df[features])

    return df[['Close', 'SMA_50', 'SMA_200', 'Vol_20', 'Price_Diff']]