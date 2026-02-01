# backtesting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from src.environment import StockTradingEnv
from src.data_processor import get_automated_data


def run_pro_backtest(ticker="^NSEI", model_path=None):
    if model_path is None:
        model_path = f"models/ppo_{ticker.replace('^', '')}_2026"

    # 1. Load Model & Data
    model = PPO.load(model_path)
    df = get_automated_data(ticker=ticker, start="2023-01-01", end="2026-02-01")

    # Isolate test data (final 20%)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    env = StockTradingEnv(test_df)
    obs, _ = env.reset()

    # 2. Simulation Loop
    signals = []
    for _ in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        signals.append(env.position)

    # 3. Metrics Calculation (Sharpe Ratio)
    results = test_df.iloc[:len(signals)].copy()
    results['Signal'] = signals
    results['Market_Return'] = results['Close'].pct_change().fillna(0)
    results['Strategy_Return'] = results['Market_Return'] * results['Signal'].shift(1).fillna(0)

    # Sharpe Ratio: sqrt(252) * (mean / std)
    sharpe = np.sqrt(252) * (results['Strategy_Return'].mean() / (results['Strategy_Return'].std() + 1e-9))

    # 4. Save PNG Report
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot((1 + results['Strategy_Return']).cumprod(), label='RL Strategy', color='blue')
    plt.plot((1 + results['Market_Return']).cumprod(), label='Market', color='black', alpha=0.3)
    plt.title(f"Backtest for {ticker} | Sharpe: {sharpe:.2f}")
    plt.legend()
    plt.savefig(f"results/backtest_{ticker.replace('^', '')}.png")
    plt.close()  # Close to free memory

    return sharpe