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
        # --- Corrected Metrics Calculation ---
        results = test_df.iloc[:len(signals)].copy()
        results['Signal'] = signals

        # Calculate daily percent change of the price
        results['Pct_Change'] = results['Close'].pct_change().fillna(0)

        # Ensure the signal applies to the NEXT day's return
        results['Strategy_Return'] = results['Signal'].shift(1) * results['Pct_Change']
        results['Strategy_Return'] = results['Strategy_Return'].fillna(0)

        # Sharpe Ratio: Annualized
        # Use a risk-free rate of 0 for simplicity
        std_dev = results['Strategy_Return'].std()
        if std_dev > 0:
            sharpe = np.sqrt(252) * (results['Strategy_Return'].mean() / std_dev)
        else:
            sharpe = 0.0

        # --- Corrected Visualization ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Top Plot: Market vs Strategy (Growth)
        ax1.plot((1 + results['Pct_Change']).cumprod(), label='Market', color='black', alpha=0.3)
        ax1.plot((1 + results['Strategy_Return']).cumprod(), label='RL Strategy', color='blue')
        ax1.set_title(f"Backtest for {ticker} | Sharpe: {sharpe:.2f}")
        ax1.legend()

        # Bottom Plot: Signal (To see if it's actually trading)
        ax2.step(results.index, results['Signal'], label='Position (1:Long, 0:Flat, -1:Short)', color='orange')
        ax2.set_ylim(-1.5, 1.5)
        ax2.legend()

        plt.savefig(f"results/backtest_{ticker.replace('^', '')}.png")