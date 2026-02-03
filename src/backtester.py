import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from src.environment import StockTradingEnv
from src.data_processor import get_automated_data

def calculate_max_drawdown(cumulative_returns):
    """Helper to calculate Max Drawdown from cumulative returns series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def run_pro_backtest(ticker="^NSEI", model_path=None, transaction_cost=0.001):
    """
    Runs a backtest on the final 20% of data.
    
    Args:
        ticker (str): Ticker symbol.
        model_path (str): Path to saved PPO model.
        transaction_cost (float): Cost per trade (e.g., 0.001 = 0.1%).
    """
    # 1. Configuration & Directories
    if model_path is None:
        model_path = f"models/ppo_{ticker.replace('^', '')}_2026"
    
    os.makedirs("results", exist_ok=True)

    # 2. Load Model & Data
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print("Error: Model file not found.")
        return

    # Fetch data (Ensure this matches training timeframe logic)
    df = get_automated_data(ticker=ticker, start="2023-01-01", end="2026-02-01")

    # Isolate test data (final 20%)
    split = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()
    
    # Initialize Environment
    env = StockTradingEnv(test_df)
    obs, _ = env.reset()

    # 3. Fast Simulation Loop (Collect Actions Only)
    print("Running simulation...")
    signals = []
    
    # We iterate len(test_df) - 1 because the last state has no 'next' price
    for _ in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        # Assuming env.position logs the position held AFTER the step
        signals.append(env.position)

    # 4. Vectorized Backtest Calculation (The "Pro" Way)
    # Align signals with the dataframe length. 
    # Note: We simulated len-1 steps. The last row has no signal for 'tomorrow'.
    analysis_df = test_df.iloc[:len(signals)].copy()
    analysis_df['Signal'] = signals

    # Market Return (Close-to-Close)
    analysis_df['Pct_Change'] = analysis_df['Close'].pct_change().fillna(0)

    # Strategy Return Calculation
    # We shift signal by 1: The position chosen 'yesterday' captures 'today's' price move.
    analysis_df['Position_Yesterday'] = analysis_df['Signal'].shift(1).fillna(0)
    
    # Raw Strategy Return
    analysis_df['Strategy_Return'] = analysis_df['Position_Yesterday'] * analysis_df['Pct_Change']

    # Transaction Costs Logic
    # Cost applies when Position_Yesterday != Position_Day_Before
    trades = analysis_df['Position_Yesterday'].diff().fillna(0).abs()
    # If position changes from 1 to -1, diff is 2. We pay cost on the volume traded? 
    # Simplified: We pay cost on the portfolio value turnover.
    # Usually, cost is applied to the notional value changed.
    analysis_df['Costs'] = trades * transaction_cost
    
    # Net Strategy Return
    analysis_df['Net_Strategy_Return'] = analysis_df['Strategy_Return'] - analysis_df['Costs']

    # 5. Calculate Metrics
    # Cumulative Returns
    analysis_df['Cum_Market'] = (1 + analysis_df['Pct_Change']).cumprod()
    analysis_df['Cum_Strategy'] = (1 + analysis_df['Net_Strategy_Return']).cumprod()

    # Total Return
    total_return = analysis_df['Cum_Strategy'].iloc[-1] - 1
    market_return = analysis_df['Cum_Market'].iloc[-1] - 1

    # Sharpe Ratio (Annualized) - Assuming 252 trading days
    std_dev = analysis_df['Net_Strategy_Return'].std()
    if std_dev > 0:
        sharpe = np.sqrt(252) * (analysis_df['Net_Strategy_Return'].mean() / std_dev)
    else:
        sharpe = 0.0

    # Max Drawdown
    max_dd = calculate_max_drawdown(analysis_df['Cum_Strategy'])
    
    # Win Rate (Days with positive return when active)
    active_days = analysis_df[analysis_df['Position_Yesterday'] != 0]
    if len(active_days) > 0:
        win_rate = len(active_days[active_days['Net_Strategy_Return'] > 0]) / len(active_days)
    else:
        win_rate = 0.0

    print("="*40)
    print(f"BACKTEST RESULTS: {ticker}")
    print("="*40)
    print(f"Total Return:    {total_return*100:.2f}% (Market: {market_return*100:.2f}%)")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_dd*100:.2f}%")
    print(f"Win Rate:        {win_rate*100:.1f}%")
    print("="*40)

    # 6. Professional Visualization
    plt.style.use('bmh') # Cleaner style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Plot: Equity Curves
    ax1.plot(analysis_df.index, analysis_df['Cum_Market'], label='Buy & Hold (Market)', color='grey', alpha=0.5, linewidth=1.5)
    ax1.plot(analysis_df.index, analysis_df['Cum_Strategy'], label='RL Strategy (Net)', color='#2980b9', linewidth=2)
    
    # Fill drawdown area
    ax1.fill_between(analysis_df.index, analysis_df['Cum_Strategy'], 1, where=(analysis_df['Cum_Strategy'] < 1), color='red', alpha=0.05)
    
    ax1.set_title(f"Backtest: {ticker} | Sharpe: {sharpe:.2f} | Return: {total_return*100:.1f}%", fontsize=14)
    ax1.set_ylabel("Growth ($1 Invested)")
    ax1.legend(loc="upper left")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Bottom Plot: Positions
    # Create a filled area for positions to make it easier to read than a line
    ax2.fill_between(analysis_df.index, analysis_df['Signal'], step="post", alpha=0.4, color='orange', label='Position')
    ax2.step(analysis_df.index, analysis_df['Signal'], where='post', color='orange', linewidth=1)
    
    ax2.set_ylabel("Position")
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    save_path = f"results/backtest_{ticker.replace('^', '')}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close() # Close memory

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd
    }

if __name__ == "__main__":
    run_pro_backtest(ticker="^NSEI")