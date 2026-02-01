import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom Imports
from src.data_processor import get_automated_data
from src.environment import StockTradingEnv
from src.backtester import run_pro_backtest


def main():
    os.makedirs("models", exist_ok=True)
    TICKER = "TCS.NS"

    # 1. Training Phase
    df = get_automated_data(ticker=TICKER, start="2021-01-01", end="2026-02-01")
    split = int(len(df) * 0.6)
    train_df = df.iloc[:split]

    train_env = DummyVecEnv([lambda: StockTradingEnv(train_df)])
    model = PPO("MlpPolicy", train_env, verbose=1)

    print("--- Starting Training ---")
    model.learn(total_timesteps=5000)

    model_path = f"models/ppo_{TICKER.replace('^', '')}_2026"
    model.save(model_path)
    print(f"Model Saved at {model_path}!")

    # 2. Automated Backtesting Phase (Using the Module)
    print("--- Starting Automated Backtest ---")
    sharpe = run_pro_backtest(ticker=TICKER, model_path=model_path)

    print(f"Workflow Complete!")
    print(f"Final Sharpe Ratio: {sharpe:.2f}")
    print(f"Performance Chart: results/backtest_{TICKER.replace('^', '')}.png")


if __name__ == "__main__":
    main()