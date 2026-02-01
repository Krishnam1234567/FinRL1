import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom modules
from src.data_processor import get_clean_data
from src.environment import StockTradingEnv


def main():
    # 1. Folders for organization
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 2. Load and Split Data
    print("--- Loading Data ---")
    file_path = "data/btcusd_1-min_data (2).csv"
    df = get_clean_data(file_path)

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    # 3. Initialize Training Environment
    env = DummyVecEnv([lambda: StockTradingEnv(train_df)])

    # 4. Define and Train Model
    # We use MlpPolicy because our data is a simple feature vector
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/", learning_rate=0.0003)

    print("--- Training RL Agent (200,000 steps) ---")
    model.learn(total_timesteps=200000)
    model.save("models/ppo_nifty_final")

    # 5. Testing / Backtesting
    print("--- Running Backtest on Test Data ---")
    test_env = StockTradingEnv(test_df)
    obs, _ = test_env.reset()

    signals = []
    for _ in range(len(test_df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        signals.append(test_env.position)

    # 6. Performance Visualization
    results = test_df.iloc[:len(signals)].copy()
    results['Signal'] = signals
    results['Market_Ret'] = results['Close'].pct_change().fillna(0)
    results['Strategy_Ret'] = results['Market_Ret'] * results['Signal'].shift(1).fillna(0)

    results['Cum_Market'] = (1 + results['Market_Ret']).cumprod()
    results['Cum_Strategy'] = (1 + results['Strategy_Ret']).cumprod()

    plt.figure(figsize=(12, 5))
    plt.plot(results['Cum_Market'], label='Market (Buy & Hold)', color='black', linestyle='--')
    plt.plot(results['Cum_Strategy'], label='RL Strategy', color='blue')
    plt.title("PPO Agent vs Nifty 50 Market Performance")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()