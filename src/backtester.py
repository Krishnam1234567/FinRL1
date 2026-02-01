import os
from stable_baselines3 import PPO
from src.data_processor import get_clean_data
from src.environment import StockTradingEnv

# Create folders if they don't exist
os.makedirs("models", exist_ok=True)

# 1. Prepare Data
df = get_clean_data("data/btcusd_1-min_data (2).csv")
split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]

# 2. Setup Environments
train_env = StockTradingEnv(train_df)

# 3. Initialize & Train Agent
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/")
print("Starting Training...")
model.learn(total_timesteps=200000)

# 4. Save the Model
model.save("models/ppo_nifty_model")
print("Model Saved!")