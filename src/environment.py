import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index()
        self.action_space = spaces.Discrete(3)
        # State: Close, SMA_50, SMA_200, Vol_20, Position (5 elements)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.position = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            self.df['Close'].iloc[self.current_step],
            self.df['SMA_50'].iloc[self.current_step],
            self.df['SMA_200'].iloc[self.current_step],
            self.df['Vol_20'].iloc[self.current_step],
            self.position
        ], dtype=np.float32)

    def step(self, action):
        price_change = self.df['Price_Diff'].iloc[self.current_step]

        # 0:Short, 1:Flat, 2:Long
        if action == 2:
            reward = price_change * 100
            self.position = 1
        elif action == 0:
            reward = -price_change * 100
            self.position = -1
        else:
            reward = -0.001
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, np.float32(reward), done, False, {}