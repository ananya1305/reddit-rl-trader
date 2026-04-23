import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class RedditTradingEnv(gym.Env):
    """Gymnasium trading environment with optional Reddit embeddings."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, ticker: str, mode: str = "train", use_reddit: bool = True):
        super().__init__()
        self.ticker     = ticker
        self.mode       = mode
        self.use_reddit = use_reddit
        self._load_data()

        self.action_space = spaces.Discrete(3)

        obs_dim = 5 + 3 + 384 if use_reddit else 5
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32
        )

        self.initial_cash      = 10_000.0
        self.cash              = self.initial_cash
        self.shares_held       = 0
        self.current_step      = 0
        self.portfolio_value   = self.initial_cash
        self.portfolio_history = []
        self.trade_history     = []

    def _load_data(self):
        prices = pd.read_csv(
            f"data/{self.ticker}_prices.csv", index_col=0, parse_dates=True
        )

        reddit     = pd.read_csv(
            f"data/embeddings/{self.ticker}_reddit_features.csv",
            index_col=0, parse_dates=True
        )
        embeddings = np.load(f"data/embeddings/{self.ticker}_embeddings.npy")

        emb_cols = [f"emb_{i}" for i in range(384)]
        emb_df   = pd.DataFrame(embeddings, index=reddit.index, columns=emb_cols)
        reddit   = pd.concat([reddit, emb_df], axis=1)

        # left join; missing Reddit days become zeros
        self.data = prices.join(reddit, how="left")
        self.data[reddit.columns] = self.data[reddit.columns].fillna(0)

        # Normalize Reddit scalars to [0, 1]
        for col in ["post_count", "avg_score", "avg_comments"]:
            max_val = self.data[col].max()
            if max_val > 0:
                self.data[col] = self.data[col] / max_val

        # 80/20 chronological split
        split = int(len(self.data) * 0.8)
        self.data = self.data.iloc[:split] if self.mode == "train" else self.data.iloc[split:]
        self.data.reset_index(inplace=True)
        print(f"  {self.ticker} [{self.mode}]: {len(self.data)} trading days loaded")

    def _get_observation(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]

        market_features = np.array([
            row["return"],
            row["volume_zscore"],
            row["volatility"],
            row["rsi"] / 100.0,
            row["ma_cross"],
        ], dtype=np.float32)

        if not self.use_reddit:
            obs = market_features
        else:
            reddit_features = np.array([
                row["post_count"],
                row["avg_score"],
                row["avg_comments"],
            ], dtype=np.float32)

            emb_cols  = [f"emb_{i}" for i in range(384)]
            embedding = row[emb_cols].values.astype(np.float32)
            obs       = np.concatenate([market_features, reddit_features, embedding])

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_portfolio_value(self) -> float:
        current_price = self.data.iloc[self.current_step]["Close"]
        return self.cash + self.shares_held * current_price

    def step(self, action: int):
        current_price  = self.data.iloc[self.current_step]["Close"]
        prev_value     = self._get_portfolio_value()
        volatility     = self.data.iloc[self.current_step]["volatility"]
        trade_cost     = 0.001  # 0.1% transaction cost
        trade_executed = False

        if action == 1:  # BUY
            if self.cash >= current_price:
                shares_to_buy = int(self.cash // current_price)
                cost          = shares_to_buy * current_price * (1 + trade_cost)
                if cost <= self.cash:
                    self.shares_held += shares_to_buy
                    self.cash        -= cost
                    trade_executed    = True
                    self.trade_history.append({
                        "step": self.current_step, "action": "BUY",
                        "price": current_price, "shares": shares_to_buy
                    })

        elif action == 2:  # SELL
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price * (1 - trade_cost)
                self.cash += proceeds
                self.trade_history.append({
                    "step": self.current_step, "action": "SELL",
                    "price": current_price, "shares": self.shares_held
                })
                self.shares_held = 0
                trade_executed   = True

        self.current_step   += 1
        new_value            = self._get_portfolio_value()
        self.portfolio_value = new_value
        self.portfolio_history.append(new_value)

        daily_return  = (new_value - prev_value) / prev_value
        reward        = daily_return - 0.1 * volatility - (trade_cost if trade_executed else 0.0)
        terminated    = self.current_step >= len(self.data) - 1

        obs  = self._get_observation()
        info = {
            "portfolio_value": new_value,
            "cash":            self.cash,
            "shares_held":     self.shares_held,
            "current_price":   current_price
        }
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cash              = self.initial_cash
        self.shares_held       = 0
        self.current_step      = 0
        self.portfolio_value   = self.initial_cash
        self.portfolio_history = []
        self.trade_history     = []
        return self._get_observation(), {}

    def render(self, mode="human"):
        print(
            f"Step {self.current_step} | "
            f"Portfolio: ${self.portfolio_value:,.2f} | "
            f"Cash: ${self.cash:,.2f} | "
            f"Shares: {self.shares_held}"
        )
