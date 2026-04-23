import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings("ignore")

from src.env import RedditTradingEnv

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS    = ["GME", "TSLA", "AAPL", "AMC"]
MODELS_DIR = "models"
LOGS_DIR   = "logs"
TIMESTEPS  = 500_000
# ─────────────────────────────────────────────────────────────────────────────


def make_env(ticker: str, mode: str = "train", use_reddit: bool = True):
    """Create a monitored environment for a given ticker."""
    def _init():
        env = RedditTradingEnv(ticker, mode=mode, use_reddit=use_reddit)
        env = Monitor(env)
        return env
    return _init


def train_ppo(ticker: str, use_reddit: bool = True):
    """Train a PPO agent on a single ticker."""
    label = "ppo_reddit" if use_reddit else "ppo_no_reddit"

    print(f"\n{'='*50}")
    print(f"Training PPO ({'with' if use_reddit else 'without'} Reddit) on {ticker}")
    print(f"{'='*50}")

    os.makedirs(f"{MODELS_DIR}/{label}", exist_ok=True)
    os.makedirs(f"{LOGS_DIR}/{label}/{ticker}", exist_ok=True)

    train_env = make_vec_env(make_env(ticker, "train", use_reddit), n_envs=1)
    eval_env  = Monitor(RedditTradingEnv(ticker, mode="test", use_reddit=use_reddit))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"{MODELS_DIR}/{label}/{ticker}_best",
        log_path             = f"{LOGS_DIR}/{label}/{ticker}",
        eval_freq            = 5000,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 256,
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        verbose         = 1,
        tensorboard_log = f"{LOGS_DIR}/{label}/{ticker}"
    )

    model.learn(
        total_timesteps = TIMESTEPS,
        callback        = eval_callback,
        progress_bar    = True
    )

    save_path = f"{MODELS_DIR}/{label}/{ticker}_final"
    model.save(save_path)
    print(f"  Saved → {save_path}")

    train_env.close()
    eval_env.close()
    return model


def train_dqn(ticker: str, use_reddit: bool = True):
    """Train a DQN agent on a single ticker."""
    label = "dqn_reddit" if use_reddit else "dqn_no_reddit"

    print(f"\n{'='*50}")
    print(f"Training DQN ({'with' if use_reddit else 'without'} Reddit) on {ticker}")
    print(f"{'='*50}")

    os.makedirs(f"{MODELS_DIR}/{label}", exist_ok=True)
    os.makedirs(f"{LOGS_DIR}/{label}/{ticker}", exist_ok=True)

    train_env = Monitor(RedditTradingEnv(ticker, mode="train", use_reddit=use_reddit))
    eval_env  = Monitor(RedditTradingEnv(ticker, mode="test", use_reddit=use_reddit))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = f"{MODELS_DIR}/{label}/{ticker}_best",
        log_path             = f"{LOGS_DIR}/{label}/{ticker}",
        eval_freq            = 5000,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0
    )

    model = DQN(
        policy                 = "MlpPolicy",
        env                    = train_env,
        learning_rate          = 1e-4,
        buffer_size            = 50_000,
        learning_starts        = 1000,
        batch_size             = 64,
        gamma                  = 0.99,
        train_freq             = 4,
        target_update_interval = 1000,
        exploration_fraction   = 0.3,
        exploration_final_eps  = 0.05,
        verbose                = 1,
        tensorboard_log        = f"{LOGS_DIR}/{label}/{ticker}"
    )

    model.learn(
        total_timesteps = TIMESTEPS,
        callback        = eval_callback,
        progress_bar    = True
    )

    save_path = f"{MODELS_DIR}/{label}/{ticker}_final"
    model.save(save_path)
    print(f"  Saved → {save_path}")

    train_env.close()
    eval_env.close()
    return model


def train_all():
    """Train all models — PPO and DQN with and without Reddit on all tickers."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    for ticker in TICKERS:
        train_ppo(ticker, use_reddit=True)   # full model
        train_ppo(ticker, use_reddit=False)  # ablation
        train_dqn(ticker, use_reddit=True)   # full model


if __name__ == "__main__":
    train_all()