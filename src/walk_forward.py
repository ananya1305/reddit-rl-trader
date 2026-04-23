"""
walk_forward.py
---------------
Walk-forward validation across 3 expanding train/test windows.

Windows (expanding — each adds one more year of training data):
  Window 1:  Train 2019-01-01 → 2021-12-31   Test 2022
  Window 2:  Train 2019-01-01 → 2022-12-31   Test 2023
  Window 3:  Train 2019-01-01 → 2023-12-31   Test 2024  ← same as original

Run from project root:
  python src/walk_forward.py

Outputs:
  results/walk_forward/walk_forward_results.csv
  results/walk_forward/walk_forward_summary.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from src.env import RedditTradingEnv

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS     = ["GME", "TSLA", "AAPL", "AMC"]
TIMESTEPS   = 500_000
RESULTS_DIR = "results/walk_forward"
MODELS_DIR  = "models/walk_forward"

WINDOWS = [
    {
        "name":        "Window_1_2022",
        "train_start": "2019-01-01",
        "train_end":   "2021-12-31",
        "test_start":  "2022-01-01",
        "test_end":    "2022-12-31",
    },
    {
        "name":        "Window_2_2023",
        "train_start": "2019-01-01",
        "train_end":   "2022-12-31",
        "test_start":  "2023-01-01",
        "test_end":    "2023-12-31",
    },
    {
        "name":        "Window_3_2024",
        "train_start": "2019-01-01",
        "train_end":   "2023-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2024-12-31",
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def make_env_fn(ticker, mode, use_reddit, window):
    def _init():
        env = RedditTradingEnv(
            ticker     = ticker,
            mode       = mode,
            use_reddit = use_reddit,
            train_end  = window["train_end"],
            test_start = window["test_start"],
            test_end   = window["test_end"],
        )
        return Monitor(env)
    return _init


def train_model(ticker, window, algo, use_reddit):
    label   = f"{algo}_{'reddit' if use_reddit else 'no_reddit'}"
    wname   = window["name"]
    savedir = f"{MODELS_DIR}/{wname}/{label}/{ticker}"
    os.makedirs(savedir, exist_ok=True)
    best_path = f"{savedir}/best_model"

    print(f"\n  Training {label.upper()} | {ticker} | {wname} "
          f"(train through {window['train_end'][:4]})")

    train_env = make_vec_env(
        make_env_fn(ticker, "train", use_reddit, window), n_envs=1
    )
    eval_env = Monitor(RedditTradingEnv(
        ticker=ticker, mode="test", use_reddit=use_reddit,
        train_end=window["train_end"],
        test_start=window["test_start"],
        test_end=window["test_end"],
    ))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = best_path,
        eval_freq            = 5_000,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0,
    )

    if algo == "ppo":
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=256, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, verbose=0,
        )
    else:
        model = DQN(
            "MlpPolicy", train_env,
            learning_rate=1e-4, buffer_size=50_000,
            learning_starts=1_000, batch_size=64,
            gamma=0.99, train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05, verbose=0,
        )

    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=True)
    train_env.close()
    eval_env.close()
    return f"{best_path}/best_model.zip"


def evaluate_model(model_path, ticker, window, algo, use_reddit):
    test_env = RedditTradingEnv(
        ticker=ticker, mode="test", use_reddit=use_reddit,
        train_end=window["train_end"],
        test_start=window["test_start"],
        test_end=window["test_end"],
    )
    model = PPO.load(model_path, env=test_env) if algo == "ppo" \
            else DQN.load(model_path, env=test_env)

    obs, _ = test_env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated

    metrics = compute_metrics(
        test_env.portfolio_history, test_env.initial_cash, test_env.trade_history
    )
    test_env.close()
    return metrics


def buy_and_hold(ticker, window):
    prices = pd.read_csv(
        f"data/{ticker}_prices.csv", index_col=0, parse_dates=True
    )
    test = prices[
        (prices.index >= window["test_start"]) &
        (prices.index <= window["test_end"])
    ]["Close"].values

    initial = 10_000.0
    shares  = int(initial // test[0])
    cash    = initial - shares * test[0]
    port    = [shares * p + cash for p in test]
    return compute_metrics(port, initial, [{"action": "BUY"}])


def compute_metrics(portfolio_values, initial_cash, trades):
    series  = pd.Series(portfolio_values)
    returns = series.pct_change().dropna()
    cum_return   = (series.iloc[-1] - initial_cash) / initial_cash * 100
    sharpe       = (returns.mean() / returns.std()) * np.sqrt(252) \
                   if returns.std() > 0 else 0.0
    peak         = series.cummax()
    max_drawdown = ((series - peak) / peak).min() * 100
    return {
        "final_value":       round(series.iloc[-1], 2),
        "cumulative_return": round(cum_return, 2),
        "sharpe_ratio":      round(sharpe, 3),
        "max_drawdown":      round(max_drawdown, 2),
        "total_trades":      len(trades),
    }


def run_walk_forward():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    all_rows = []

    for window in WINDOWS:
        wname = window["name"]
        print(f"\n{'='*60}")
        print(f"WINDOW: {wname}")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Test:  {window['test_start']} to {window['test_end']}")
        print(f"{'='*60}")

        for ticker in TICKERS:
            print(f"\n--- {ticker} ---")

            for algo, use_reddit, label in [
                ("ppo", True,  "PPO + Reddit"),
                ("ppo", False, "PPO no Reddit"),
                ("dqn", True,  "DQN + Reddit"),
            ]:
                try:
                    path    = train_model(ticker, window, algo, use_reddit)
                    metrics = evaluate_model(path, ticker, window, algo, use_reddit)
                    all_rows.append({
                        "window": wname, "test_year": window["test_start"][:4],
                        "ticker": ticker, "model": label, **metrics,
                    })
                    print(f"  {label:20s} | Sharpe {metrics['sharpe_ratio']:.3f} "
                          f"| Return {metrics['cumulative_return']:+.1f}%")
                except Exception as e:
                    print(f"  ERROR {label} {ticker}: {e}")

            try:
                bah = buy_and_hold(ticker, window)
                all_rows.append({
                    "window": wname, "test_year": window["test_start"][:4],
                    "ticker": ticker, "model": "Buy-and-Hold", **bah,
                })
                print(f"  {'Buy-and-Hold':20s} | Sharpe {bah['sharpe_ratio']:.3f} "
                      f"| Return {bah['cumulative_return']:+.1f}%")
            except Exception as e:
                print(f"  ERROR Buy-and-Hold {ticker}: {e}")

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv(f"{RESULTS_DIR}/walk_forward_results.csv", index=False)

    summary = (
        df.groupby(["ticker", "model"])["sharpe_ratio"]
        .agg(["mean", "std", "min", "max"])
        .round(3).reset_index()
    )
    summary.columns = ["Ticker", "Model",
                        "Mean Sharpe", "Std Sharpe", "Min Sharpe", "Max Sharpe"]
    summary.to_csv(f"{RESULTS_DIR}/walk_forward_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY  (mean across 3 test years)")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    return df, summary


if __name__ == "__main__":
    run_walk_forward()