"""
Tests for src/env.py's RedditTradingEnv.

These tests build minimal synthetic price/reddit/embedding files on disk
(matching the exact schema RedditTradingEnv._load_data() reads) instead of
running the real data pipeline, so they need no network access, no yfinance
or Reddit downloads, and no sentence-transformers. Only gymnasium, numpy,
pandas, and pytest are required.
"""
import os
import sys

# Make sure "src" is importable regardless of how pytest is invoked
# (plain `pytest`, `python -m pytest`, from a different cwd, etc.).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from src.env import RedditTradingEnv

TICKER = "TEST"
N_EMBED = 384


def _write_synthetic_data(data_dir, ticker=TICKER, n_days=10, seed=0):
    """Write a minimal prices CSV, reddit features CSV, and embeddings .npy
    file under `data_dir`, matching exactly what RedditTradingEnv._load_data
    expects to read (data/{ticker}_prices.csv,
    data/embeddings/{ticker}_reddit_features.csv,
    data/embeddings/{ticker}_embeddings.npy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="D")

    # Prices CSV: env reads Close, return, volume_zscore, volatility, rsi,
    # ma_cross (plus the date index).
    close = 100.0 + np.cumsum(rng.normal(0, 1, size=n_days))
    prices = pd.DataFrame(
        {
            "Close": close,
            "return": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
            "volume_zscore": rng.normal(0, 1, size=n_days),
            "volatility": np.abs(rng.normal(0.02, 0.01, size=n_days)),
            "rsi": rng.uniform(0, 100, size=n_days),
            "ma_cross": rng.integers(0, 2, size=n_days).astype(float),
        },
        index=dates,
    )
    prices.index.name = "Date"

    # Reddit features CSV: env reads post_count, avg_score, avg_comments.
    reddit = pd.DataFrame(
        {
            "post_count": rng.integers(0, 50, size=n_days).astype(float),
            "avg_score": rng.uniform(0, 500, size=n_days),
            "avg_comments": rng.uniform(0, 200, size=n_days),
        },
        index=dates,
    )
    reddit.index.name = "Date"

    # Embeddings .npy: shape (n_days, 384), one row per reddit-feature row.
    embeddings = rng.normal(0, 1, size=(n_days, N_EMBED)).astype(np.float32)

    prices_path = os.path.join(data_dir, f"{ticker}_prices.csv")
    embed_dir = os.path.join(data_dir, "embeddings")
    os.makedirs(embed_dir, exist_ok=True)
    reddit_path = os.path.join(embed_dir, f"{ticker}_reddit_features.csv")
    embed_path = os.path.join(embed_dir, f"{ticker}_embeddings.npy")

    prices.to_csv(prices_path)
    reddit.to_csv(reddit_path)
    np.save(embed_path, embeddings)


@pytest.fixture
def synthetic_env_factory(tmp_path, monkeypatch):
    """Returns a factory that builds a RedditTradingEnv against synthetic
    data written into tmp_path, with cwd changed to tmp_path so the env's
    relative "data/..." paths resolve there.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n_days_holder = {}

    def _factory(mode="train", use_reddit=True, n_days=10):
        _write_synthetic_data(str(data_dir), n_days=n_days)
        n_days_holder["n"] = n_days
        monkeypatch.chdir(tmp_path)
        env = RedditTradingEnv(ticker=TICKER, mode=mode, use_reddit=use_reddit)
        return env

    _factory.n_days = n_days_holder
    return _factory


def test_action_space_is_discrete_3(synthetic_env_factory):
    env = synthetic_env_factory()
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 3


def test_reset_observation_shape_without_reddit(synthetic_env_factory):
    env = synthetic_env_factory(use_reddit=False)
    obs, info = env.reset()
    assert obs.shape == (5,)
    assert env.observation_space.shape == (5,)
    assert isinstance(info, dict)


def test_reset_observation_shape_with_reddit(synthetic_env_factory):
    env = synthetic_env_factory(use_reddit=True)
    obs, info = env.reset()
    assert obs.shape == (5 + 3 + 384,)
    assert obs.shape == (392,)
    assert env.observation_space.shape == (392,)


@pytest.mark.parametrize("action", [0, 1, 2])  # hold, buy, sell
def test_step_runs_for_each_action(synthetic_env_factory, action):
    env = synthetic_env_factory(n_days=10)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (392,)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "portfolio_value" in info
    assert "cash" in info
    assert "shares_held" in info


def test_step_sequence_hold_buy_sell_runs_without_error(synthetic_env_factory):
    env = synthetic_env_factory(n_days=10, use_reddit=False)
    env.reset()
    for action in [0, 1, 0, 2, 0]:
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (5,)
        if terminated:
            break


def test_episode_terminates_at_right_point(synthetic_env_factory):
    # 10 total rows, 80/20 chronological split -> 8 rows in the train split.
    # current_step starts at 0 and increments once per step(); terminated
    # becomes True once current_step >= len(data) - 1, i.e. after exactly
    # len(data) - 1 = 7 step() calls.
    env = synthetic_env_factory(mode="train", n_days=10)
    env.reset()
    expected_train_len = int(10 * 0.8)
    assert len(env.data) == expected_train_len

    steps_taken = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(0)  # hold
        steps_taken += 1
        assert steps_taken <= expected_train_len  # safety net against infinite loop

    assert steps_taken == expected_train_len - 1


def test_episode_terminates_at_right_point_test_split(synthetic_env_factory):
    # Same 10 rows, but the "test" mode split is the remaining 20%: 2 rows.
    # That means exactly 1 step() call before termination.
    env = synthetic_env_factory(mode="test", n_days=10)
    env.reset()
    expected_test_len = 10 - int(10 * 0.8)
    assert len(env.data) == expected_test_len

    _, _, terminated, _, _ = env.step(0)
    assert terminated is True or terminated == True  # noqa: E712


def test_reset_restores_initial_portfolio_state(synthetic_env_factory):
    env = synthetic_env_factory(n_days=10)
    env.reset()
    env.step(1)  # buy
    assert env.shares_held > 0 or env.cash < env.initial_cash

    obs, info = env.reset()
    assert env.cash == env.initial_cash
    assert env.shares_held == 0
    assert env.current_step == 0
    assert obs.shape == (392,)
