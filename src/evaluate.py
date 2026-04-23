import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from src.env import RedditTradingEnv

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS = ["GME", "TSLA", "AAPL", "AMC"]   
MODELS_DIR = "models"
RESULTS_DIR = "results"
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_agent(model, env, model_name: str, ticker: str) -> dict:
    """Run a trained agent on the test environment and collect results."""
    obs, _ = env.reset()
    done    = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    portfolio_values = env.portfolio_history
    trades          = env.trade_history

    returns = pd.Series(portfolio_values).pct_change().dropna()

    cumulative_return = (portfolio_values[-1] - env.initial_cash) / env.initial_cash * 100
    sharpe       = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    peak         = pd.Series(portfolio_values).cummax()
    max_drawdown = ((pd.Series(portfolio_values) - peak) / peak).min() * 100
    final_value  = portfolio_values[-1]

    print(f"\n{'='*50}")
    print(f"{model_name} on {ticker} (Test Period)")
    print(f"{'='*50}")
    print(f"  Final Portfolio Value : ${final_value:,.2f}")
    print(f"  Cumulative Return     : {cumulative_return:.2f}%")
    print(f"  Sharpe Ratio          : {sharpe:.3f}")
    print(f"  Max Drawdown          : {max_drawdown:.2f}%")
    print(f"  Total Trades          : {len(trades)}")

    return {
        "model"             : model_name,
        "ticker"            : ticker,
        "final_value"       : final_value,
        "cumulative_return" : cumulative_return,
        "sharpe_ratio"      : sharpe,
        "max_drawdown"      : max_drawdown,
        "total_trades"      : len(trades),
        "portfolio_history" : portfolio_values,
        "trade_history"     : trades
    }


def buy_and_hold(ticker: str) -> dict:
    """Calculate buy and hold return for the test period."""
    env    = RedditTradingEnv(ticker, mode="test", use_reddit=False)
    prices = env.data["Close"].values

    shares            = int(env.initial_cash // prices[0])
    leftover_cash     = env.initial_cash - shares * prices[0]
    final_value       = shares * prices[-1] + leftover_cash
    cumulative_return = (final_value - env.initial_cash) / env.initial_cash * 100
    portfolio_history = [shares * p + leftover_cash for p in prices]
    returns           = pd.Series(portfolio_history).pct_change().dropna()
    sharpe            = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    peak         = pd.Series(portfolio_history).cummax()
    drawdown     = (pd.Series(portfolio_history) - peak) / peak
    max_drawdown = drawdown.min() * 100

    print(f"\n{'='*50}")
    print(f"Buy and Hold on {ticker} (Test Period)")
    print(f"{'='*50}")
    print(f"  Final Portfolio Value : ${final_value:,.2f}")
    print(f"  Cumulative Return     : {cumulative_return:.2f}%")
    print(f"  Sharpe Ratio          : {sharpe:.3f}")
    print(f"  Max Drawdown          : {max_drawdown:.2f}%")
    print(f"  Total Trades          : 1 (buy and hold)")

    return {
        "model"             : "Buy and Hold",
        "ticker"            : ticker,
        "final_value"       : final_value,
        "cumulative_return" : cumulative_return,
        "sharpe_ratio"      : sharpe,
        "max_drawdown"      : max_drawdown,
        "total_trades"      : 1,
        "portfolio_history" : portfolio_history,
        "trade_history"     : []
    }


def run_evaluation():
    """Evaluate all models on all tickers and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for ticker in TICKERS:
        print(f"\nEvaluating on {ticker}...")

        # PPO with Reddit
        test_env_ppo = RedditTradingEnv(ticker, mode="test", use_reddit=True)
        ppo_path = f"{MODELS_DIR}/ppo_reddit/{ticker}_best/best_model.zip"
        if os.path.exists(ppo_path):
            ppo_model  = PPO.load(ppo_path, env=test_env_ppo)
            ppo_result = evaluate_agent(ppo_model, test_env_ppo, "PPO with Reddit", ticker)
            all_results.append(ppo_result)

        # PPO without Reddit (ablation)
        test_env_ppo_no = RedditTradingEnv(ticker, mode="test", use_reddit=False)
        ppo_no_path = f"{MODELS_DIR}/ppo_no_reddit/{ticker}_best/best_model.zip"
        if os.path.exists(ppo_no_path):
            ppo_no_model  = PPO.load(ppo_no_path, env=test_env_ppo_no)
            ppo_no_result = evaluate_agent(ppo_no_model, test_env_ppo_no, "PPO no Reddit", ticker)
            all_results.append(ppo_no_result)

        # DQN with Reddit
        test_env_dqn = RedditTradingEnv(ticker, mode="test", use_reddit=True)
        dqn_path = f"{MODELS_DIR}/dqn_reddit/{ticker}_best/best_model.zip"
        if os.path.exists(dqn_path):
            dqn_model  = DQN.load(dqn_path, env=test_env_dqn)
            dqn_result = evaluate_agent(dqn_model, test_env_dqn, "DQN with Reddit", ticker)
            all_results.append(dqn_result)

        # Buy and Hold
        bah_result = buy_and_hold(ticker)
        all_results.append(bah_result)

    # Save summary table
    summary = pd.DataFrame([{
        "Model"            : r["model"],
        "Ticker"           : r["ticker"],
        "Final Value ($)"  : round(r["final_value"], 2),
        "Return (%)"       : round(r["cumulative_return"], 2),
        "Sharpe Ratio"     : round(r["sharpe_ratio"], 3),
        "Max Drawdown (%)" : round(r["max_drawdown"], 2),
        "Total Trades"     : r["total_trades"]
    } for r in all_results])

    summary_path = f"{RESULTS_DIR}/summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*50}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    print(summary.to_string(index=False))
    print(f"\nSaved to {summary_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()