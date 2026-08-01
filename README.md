# Reddit-Augmented RL Stock Trader

CS 5180 (Reinforcement Learning) final project. Trains PPO and DQN agents to
trade GME, TSLA, AAPL, and AMC, and tests whether adding Reddit post
embeddings (from r/WallStreetBets and related subreddits) to the agent's
observations improves trading performance over price-only agents.

Full writeup: [`report_final.pdf`](report_final.pdf) / [`report_final.tex`](report_final.tex).

## Summary of findings

- **TSLA**: Reddit helps clearly — DQN + Reddit reaches Sharpe 1.693 vs. 1.215 for buy-and-hold.
- **AAPL**: Reddit helps modestly; price is mostly driven by fundamentals, not sentiment.
- **AMC**: All RL agents correctly sit out the 57% decline over the test period.
- **GME**: Reddit looks helpful on the fixed 80/20 split, but walk-forward validation
  (2022–2024) shows it actually *hurts* performance — the community stayed bullish long
  after the stock stopped responding to sentiment, so a model trained on the 2021 squeeze
  learns the wrong causal relationship.

## Project structure

```
src/
  data_pipeline.py    # Downloads OHLCV price data + technical indicators (yfinance),
                       # and builds daily Reddit features (post count, avg score/comments,
                       # mean Sentence-BERT embedding) from raw WSB post data.
  env.py               # Gymnasium trading environment (RedditTradingEnv). 3 actions
                       # (hold/buy/sell), optional 384-dim Reddit embedding in the obs.
  train.py             # Trains PPO/DQN, with and without Reddit, for all four tickers.
  evaluate.py          # Evaluates trained agents + buy-and-hold on the test split.
  walk_forward.py       # Expanding-window walk-forward validation (test years 2022-2024).
  generate_figures.py  # Renders the report's figures from results/.
results/
  figures/             # Generated plots (return, Sharpe, drawdown, walk-forward, etc.)
  summary.csv, walk_forward/  # Raw metrics backing the report tables.
report_final.tex        # AAAI-style paper (source for report_final.pdf).
references.bib
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install yfinance pandas numpy ta sentence-transformers gymnasium stable-baselines3 matplotlib
```

You'll also need a raw WSB post dump at `data/wsb_raw/r_wallstreetbets_posts.csv`
(columns: `title`, `score`, `num_comments`, `created_utc`) — e.g. a Kaggle
r/wallstreetbets posts export. `data/`, `models/`, and `logs/` are gitignored
since they're large/regenerable.

## Running the pipeline

```bash
# 1. Build price + Reddit feature data into data/
python src/data_pipeline.py

# 2. Train all PPO/DQN variants (with and without Reddit) for GME, TSLA, AAPL, AMC
python src/train.py

# 3. Evaluate trained agents + buy-and-hold baseline on the 80/20 test split
python src/evaluate.py

# 4. Walk-forward validation across 2022 / 2023 / 2024 test years
python src/walk_forward.py

# 5. Regenerate report figures from results/
python src/generate_figures.py
```

## Environment details

- **Observation**: 5 price features (return, volume z-score, 5-day volatility, RSI,
  SMA10/SMA50 crossover) plus, when Reddit is enabled, 3 scalar Reddit features
  (post count, avg score, avg comments) and a 384-dim mean-pooled Sentence-BERT
  (`all-MiniLM-L6-v2`) embedding of that day's post titles — 392 dims total.
- **Actions**: hold, buy all, or sell all, starting from a $10,000 cash account.
- **Reward**: daily portfolio return, minus a volatility penalty and a 0.1% trade cost.
- **Metrics**: cumulative return, Sharpe ratio, and max drawdown.
