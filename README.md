# Reddit-Augmented RL Stock Trader
Trains PPO and DQN agents to
trade GME, TSLA, AAPL, and AMC, and tests whether adding Reddit post
embeddings (from r/WallStreetBets and related subreddits) to the agent's
observations improves trading performance over price-only agents.

**Interactive dashboard**: a standalone backtest explorer built from `results/`
lives in [`docs/`](docs/) — pick a ticker/model and compare Sharpe, return, and
drawdown against buy-and-hold, see the walk-forward breakdown by year, and read
the report's key findings per ticker. See
[Deploying the dashboard](#deploying-the-dashboard) to put it online in under a
minute.

## Summary of findings

- **TSLA**: Reddit helps clearly — DQN + Reddit reaches Sharpe 1.693 vs. 1.215 for buy-and-hold.
- **AAPL**: Reddit helps modestly; price is mostly driven by fundamentals, not sentiment.
- **AMC**: All RL agents correctly sit out the 57% decline over the test period.
- **GME**: Reddit looks helpful on the fixed 80/20 split, but walk-forward validation
  (2022–2024) shows it actually *hurts* performance — the community stayed bullish long
  after the stock stopped responding to sentiment, so a model trained on the 2021 squeeze
  learns the wrong causal relationship.

### Limitations

- **GME degenerate policy**: on the fixed 80/20 split, `results/summary.csv` shows PPO
  with Reddit and DQN with Reddit converging to the *exact same* final portfolio value
  (\$23,687.41) with only 1 total trade each. Both algorithms independently learned the
  same "buy once and hold" policy rather than distinct trading strategies. This isn't a
  training bug — GME's test-period price path is close to monotonically increasing, and
  the environment's 3-action space (hold / buy-all / sell-all) makes "buy on day one and
  never sell" the reward-maximizing policy regardless of algorithm or whether Reddit
  features are present. It's a reminder that a single monotonic test run can't
  distinguish "the agent learned a good policy" from "the agent learned the only
  policy that could win here," and that Sharpe/return numbers on GME should be read
  with that caveat rather than as evidence of meaningfully different PPO vs. DQN or
  Reddit vs. no-Reddit behavior.

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
docs/
  index.html            # Standalone interactive dashboard (no build step, no deps).
  report_final.pdf       # Copy of the paper the dashboard links out to.
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For running the test suite, also install `requirements-dev.txt`
(`pip install -r requirements-dev.txt`).

You'll also need a raw WSB post dump at `data/wsb_raw/r_wallstreetbets_posts.csv` with
exactly these columns:

| column         | type                        | meaning                                |
|----------------|-----------------------------|-----------------------------------------|
| `title`        | string                      | post title text                        |
| `score`        | int                         | post score (upvotes − downvotes)       |
| `num_comments` | int                         | number of comments on the post         |
| `created_utc`  | int (unix seconds)          | post creation time, UTC epoch seconds  |

This is the schema produced by PRAW's `Submission` object (`.title`, `.score`,
`.num_comments`, `.created_utc`), so any PRAW-based WSB scrape will line up
directly. Kaggle also hosts several r/wallstreetbets posts exports (e.g.
[`gpreda/reddit-wallstreetsbets-posts`](https://www.kaggle.com/datasets/gpreda/reddit-wallstreetsbets-posts)
and the well-known Jan 2021 GME-squeeze `reddit_wsb.csv` dump) — but be aware some
of these use slightly different column names (`comms_num` instead of `num_comments`,
`created` instead of `created_utc`), so check the header and rename columns to match
the schema above before pointing `data_pipeline.py` at them.

`data/`, `models/`, and `logs/` are gitignored since they're large/regenerable.

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

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/
```

`tests/test_env.py` covers `RedditTradingEnv` against small synthetic CSV/`.npy`
fixtures (no network access, no real data pipeline needed). CI runs the same suite
on every push/PR via `.github/workflows/ci.yml`.

## Deploying the dashboard

`docs/index.html` is a self-contained static page (no build step, no npm, no
server) — it just needs to be served alongside `docs/report_final.pdf`.

**Vercel (recommended — free, custom domain, instant HTTPS):**

```bash
npm i -g vercel        # one-time
cd docs
vercel --prod
```

Or without the CLI: on [vercel.com](https://vercel.com), "Add New Project" →
import this repo → set **Root Directory** to `docs` → deploy. Every push to
`main` then redeploys automatically.

**GitHub Pages (free, zero extra accounts, already tied to this repo):**
Settings → Pages → Deploy from a branch → `main` → folder `/docs`.

Either way the dashboard is fully static: it reads no external APIs and only
needs `index.html` and `report_final.pdf` in the same directory.

## License

[MIT](LICENSE) © 2026 Ananya Parag Joshi.
