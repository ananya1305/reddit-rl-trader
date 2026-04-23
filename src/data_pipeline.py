import yfinance as yf
import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["GME", "TSLA", "AAPL", "AMC"]
START     = "2019-01-01"
END       = "2024-12-31"
DATA_DIR  = "data"
# ─────────────────────────────────────────────────────────────────────────────


def fetch_stock_data(ticker: str, start: str = START, end: str = END) -> pd.DataFrame:
    """Download OHLCV data and compute technical indicators."""
    print(f"Fetching stock data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    df.columns = df.columns.droplevel(1) if isinstance(df.columns, pd.MultiIndex) else df.columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)

    # Daily return
    df["return"]     = df["Close"].pct_change()

    # Volume z-score (rolling 20-day)
    df["volume_zscore"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean())
        / df["Volume"].rolling(20).std()
    )

    # Volatility (5-day rolling std of returns)
    df["volatility"] = df["return"].rolling(5).std()

    # RSI (14-day)
    df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # Moving average crossover signal: 1 if SMA10 > SMA50 else 0
    sma10 = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    sma50 = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["ma_cross"] = (sma10 > sma50).astype(int)

    df.dropna(inplace=True)
    return df


def save_stock_data():
    """Fetch and save stock data for all tickers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for ticker in TICKERS:
        df = fetch_stock_data(ticker)
        path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
        df.to_csv(path)
        print(f"  Saved {len(df)} rows → {path}")

# ── Reddit Config ─────────────────────────────────────────────────────────────
WSB_PATH      = "data/wsb_raw/r_wallstreetbets_posts.csv"
EMBEDDINGS_DIR = "data/embeddings"
MODEL_NAME    = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────


def extract_ticker_mentions(title: str, tickers: list) -> list:
    """Return which tickers from our list are mentioned in a post title."""
    title_upper = title.upper()
    return [t for t in tickers if re.search(rf'\b{t}\b', title_upper)]


def load_wsb_posts(tickers: list = TICKERS) -> pd.DataFrame:
    """Load WSB posts, filter to only those mentioning our tickers."""
    print("Loading WSB posts...")
    df = pd.read_csv(WSB_PATH, low_memory=False)

    # Convert UTC timestamp to date
    df["date"] = pd.to_datetime(df["created_utc"], unit="s").dt.date

    # Keep only relevant columns
    df = df[["title", "score", "num_comments", "date"]].copy()
    df.dropna(subset=["title", "date"], inplace=True)

    df["tickers_mentioned"] = df["title"].apply(
        lambda x: extract_ticker_mentions(str(x), tickers)
    )
    df = df[df["tickers_mentioned"].map(len) > 0].copy()

    print(f"  Found {len(df)} posts mentioning our tickers")
    return df


def build_daily_reddit_features(ticker: str, posts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily Reddit posts for a ticker into scalar features and a mean embedding."""
    print(f"Building Reddit features for {ticker}...")

    ticker_posts = posts_df[
        posts_df["tickers_mentioned"].apply(lambda x: ticker in x)
    ].copy()

    if len(ticker_posts) == 0:
        print(f"  No posts found for {ticker}")
        return pd.DataFrame()

    print(f"  {len(ticker_posts)} posts found for {ticker}")
    model = SentenceTransformer(MODEL_NAME)

    # encode all titles in one batched pass
    print(f"  Encoding titles...")
    embeddings = model.encode(
        ticker_posts["title"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )
    ticker_posts = ticker_posts.copy()
    ticker_posts["embedding"] = list(embeddings)

    daily_records = []
    for date, group in ticker_posts.groupby("date"):
        mean_embedding = np.mean(np.stack(group["embedding"].values), axis=0)

        daily_records.append({
            "date": date,
            "post_count": len(group),
            "avg_score": group["score"].mean(),
            "avg_comments": group["num_comments"].mean(),
            "embedding": mean_embedding
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df.set_index("date", inplace=True)
    daily_df.sort_index(inplace=True)

    return daily_df


def save_reddit_features():
    """Process and save Reddit features for all tickers."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    posts_df = load_wsb_posts()

    for ticker in TICKERS:
        daily_df = build_daily_reddit_features(ticker, posts_df)
        if len(daily_df) == 0:
            continue

        # embeddings saved separately because they're large
        embeddings_array = np.stack(daily_df["embedding"].values)
        np.save(f"{EMBEDDINGS_DIR}/{ticker}_embeddings.npy", embeddings_array)

        daily_df_save = daily_df.drop(columns=["embedding"])
        daily_df_save.to_csv(f"{EMBEDDINGS_DIR}/{ticker}_reddit_features.csv")

        print(f"  Saved {ticker} → {len(daily_df)} days of Reddit features")


if __name__ == "__main__":
    save_stock_data()
    save_reddit_features()