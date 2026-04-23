"""
Generate all PDF figures for the final report.
Color scheme: navy #122048, red #C8102E, gold #A67C00, grey #9A9A9A
Run from project root: python src/generate_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── Directories ────────────────────────────────────────────────────────────────
FIGS_DIR = "results/figures"
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
C_NAVY  = "#122048"   # PPO + Reddit   (primary)
C_RED   = "#C8102E"   # DQN + Reddit   (secondary)
C_GOLD  = "#A67C00"   # Buy and Hold   (baseline)
C_GREY  = "#9A9A9A"   # PPO no Reddit  (ablation)

COLORS = {
    "PPO with Reddit" : C_NAVY,
    "PPO + Reddit"    : C_NAVY,
    "PPO no Reddit"   : C_GREY,
    "DQN with Reddit" : C_RED,
    "DQN + Reddit"    : C_RED,
    "Buy and Hold"    : C_GOLD,
    "Buy-and-Hold"    : C_GOLD,
}

MODEL_ORDER   = ["PPO with Reddit", "PPO no Reddit", "DQN with Reddit", "Buy and Hold"]
MODEL_LABELS  = ["PPO + Reddit", "PPO no Reddit", "DQN + Reddit", "Buy-and-Hold"]
TICKERS       = ["GME", "TSLA", "AAPL", "AMC"]

# ── Matplotlib base style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"   : "white",
    "axes.facecolor"     : "white",
    "axes.edgecolor"     : "#CCCCCC",
    "axes.labelcolor"    : "#333333",
    "axes.titlecolor"    : "#111111",
    "axes.titlesize"     : 13,
    "axes.titleweight"   : "bold",
    "axes.labelsize"     : 11,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.grid"          : False,
    "axes.axisbelow"     : True,
    "grid.color"         : "#EEEEEE",
    "grid.linewidth"     : 0.8,
    "xtick.color"        : "#555555",
    "ytick.color"        : "#555555",
    "xtick.bottom"       : False,
    "text.color"         : "#333333",
    "legend.framealpha"  : 0.9,
    "legend.edgecolor"   : "#CCCCCC",
    "legend.fontsize"    : 9,
    "lines.linewidth"    : 2.2,
    "font.family"        : "sans-serif",
    "font.size"          : 10,
    "savefig.dpi"        : 200,
    "savefig.bbox"       : "tight",
    "figure.dpi"         : 100,
})

def ygrid(ax):
    """Apply y-only light gridlines."""
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

def save(name):
    path = f"{FIGS_DIR}/{name}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Cumulative Return Comparison (80/20 split)
# ─────────────────────────────────────────────────────────────────────────────
def fig_return_comparison():
    df = pd.read_csv("results/summary.csv")
    x  = np.arange(len(TICKERS))
    w  = 0.19

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Cumulative Return (%) by Model and Stock — 80/20 Test Split")

    for i, (model, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        vals = [df[(df["Model"] == model) & (df["Ticker"] == t)]["Return (%)"].values for t in TICKERS]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        color = COLORS[model]
        bars = ax.bar(x + i * w - w * 1.5, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            offset = 1.5 if val >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{val:.0f}%",
                    ha="center", va=va, fontsize=7.5, color=color, fontweight="bold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(bottom=min(-70, df["Return (%)"].min() - 15))
    ygrid(ax)
    save("return_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Sharpe Ratio Comparison (80/20 split)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sharpe_comparison():
    df = pd.read_csv("results/summary.csv")
    x  = np.arange(len(TICKERS))
    w  = 0.19

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Sharpe Ratio by Model and Stock — 80/20 Test Split")

    for i, (model, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        vals = [df[(df["Model"] == model) & (df["Ticker"] == t)]["Sharpe Ratio"].values for t in TICKERS]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        color = COLORS[model]
        bars = ax.bar(x + i * w - w * 1.5, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.03,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=color, fontweight="bold")

    ax.axhline(1.0, color="#C8102E", linewidth=1.0, linestyle="--",
               alpha=0.6, label="Sharpe = 1.0 threshold")
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_ylim(bottom=min(-0.4, df["Sharpe Ratio"].min() - 0.2))
    ygrid(ax)
    save("sharpe_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Max Drawdown Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_drawdown_comparison():
    df = pd.read_csv("results/summary.csv")
    x  = np.arange(len(TICKERS))
    w  = 0.19

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Maximum Drawdown (%) by Model and Stock — Lower is Better")

    for i, (model, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        vals = [df[(df["Model"] == model) & (df["Ticker"] == t)]["Max Drawdown (%)"].values for t in TICKERS]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        color = COLORS[model]
        bars = ax.bar(x + i * w - w * 1.5, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, vals):
            if abs(val) > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - 1.0,
                        f"{val:.0f}%",
                        ha="center", va="top", fontsize=7.5,
                        color=color, fontweight="bold")

    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=12, fontweight="bold")
    ax.set_ylabel("Max Drawdown (%)")
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ygrid(ax)
    save("drawdown_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Ablation: Reddit vs No-Reddit (PPO)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    df = pd.read_csv("results/summary.csv")

    metrics = ["Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    m_labels = ["Cumulative Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Ablation Study: Effect of Reddit Embeddings (PPO)", fontsize=13, fontweight="bold")

    x = np.arange(len(TICKERS))
    w = 0.35

    for ax, metric, mlabel in zip(axes, metrics, m_labels):
        r_vals = [df[(df["Model"] == "PPO with Reddit") & (df["Ticker"] == t)][metric].values for t in TICKERS]
        n_vals = [df[(df["Model"] == "PPO no Reddit")   & (df["Ticker"] == t)][metric].values for t in TICKERS]
        r_vals = [v[0] if len(v) > 0 else 0 for v in r_vals]
        n_vals = [v[0] if len(v) > 0 else 0 for v in n_vals]

        b1 = ax.bar(x - w/2, r_vals, w, label="PPO + Reddit",   color=C_NAVY, alpha=0.88, edgecolor="white")
        b2 = ax.bar(x + w/2, n_vals, w, label="PPO no Reddit",  color=C_GREY, alpha=0.88, edgecolor="white")

        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(TICKERS, fontsize=11)
        ax.legend(fontsize=8)
        ygrid(ax)

    plt.tight_layout()
    save("ablation_study")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Walk-Forward: Sharpe Ratio by Window
# ─────────────────────────────────────────────────────────────────────────────
def fig_walk_forward_sharpe():
    df = pd.read_csv("results/walk_forward/walk_forward_results.csv")
    windows = [("Window_1_2022", "2022"), ("Window_2_2023", "2023"), ("Window_3_2024", "2024")]
    wf_models = ["PPO + Reddit", "PPO no Reddit", "DQN + Reddit", "Buy-and-Hold"]
    wf_colors = [C_NAVY, C_GREY, C_RED, C_GOLD]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Walk-Forward Validation: Sharpe Ratio by Window and Stock", fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        x   = np.arange(len(windows))
        w   = 0.20
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([yr for _, yr in windows], fontsize=9)
        ax.set_xlabel("Test Year")
        if ticker == TICKERS[0]:
            ax.set_ylabel("Sharpe Ratio")

        for i, (model, color) in enumerate(zip(wf_models, wf_colors)):
            vals = []
            for win, _ in windows:
                row = df[(df["window"] == win) & (df["ticker"] == ticker) & (df["model"] == model)]["sharpe_ratio"]
                vals.append(row.values[0] if len(row) > 0 else 0)
            ax.bar(x + (i - 1.5) * w, vals, w, label=model, color=color,
                   alpha=0.88, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ax.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.5)
        ygrid(ax)

    handles = [mpatches.Patch(color=c, label=m) for m, c in zip(wf_models, wf_colors)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save("walk_forward_sharpe")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Walk-Forward: Cumulative Return by Window
# ─────────────────────────────────────────────────────────────────────────────
def fig_walk_forward_return():
    df = pd.read_csv("results/walk_forward/walk_forward_results.csv")
    windows = [("Window_1_2022", "2022"), ("Window_2_2023", "2023"), ("Window_3_2024", "2024")]
    wf_models = ["PPO + Reddit", "PPO no Reddit", "DQN + Reddit", "Buy-and-Hold"]
    wf_colors = [C_NAVY, C_GREY, C_RED, C_GOLD]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Walk-Forward Validation: Cumulative Return (%) by Window and Stock", fontsize=13, fontweight="bold")

    for ax, ticker in zip(axes, TICKERS):
        x   = np.arange(len(windows))
        w   = 0.20
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([yr for _, yr in windows], fontsize=9)
        ax.set_xlabel("Test Year")
        if ticker == TICKERS[0]:
            ax.set_ylabel("Cumulative Return (%)")

        for i, (model, color) in enumerate(zip(wf_models, wf_colors)):
            vals = []
            for win, _ in windows:
                row = df[(df["window"] == win) & (df["ticker"] == ticker) & (df["model"] == model)]["cumulative_return"]
                vals.append(row.values[0] if len(row) > 0 else 0)
            ax.bar(x + (i - 1.5) * w, vals, w, label=model, color=color,
                   alpha=0.88, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="#AAAAAA", linewidth=0.8)
        ygrid(ax)

    handles = [mpatches.Patch(color=c, label=m) for m, c in zip(wf_models, wf_colors)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save("walk_forward_return")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Walk-Forward Summary: Mean Sharpe Heat-style grouped bars
# ─────────────────────────────────────────────────────────────────────────────
def fig_walk_forward_summary():
    df = pd.read_csv("results/walk_forward/walk_forward_summary.csv")
    wf_models = ["PPO + Reddit", "PPO no Reddit", "DQN + Reddit", "Buy-and-Hold"]
    wf_colors = [C_NAVY, C_GREY, C_RED, C_GOLD]

    x = np.arange(len(TICKERS))
    w = 0.19

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Walk-Forward Summary: Mean ± Std Sharpe Ratio Across Three Windows",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.set_title("Mean Sharpe Ratio (3-Year Average)")
    for i, (model, color) in enumerate(zip(wf_models, wf_colors)):
        sub = df[df["Model"] == model]
        means = [sub[sub["Ticker"] == t]["Mean Sharpe"].values for t in TICKERS]
        stds  = [sub[sub["Ticker"] == t]["Std Sharpe"].values  for t in TICKERS]
        means = [v[0] if len(v) > 0 else 0 for v in means]
        stds  = [v[0] if len(v) > 0 else 0 for v in stds]
        ax.bar(x + (i - 1.5) * w, means, w, yerr=stds, label=model, color=color,
               alpha=0.88, edgecolor="white", linewidth=0.5,
               error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#555555"))
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Sharpe Ratio")
    ax.legend(fontsize=8, ncol=1)
    ygrid(ax)

    ax = axes[1]
    ax.set_title("Sharpe Ratio Std Dev (Consistency)")
    for i, (model, color) in enumerate(zip(wf_models, wf_colors)):
        sub = df[df["Model"] == model]
        stds = [sub[sub["Ticker"] == t]["Std Sharpe"].values for t in TICKERS]
        stds = [v[0] if len(v) > 0 else 0 for v in stds]
        ax.bar(x + (i - 1.5) * w, stds, w, label=model, color=color,
               alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=11, fontweight="bold")
    ax.set_ylabel("Std Dev of Sharpe Ratio")
    ax.legend(fontsize=8, ncol=1)
    ygrid(ax)

    plt.tight_layout()
    save("walk_forward_summary")


# ─────────────────────────────────────────────────────────────
# Figure 8 — Learning Curves (actual eval logs)
# ─────────────────────────────────────────────────────────────
def fig_learning_curves():
    models = [
        ("ppo_reddit",    C_NAVY, "PPO + Reddit"),
        ("ppo_no_reddit", C_GREY, "PPO no Reddit"),
        ("dqn_reddit",    C_RED,  "DQN + Reddit"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Training Learning Curves: Mean Evaluation Reward over 500K Timesteps",
                 fontsize=12, fontweight="bold")
    axes = axes.flatten()

    def smooth(y, w=9):
        return pd.Series(y).rolling(w, min_periods=1).mean().values

    for ax, ticker in zip(axes, TICKERS):
        ax.set_title(ticker, fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Timesteps", fontsize=9)
        ax.set_ylabel("Mean Eval Reward", fontsize=9)
        ax.axhline(0, color="#CCCCCC", linewidth=0.8)
        ygrid(ax)

        for label, color, name in models:
            path = f"logs/{label}/{ticker}/evaluations.npz"
            if not os.path.exists(path):
                continue
            d = np.load(path)
            ts = d["timesteps"]
            rewards = d["results"].mean(axis=1)
            sm = smooth(rewards, 9)
            ax.plot(ts, sm, color=color, label=name, linewidth=1.8)
            ax.fill_between(ts, sm - rewards.std() * 0.3,
                            sm + rewards.std() * 0.3,
                            alpha=0.12, color=color)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 500000)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))

    plt.tight_layout()
    save("learning_curves")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating report figures...")
    fig_return_comparison()
    fig_sharpe_comparison()
    fig_drawdown_comparison()
    fig_ablation()
    fig_walk_forward_sharpe()
    fig_walk_forward_return()
    fig_walk_forward_summary()
    fig_learning_curves()
    print("Done.")

