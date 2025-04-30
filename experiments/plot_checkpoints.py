#!/usr/bin/env python3
"""
plot_checkpoints.py

Load CSVs from a checkpoints directory and re-create the original plots:

  a) Gini coefficient over updates
  b) Resource stock over steps
  c) Per-agent harvest rates over steps
  d) Spatial distribution of harvest positions for each agent
  e) Pairwise correlation matrix of cumulative returns
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_gini(gini_csv: pathlib.Path, out_dir: pathlib.Path):
    df = pd.read_csv(gini_csv)
    # assume the gini values are in the first non-index column
    series = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]
    plt.figure()
    plt.plot(series, marker='o')
    plt.title('Gini Coefficient of Returns')
    plt.xlabel('Update')
    plt.ylabel('Gini')
    plt.savefig(out_dir / f"{gini_csv.stem}.png", bbox_inches='tight')
    plt.close()


def plot_resource_stock(stock_csv: pathlib.Path, out_dir: pathlib.Path):
    df = pd.read_csv(stock_csv)
    series = df.iloc[:, 1] if df.shape[1] > 1 else df.iloc[:, 0]
    plt.figure()
    plt.plot(series)
    plt.title('Resource Stock Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Stock')
    plt.savefig(out_dir / f"{stock_csv.stem}.png", bbox_inches='tight')
    plt.close()


def plot_harvest_rates(harvest_csv: pathlib.Path, out_dir: pathlib.Path):
    df = pd.read_csv(harvest_csv)
    # expects columns ["step", "agent", "harvest"]
    piv = df.pivot(index='step', columns='agent', values='harvest')
    plt.figure()
    for agent in piv.columns:
        plt.plot(piv.index, piv[agent], label=str(agent))
    plt.title('Per-Agent Harvest Rates')
    plt.xlabel('Step')
    plt.ylabel('Harvest')
    plt.legend(loc='best')
    plt.savefig(out_dir / f"{harvest_csv.stem}.png", bbox_inches='tight')
    plt.close()


def plot_spatial(pos_csv: pathlib.Path, out_dir: pathlib.Path):
    df = pd.read_csv(pos_csv)
    # expects columns ["x", "y"] or first two columns
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    agent_id = pos_csv.stem.split('_')[-1]
    plt.figure()
    plt.scatter(x, y, s=5)
    plt.title(f'Harvest Positions for Agent {agent_id}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(out_dir / f"{pos_csv.stem}.png", bbox_inches='tight')
    plt.close()


def plot_pairwise_corr(corr_csv: pathlib.Path, out_dir: pathlib.Path):
    df = pd.read_csv(corr_csv, index_col=0)
    plt.figure()
    im = plt.imshow(df.values, interpolation='nearest')
    plt.colorbar(im)
    labels = df.index.astype(str)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title('Pairwise Correlation of Cumulative Returns')
    plt.savefig(out_dir / f"{corr_csv.stem}.png", bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Re-create plots from checkpoint CSVs"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=pathlib.Path,
        help="Path to your checkpoints directory"
    )
    args = parser.parse_args()
    cd = args.checkpoint_dir

    # map file patterns to plotting functions
    tasks = [
        ("*_gini_history.csv", plot_gini),
        ("*_resource_stock.csv", plot_resource_stock),
        ("*_harvest_events.csv", plot_harvest_rates),
        ("*_pairwise_corr.csv", plot_pairwise_corr),
    ]

    # 1. run through the standard CSVs
    for pattern, func in tasks:
        for path in cd.glob(pattern):
            try:
                func(path, cd)
            except Exception as e:
                print(f"[ERROR] {path.name} → {e}")

    # 2. spatial files
    for pos_csv in cd.glob("*_positions_agent_*.csv"):
        try:
            plot_spatial(pos_csv, cd)
        except Exception as e:
            print(f"[ERROR] {pos_csv.name} → {e}")

    print("All plots have been regenerated and saved in", cd)


if __name__ == "__main__":
    main()
