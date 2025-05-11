import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_logging(level: str) -> None:
    """
    Configure the root logger to emit messages at the given verbosity level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data(
    stats_dir: Path
) -> Tuple[int, List[List[float]]]:
    """
    Load training statistics from files in `stats_dir`.

    Expects:
      - reward_history.csv         : columns = one per player, rows = updates
      - zap_data.npz               : compressed archive with arrays "zap" and "death"
    
    Returns
    -------
    total_steps : int
        Total number of environment steps, taken from zap.shape[0].
    rewards_list : List[List[float]]
        A list of reward‐sequences (one list per player), each of length = # updates.
    """
    reward_path = stats_dir / "reward_history.csv"
    total_zap_path = stats_dir / "zap_matrix.csv"
    total_death_zap_path = stats_dir / "death_zap_matrix.csv"
    npz_path = stats_dir / "zap_data_through_time.npz"

    missing = [p for p in (reward_path, npz_path, total_death_zap_path, total_zap_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing file(s): {missing}")

    # Load reward history: DataFrame of shape (n_updates, n_players)
    df_rewards = pd.read_csv(reward_path)
    df_total_zap = pd.read_csv(total_zap_path)
    df_death_total_zap = pd.read_csv(total_death_zap_path)
    rewards_list = [df_rewards[col].tolist() for col in df_rewards.columns]

    # Load zap arrays
    data = np.load(npz_path, mmap_mode="r")
    zap_tt = data["zap"]              # shape [T, N, N]
    death_zap_tt = data["death"]      # shape [T, N, N]
    total_steps = death_zap_tt.shape[0]

    logging.info(
        "Loaded reward_history (%d updates for %d players) and zap_data (%d steps)",
        len(rewards_list[0]), len(rewards_list), total_steps
    )
    return total_steps, rewards_list, total_steps, zap_tt, death_zap_tt, df_total_zap, df_death_total_zap


def plot_cumulative_reward(
    rewards_list: List[List[float]],
    batch_size: int,
    save_dir: Path,
    title: str = "Cumulative Reward Through Time"
) -> None:
    """
    Plot and save cumulative reward as a function of environment steps.

    Parameters
    ----------
    rewards_list : List[List[float]]
        A list of reward sequences, one per player.
    batch_size : int
        Number of environment steps per update.
    save_dir : Path
        Directory where plots will be saved.
    title : str
        Title of the plot.
    """
    logger = logging.getLogger(__name__)
    n_updates = len(rewards_list[0])
    steps = [(i + 1) * batch_size for i in range(n_updates)]

    plt.figure(figsize=(8, 5))
    for idx, rewards in enumerate(rewards_list, start=1):
        cumulative = np.cumsum(rewards)
        plt.plot(steps, cumulative, marker="o", label=f"Player {idx}")

    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save figure
    save_path = save_dir / "cumulative_reward.png"
    plt.savefig(save_path)
    logger.info("Saved cumulative reward plot to %s", save_path)

def plot_zapping_through_time(zap_matrix, total_steps, save_dir, zap_kind='zap'):
    logger = logging.getLogger(__name__)
    n = zap_matrix.shape[1] # square matrix
    matrix = np.zeros((total_steps, n))
    for i in range(total_steps):
        for active in range(n):
            for passive in range(n):
                matrix[i][active] = zap_matrix[i][active][passive]  # TODO: This has to be divided by 2 or smth see page 7 implementation
    plt.figure(figsize=(12, 6))
    for agent_id in range(n):
        plt.plot(range(total_steps), matrix[:, agent_id], label=f'Agent {agent_id}')

    plt.xlabel('Timestep')
    plt.ylabel(zap_kind + 'Given')
    plt.title(zap_kind + 'Given per Agent Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = save_dir / (zap_kind + "zaps_given.png")
    plt.savefig(save_path)
    logger.info("Saved %s given plot to %s", zap_kind, save_path)

def plot_interaction_heatmap(
    df_matrix: pd.DataFrame,
    save_dir: Path,
    name: str,
    cmap: str = "viridis"
) -> None:
    """
    Plot and save a heatmap of agent-to-agent interactions.
    
    Parameters
    ----------
    df_matrix : pd.DataFrame
        Square DataFrame where rows=active agents and cols=passive agents.
    save_dir : Path
        Directory where the figure will be saved.
    name : str
        A short name for the metric, e.g. "zap_matrix" or "death_zap_matrix".
    cmap : str
        Any matplotlib colormap; default is "viridis".
    """
    logger = logging.getLogger(__name__)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df_matrix,
        annot=True,
        fmt="g",
        cmap=cmap,
        cbar_kws={"label": "Number of Kills"}
    )
    plt.xlabel("Passive Agent ID")
    plt.ylabel("Active Agent ID")
    plt.title(f"{name.replace('_', ' ').title()} Heatmap")
    plt.tight_layout()

    save_path = save_dir / f"{name}_heatmap.png"
    plt.savefig(save_path)
    logger.info("Saved %s heatmap to %s", name, save_path)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative reward from training statistics."
    )
    parser.add_argument(
        "stats_dir",
        type=Path,
        help="Directory containing reward_history.csv and zap_data.npz"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity level."
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.debug("Arguments: %s", args)

    # Load step count and reward streams
    total_steps, rewards_list, total_steps, zap_tt, death_zap_tt, total_zap, total_death_zap = load_data(args.stats_dir)
    plot_zapping_through_time(zap_tt, total_steps, args.stats_dir)
    plot_zapping_through_time(death_zap_tt, total_steps, args.stats_dir, 'death')
    plot_interaction_heatmap(total_zap, args.stats_dir, "zap_matrix")
    plot_interaction_heatmap(total_death_zap, args.stats_dir, "death_zap_matrix")


    # Compute batch size: how many env steps per update
    n_updates = len(rewards_list[0])
    if n_updates == 0:
        logger.error("No updates found in reward_history.csv")
        return

    batch_size = total_steps // n_updates
    if batch_size <= 0:
        logger.error(
            "Invalid batch size (total_steps=%d, updates=%d)",
            total_steps, n_updates
        )
        return

    plot_cumulative_reward(rewards_list, batch_size, args.stats_dir)


if __name__ == "__main__":
    main()
