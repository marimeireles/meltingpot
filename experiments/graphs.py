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


def load_data(stats_dir: Path) -> Tuple[int, List[List[float]]]:
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

    missing = [
        p
        for p in (reward_path, npz_path, total_death_zap_path, total_zap_path)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing file(s): {missing}")

    # Load reward history: DataFrame of shape (n_updates, n_players)
    df_rewards = pd.read_csv(reward_path, header=None)
    df_total_zap = pd.read_csv(total_zap_path, header=None)
    df_death_total_zap = pd.read_csv(total_death_zap_path, header=None)
    rewards_list = [df_rewards[col].tolist() for col in df_rewards.columns]

    # Load zap arrays
    data = np.load(npz_path, mmap_mode="r")
    zap_tt = data["zap"]  # shape [T, N, N]
    death_zap_tt = data["death"]  # shape [T, N, N]
    total_steps = death_zap_tt.shape[0]

    logging.info(
        "Loaded reward_history (%d updates for %d players) and zap_data (%d steps)",
        len(rewards_list[0]),
        len(rewards_list),
        total_steps,
    )
    return (
        total_steps,
        rewards_list,
        total_steps,
        zap_tt,
        death_zap_tt,
        df_total_zap,
        df_death_total_zap,
    )


def plot_cumulative_reward_per_step(
    rewards_list: List[List[float]],
    save_dir: Path,
    title: str = "Cumulative Reward Through Time (Per Step)",
) -> None:
    """
    Plot and save cumulative reward as a function of environment steps,
    at per-step granularity.

    Parameters
    ----------
    rewards_list : List[List[float]]
        A list of cumulative‐reward sequences, one list per agent,
        each of length = total environment steps.
    save_dir : Path
        Directory where the figure will be saved.
    title : str
        Title of the plot.
    """
    logger = logging.getLogger(__name__)

    # Ensure we have at least one agent
    if not rewards_list or not rewards_list[0]:
        logger.error("No reward data to plot.")
        return

    # X axis: 1, 2, 3, ..., total_steps
    total_steps = len(rewards_list[0])
    steps = list(range(total_steps))

    plt.figure(figsize=(8, 5))
    for idx, cumulative in enumerate(rewards_list, start=1):
        if len(cumulative) != total_steps:
            logger.warning(
                "Agent %d has %d steps of data (expected %d); skipping.",
                idx,
                len(cumulative),
                total_steps,
            )
            continue
        plt.plot(
            steps,
            cumulative,
            marker=".",
            markevery=max(1, total_steps // 20),
            label=f"Agent {idx}",
        )

    plt.xlabel("Environment Step")
    plt.ylabel("Cumulative Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save figure
    save_path = save_dir / "cumulative_reward_per_step.png"
    plt.savefig(save_path)
    logger.info("Saved per-step cumulative reward plot to %s", save_path)


def plot_zapping_through_time(
    zap_tt: np.ndarray, save_dir: Path, kind: str = "zap", cumulative: bool = True
) -> None:
    """
    Plot per-agent 'kind' given over every timestep.

    Parameters
    ----------
    zap_tt : np.ndarray
        Either shape (T, N, N) or (U, S, N, N).
    save_dir : Path
    kind : str
        "zap" or "death"
    cumulative : bool
        If True, plot the cumulative total up to each time;
        otherwise plot per-step counts.
    """
    logger = logging.getLogger(__name__)

    # 1) If 4-D (updates × steps × agents × agents), flatten into time
    if zap_tt.ndim == 4:
        U, S, N, _ = zap_tt.shape
        T = U * S
        zap_flat = zap_tt.reshape(T, N, N)
    elif zap_tt.ndim == 3:
        zap_flat = zap_tt
        T, N, _ = zap_flat.shape
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {zap_tt.shape}")

    # 2) Sum out the passive-agent axis → shape (T, N)
    per_step = zap_flat.sum(axis=2)

    # 3) Optionally make it cumulative over time
    if cumulative:
        data = np.cumsum(per_step, axis=0)
        ylabel = f"Cumulative {kind.title()}s Given"
        title = f"Cumulative {kind.title()}s Given per Agent Over Time"
    else:
        data = per_step
        ylabel = f"{kind.title()}s Given per Step"
        title = f"{kind.title()}s Given per Agent per Timestep"

    # 4) Plot
    plt.figure(figsize=(12, 6))
    for agent_id in range(N):
        plt.plot(np.arange(data.shape[0]), data[:, agent_id], label=f"Agent {agent_id}")
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 5) Save
    filename = f"{kind}_{'cum' if cumulative else 'step'}_given.png"
    save_path = save_dir / filename
    plt.savefig(save_path)
    logger.info("Saved %s plot to %s", kind, save_path)


def plot_average_reward_through_time(
    stats_dir: Path,
    save_dir,
    title: str = "Average Reward Through Time",
    xlabel: str = "Update",
    ylabel: str = "Average Reward",
    log_level: str = "INFO",
) -> None:
    """
    Load reward_history.csv from stats_dir, compute the average reward
    across all agents at each update, and plot it.

    Parameters
    ----------
    stats_dir : Path
        Directory containing `reward_history.csv` (no header, one column per agent).
    save_dir : Optional[Path]
        Directory where to save the figure. If None, defaults to stats_dir.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis (default: "Update").
    ylabel : str
        Label for the y-axis (default: "Average Reward").
    log_level : str
        Logging verbosity level.
    """
    # set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    reward_path = stats_dir / "reward_history.csv"
    if not reward_path.exists():
        logger.error("Could not find reward_history.csv in %s", stats_dir)
        return

    # load as DataFrame, one column per agent
    df = pd.read_csv(reward_path, header=None)
    # compute row‐wise mean
    avg_rewards = df.mean(axis=1)

    # prepare x axis (1-based update count)
    updates = avg_rewards.index.to_numpy() + 1

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(updates, avg_rewards, marker=".", markevery=max(1, len(updates) // 20))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    # decide where to save
    out_dir = save_dir or stats_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "average_reward_through_time.png"
    plt.savefig(out_path)
    logger.info("Saved average‐reward plot to %s", out_path)


def plot_interaction_heatmap(
    df_matrix: pd.DataFrame, save_dir: Path, name: str, cmap: str = "viridis"
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
    plt.figure()
    sns.heatmap(
        df_matrix, annot=True, fmt="g", cmap=cmap, cbar_kws={"label": "Number of Kills"}
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
        help="Directory containing reward_history.csv and zap_data.npz",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity level.",
    )
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.debug("Arguments: %s", args)

    # read hyperparameters from CSV
    hyper_path = args.stats_dir / "hyperparameters.csv"
    df_hparams = pd.read_csv(hyper_path)
    try:
        n_updates = int(df_hparams["TOTAL_TRAINING_UPDATES"].iloc[0])
        batch_size = int(df_hparams["BATCH_SIZE"].iloc[0])
    except (KeyError, IndexError, ValueError) as e:
        logger.error(
            "Failed to read TOTAL_TRAINING_UPDATES or BATCH_SIZE from %s: %s",
            hyper_path,
            e,
        )
        return

    (
        total_steps,
        rewards_list,
        total_steps,
        zap_tt,
        death_zap_tt,
        total_zap,
        total_death_zap,
    ) = load_data(args.stats_dir)
    plot_zapping_through_time(zap_tt, args.stats_dir, kind="zap", cumulative=True)
    plot_zapping_through_time(
        death_zap_tt, args.stats_dir, kind="death", cumulative=True
    )
    plot_interaction_heatmap(total_zap, args.stats_dir, "zap_matrix")
    plot_interaction_heatmap(total_death_zap, args.stats_dir, "death_zap_matrix")

    plot_cumulative_reward_per_step(rewards_list, args.stats_dir)

    plot_average_reward_through_time(
        stats_dir=args.stats_dir,
        save_dir=args.stats_dir,
        title="Mean Reward per Update",
        xlabel="PPO Update",
        ylabel="Mean Cumulative Reward",
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
