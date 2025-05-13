import argparse
import datetime
import pathlib

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import serialization
from flax.core import freeze
from jax import random, value_and_grad
from jax.lib import xla_bridge
from ml_collections import config_dict

import meltingpot.human_players.level_playing_utils as level_playing_utils
import wandb
from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.utils.substrates import builder

# ── Hyperparameters ─────────────────────────────────────────────────────────────
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 5e-5  # 3e-4
PPO_CLIP_EPSILON = 0.2  # 0.2
BATCH_SIZE = 256  # 128
PPO_EPOCHS = 5
TOTAL_TRAINING_UPDATES = 50
KL_THRESHOLD = 0.001089358402136939  # 1e-2
# ────────────────────────────────────────────────────────────────────────────────

# Utility Functions
# -------------------------------------------------------------------

import logging
import sys


def parse_args():
    """Parse command line arguments for training or human play modes.
    
    Returns:
        argparse.Namespace: Parsed command line arguments including:
            - mode: 'train' for PPO training or 'human' for interactive play
            - level-name: Which MeltingPot level to load
            - fps: Target framerate for rendering
            - seed: Random seed
            - log-level: Logging verbosity
            - num-workers: Number of parallel data collection workers
            - Various hyperparameters that can override defaults
    """
    p = argparse.ArgumentParser(
        description="Train PPO on commons_harvest or play by hand"
    )
    p.add_argument(
        "--mode",
        choices=("train", "human"),
        default="train",
        help="train: run PPO self‐play; human: launch interactive player",
    )
    # TODO: make the code generic for both commons harvest and clean up
    p.add_argument(
        "--level-name",
        type=str,
        default="commons_harvest__open",
        choices={"commons_harvest__open"},
        help="which MeltingPot level to load",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=15,
        help="target framerate for rendering",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for JAX",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root logger verbosity",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel actor processes for data collection",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="skip wandb.init and use hardcoded defaults",
    )
    # Hyperparameter arguments that can override defaults
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--ppo_clip_epsilon", type=float, default=PPO_CLIP_EPSILON)
    p.add_argument("--ppo_epochs", type=int, default=PPO_EPOCHS)
    p.add_argument("--kl_threshold", type=float, default=KL_THRESHOLD)
    p.add_argument("--discount_factor", type=float, default=DISCOUNT_FACTOR)
    p.add_argument("--total_training_updates", type=int, default=TOTAL_TRAINING_UPDATES)
    return p.parse_args()


def configure_logging(level: str) -> None:
    """Configure root and library loggers for consistent output format.
    
    Args:
        level: Logging level as string ('DEBUG', 'INFO', etc.)
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def log_jax_devices():
    """Log available JAX backend and devices for debugging.
    
    Prints the XLA backend platform and details of each available device.
    """
    logger = logging.getLogger(__name__)
    backend = xla_bridge.get_backend()
    logger.info("JAX XLA backend platform: %s", backend.platform)
    devices = jax.devices()
    for dev in devices:
        logger.info(
            "  Device: %s (id=%d, process_index=%d)",
            dev.device_kind,
            dev.id,
            dev.process_index,
        )


# Define convolutional actor-critic network
# -------------------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    action_dimension: int

    @nn.compact
    def __call__(self, observations):
        x = observations / 255.0
        x = nn.Conv(32, (8, 8), (4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_dimension)(x)
        state_value = nn.Dense(1)(x)
        return logits, jnp.squeeze(state_value, axis=-1)


# Data-collection
# -------------------------------------------------------------------
# TODO: ACTION_SET should be properly global and be a default arg like steps_per_agent
def get_multi_rewards(timestep):
    """Returns a dict mapping each 'prefix' → float reward."""
    rewards = {}
    for key, val in timestep.observation.items():
        if key.endswith(".REWARD"):
            prefix, name = key.split(".", 1)
            if name == "REWARD":
                rewards[prefix] = float(val)
    return rewards


# Collects agent's trajectories
# -------------------------------------------------------------------
def collect_trajectory_batch_per_agent(
    agent_list,
    env,
    primary_agent_id,
    action_dimension,
    network_parameters,
    rng_key_per_agent,
    ACTION_SET,
    steps_per_worker,
    discount_factor,
    network_model,
):
    buffer = {
        agent: {
            "observations": [],
            "actions": [],
            "logp": [],
            "values": [],
            "rewards": [],
            "zapped": [],
            "death_zapped": [],
        }
        for agent in agent_list
    }

    # start in a fresh episode
    timestep = env.reset()
    current_ep_steps = 0

    while len(buffer[primary_agent_id]["observations"]) < steps_per_worker:
        action_dict = {}

        # for each agent: compute policy, sample, and store obs/action/logp/value
        for agent in agent_list:
            # agent = int(agent)
            img = timestep.observation[f"{agent}.RGB"]
            x = jnp.asarray(img, jnp.float32).transpose(2, 0, 1)[None, ...]

            logits, value = network_model.apply(network_parameters[agent], x)
            dist = distrax.Categorical(logits=logits[0])
            rng_key_per_agent[agent], sub = random.split(rng_key_per_agent[agent])
            a = dist.sample(seed=sub)
            logp = dist.log_prob(a)

            # map to primitive actions
            for name, val in ACTION_SET[int(a)].items():
                action_dict[f"{agent}.{name}"] = val

            buf = buffer[agent]
            buf["observations"].append(x[0])
            buf["actions"].append(a)
            buf["logp"].append(logp)
            buf["values"].append(value[0])

        # step the environment
        timestep = env.step(action_dict)
        current_ep_steps += 1

        # extract all agents' rewards once, then distribute
        reward_dict = get_multi_rewards(timestep)
        for agent in agent_list:
            r = reward_dict.get(agent, 0.0)
            buffer[agent]["rewards"].append(jnp.asarray(r, dtype=jnp.float32))

        # extract global metrics
        buffer[primary_agent_id]["zapped"].append(
            np.array(timestep.observation["WORLD.WHO_ZAPPED_WHO"])
        )
        buffer[primary_agent_id]["death_zapped"].append(
            np.array(timestep.observation["WORLD.WHO_DEATH_ZAPPED_WHO"])
        )

        # if episode ended, reset
        if timestep.last():
            current_ep_steps = 0
            timestep = env.reset()

    # compute discounted returns
    for agent in agent_list:
        G, rets = 0.0, []
        for r in reversed(buffer[agent]["rewards"]):
            G = r + discount_factor * G
            rets.insert(0, G)
        buffer[agent]["returns"] = rets

    return buffer


# Parallelization
# -------------------------------------------------------------------
import jax.random as jr


def actor_worker(
    worker_id,
    args,
    network_parameters,  # dict[str → PyTree] of params, keyed by agent ID
    worker_key,  # a single PRNGKey for this worker
    env_config,
    ACTION_SET,
    discount_factor,
    batch_size,
    network_model,
):
    """Collect trajectory data in parallel using a worker process.
    
    This function rebuilds the environment and collects trajectories for a subset of the total
    batch size. It's designed to be run in parallel across multiple workers.
    
    Args:
        worker_id: Integer ID of this worker process
        args: Parsed command line arguments
        network_parameters: Dict mapping agent IDs to their neural network parameters
        worker_key: JAX PRNGKey for this worker's random number generation
        env_config: MeltingPot environment configuration
        ACTION_SET: Set of possible actions in the environment
        discount_factor: Gamma parameter for computing returns
        batch_size: Total batch size (will be divided among workers)
        
    Returns:
        dict: Collected trajectory data for each agent containing observations,
             actions, rewards, etc.
    """
    # rebuild env
    with env_config.unlocked() as cfg:
        cfg.default_player_roles = ["PLAYER_ROLE_HARVESTER"] * 7
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)
    env = builder.builder(**env_config)

    # Create worker-specific random key
    local_key = jr.fold_in(worker_key, worker_id)

    # Split into one subkey per agent
    agent_ids = list(network_parameters.keys())  # e.g. ["1","2","3",...]
    num_agents = len(agent_ids)
    subkeys = jr.split(local_key, num=num_agents)

    # Build the RNG dict for each agent
    rngs = {agent_id: subkey for agent_id, subkey in zip(agent_ids, subkeys)}

    # Compute steps per worker based on total batch size
    steps_per_worker = batch_size // args.num_workers
    return collect_trajectory_batch_per_agent(
        agent_list=agent_ids,
        env=env,
        primary_agent_id=agent_ids[0],
        action_dimension=len(ACTION_SET),
        network_parameters=network_parameters,
        rng_key_per_agent=rngs,  # each agent has its own key now
        ACTION_SET=ACTION_SET,
        steps_per_worker=steps_per_worker,
        discount_factor=discount_factor,
        network_model=network_model,
    )


def merge_trajs(worker_trajs):
    """Merge trajectory data from multiple workers into a single batch.
    
    Args:
        worker_trajs: List of trajectory dictionaries from each worker
        
    Returns:
        dict: Merged trajectory data with concatenated arrays for each field
    """
    merged = {}
    agent_list = list(worker_trajs[0].keys())

    for agent in agent_list:
        merged[agent] = {}
        for key, first_list in worker_trajs[0][agent].items():
            # If there are no entries for this key, produce an empty array:
            if len(first_list) == 0:
                merged[agent][key] = jnp.array([], dtype=jnp.float32)
                continue

            # Otherwise stack each worker's lists and then concatenate:
            per_worker_arrs = [jnp.stack(w[agent][key]) for w in worker_trajs]
            merged[agent][key] = jnp.concatenate(per_worker_arrs, axis=0)

    return merged


def main():
    """Main training function implementing PPO algorithm for commons harvest.
    
    This function:
    1. Initializes the environment and agents
    2. Sets up logging and metrics tracking
    3. Runs the main training loop collecting experience and updating policies
    4. Saves checkpoints and metrics
    """
    args = parse_args()

    # seed the global key using the command line argument
    global_rng_key = random.PRNGKey(args.seed)
    
    # Also set numpy random seed for consistency
    np.random.seed(args.seed)

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Initialize Weights & Biases for experiment tracking
    # -------------------------------------------------------------------
    if not args.no_wandb:
        wandb.init(
            project="commons_harvest_ppo",
            config={
                "discount_factor": args.discount_factor,
                "learning_rate": args.learning_rate,
                "ppo_clip_epsilon": args.ppo_clip_epsilon,
                "batch_size": args.batch_size,
                "ppo_epochs": args.ppo_epochs,
                "total_training_updates": args.total_training_updates,
                "kl_threshold": args.kl_threshold,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
            tags=["jax", "flax", "ppo", args.mode],
            reinit=True,
        )

        config = wandb.config
    else:
        # Use hardcoded defaults if not using W&B
        class C:
            pass

        config = C()
        config.discount_factor = DISCOUNT_FACTOR
        config.learning_rate = LEARNING_RATE
        config.ppo_clip_epsilon = PPO_CLIP_EPSILON
        config.batch_size = BATCH_SIZE
        config.ppo_epochs = PPO_EPOCHS
        config.total_training_updates = TOTAL_TRAINING_UPDATES
        config.kl_threshold = KL_THRESHOLD
        config.num_workers = args.num_workers

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Build the MeltingPot environment and initialize parameters
    # -------------------------------------------------------------------
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Launching in %s mode", args.mode)
    logger.info("Using random seed: %d", args.seed)

    # Load and configure the environment
    env_config = commons_harvest__open.get_config()
    with env_config.unlocked() as cfg:
        # Set up 7 harvester agents
        cfg.default_player_roles = ["PLAYER_ROLE_HARVESTER"] * 7
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)

    # Create environment instance
    env = builder.builder(**env_config)  # returns a `dmlab2d.Environment`

    # Set up agent IDs and action space
    agent_list = [str(i + 1) for i in range(len(roles))]
    n_players = len(agent_list)
    primary_agent_id = agent_list[0]
    ACTION_SET = commons_harvest__open.ACTION_SET
    action_dimension = len(ACTION_SET)

    # Get observation shape from a dummy environment reset
    dummy_timestep = env.reset()
    rgb = dummy_timestep.observation[f"{primary_agent_id}.RGB"]
    obs_height, obs_width, obs_channels = rgb.shape
    observation_shape = (obs_channels, obs_height, obs_width)

    # Initialize metrics tracking matrices
    zap_matrix = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    death_zap_matrix = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    death_zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    all_zaps = []
    all_deaths = []

    # Initialize reward tracking
    reward_history = {agent: [] for agent in agent_list}
    cum_reward = {agent: 0.0 for agent in agent_list}

    # Set up optimizer
    optimizer = optax.adam(config.learning_rate)

    # Handle human play mode if specified
    if args.mode == "human":
        _ACTION_MAP = {
            "move": level_playing_utils.get_direction_pressed,
            "turn": level_playing_utils.get_turn_pressed,
            "fireZap": level_playing_utils.get_space_key_pressed,
            "deathZap": level_playing_utils.get_enter_key_pressed,
        }

        with config_dict.ConfigDict(env_config).unlocked() as env_config:
            cfg.default_player_roles = ["PLAYER_ROLE_HARVESTER"] * 7
            roles = env_config.default_player_roles
            env_config.lab2d_settings = commons_harvest__open.build(roles, env_config)
        level_playing_utils.run_episode(
            "RGB",
            {},
            _ACTION_MAP,
            env_config,
            level_playing_utils.RenderType.PYGAME,
        )
        return

    # Initialize network parameters and optimizer states for each agent
    network_parameters = {}
    optimizer_states = {}
    
    # Create a single network model instance to be used throughout
    network_model = ActorCriticNetwork(action_dimension)

    for agent_id in agent_list:
        # Initialize network parameters with a dummy observation
        _, init_rng = random.split(global_rng_key)
        dummy_obs = jnp.zeros((1, *observation_shape), jnp.float32)
        params = network_model.init(init_rng, dummy_obs)
        opt_state = optimizer.init(params)

        network_parameters[agent_id] = params
        optimizer_states[agent_id] = opt_state

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Main training loop
    # -------------------------------------------------------------------
    # Set up parallel workers for data collection
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    logger = logging.getLogger("train")
    log_jax_devices()

    # Define PPO loss and update functions that will be used in training
    # action_dimension required in ActorCriticNetwork can't be passed as
    # as argument due to @jax.jit restrictions, it has to come from
    # context
    # -------------------------------------------------------------------
    def compute_ppo_loss(network_model, params, obs, acts, old_logp, old_val, rets):
        """Compute the PPO loss combining policy and value losses.
        
        This function implements the PPO-Clip objective, combining a clipped policy loss
        with a value function loss. The clipping prevents too large policy updates.
        
        Args:
            network_model: The ActorCriticNetwork model
            params: Neural network parameters
            obs: Batch of observations
            acts: Batch of actions taken
            old_logp: Log probabilities of actions under old policy
            old_val: Value estimates from old policy
            rets: Discounted returns
            
        Returns:
            float: Combined PPO loss (policy loss + value loss)
        """
        # Forward pass through network to get new policy and values
        logits, vals = network_model.apply(params, obs)
        dist = distrax.Categorical(logits=logits)

        # Compute log probabilities of actions under new policy
        logp = dist.log_prob(acts)
        
        # Compute importance sampling ratio
        ratio = jnp.exp(logp - old_logp)

        # Compute advantages and normalize them
        adv = rets - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Compute PPO clipped surrogate objective
        unclipped = ratio * adv  # Unclipped surrogate
        clipped = (
            jnp.clip(ratio, 1 - config.ppo_clip_epsilon, 1 + config.ppo_clip_epsilon)
            * adv
        )  # Clipped surrogate
        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

        # Compute value function loss on raw returns
        value_loss = jnp.mean((rets - vals) ** 2)
        
        # Return combined loss
        return policy_loss + 0.5 * value_loss

    # Create a JIT-compiled update function
    def create_ppo_update_step(network_model):
        def update_step(params, opt_state, obs, acts, old_logp, old_val, rets):
            """Perform one PPO update step using the provided batch of data."""
            # Compute loss and gradients
            loss_fn = lambda p: compute_ppo_loss(network_model, p, obs, acts, old_logp, old_val, rets)
            loss, grads = value_and_grad(loss_fn)(params)
            
            # Apply optimizer update
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state, loss
        
        return jax.jit(update_step)
    
    # Create the JIT-compiled update function with our network model
    ppo_update_step = create_ppo_update_step(network_model)

    for update_idx in range(config.total_training_updates):
        # Generate new RNG keys for each worker
        subkeys = random.split(global_rng_key, num=args.num_workers + 1)
        global_rng_key = subkeys[0]
        worker_keys = subkeys[1:]  # a list of length num_workers

        # make a one‐off, isolated copy of the current parameters
        # in order to avoid race conditions and overwriting
        network_params_snapshot = freeze(network_parameters)

        # Collect trajectories in parallel
        futures = [
            pool.submit(
                actor_worker,
                wid,
                args,
                network_params_snapshot,
                worker_keys[wid],  # One PRNGKey per worker
                env_config,
                ACTION_SET,
                config.discount_factor,
                config.batch_size,
                network_model,
            )
            for wid in range(args.num_workers)
        ]
        
        # Gather results from workers
        worker_trajs = []
        for wid, fut in enumerate(futures):
            try:
                worker_trajs.append(fut.result())
            except Exception as e:
                logger.error(f"actor_worker {wid} failed", exc_info=True)
                #  - raise   # to stop training immediately
                #  - continue  # to skip this worker and merge the rest
                raise
                
        # Merge trajectories from all workers
        traj = merge_trajs(worker_trajs)

        # Log progress
        steps_this_iter = len(traj[primary_agent_id]["observations"])
        total_steps = steps_this_iter * (update_idx + 1)
        logger.debug(
            "Collected %d steps (this iter) and %d steps (total)",
            steps_this_iter,
            total_steps,
        )

        # Update cumulative rewards
        for agent in agent_list:
            for _, raw_r in enumerate(traj[agent]["rewards"]):
                r = float(raw_r)

                # update running total
                cum_reward[agent] += r
                reward_history[agent].append(cum_reward[agent])

        # Collect zapping statistics
        batch_zaps = np.stack(traj[primary_agent_id]["zapped"])
        batch_deaths = np.stack(traj[primary_agent_id]["death_zapped"])
        all_zaps.append(batch_zaps)
        all_deaths.append(batch_deaths)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Training iteration: %d", update_idx)

        # Perform PPO updates for each agent
        for agent in agent_list:
            # wandb logging
            params = network_parameters[agent]
            opt_state = optimizer_states[agent]

            # Prepare trajectory data for updates
            o = jnp.stack(traj[agent]["observations"])
            a = jnp.stack(traj[agent]["actions"])
            lp = jnp.stack(traj[agent]["logp"])
            v = jnp.stack(traj[agent]["values"])
            G = jnp.stack(traj[agent]["returns"])

            # Track metrics for this agent's updates
            epoch_losses = []
            epoch_kls = []

            # Run multiple epochs of PPO updates
            for epoch_idx in range(config.ppo_epochs):
                # Perform PPO update step
                new_params, new_opt_state, loss = ppo_update_step(params, opt_state, o, a, lp, v, G)
                epoch_losses.append(float(loss))

                # Compute KL divergence between old and new policies
                logits_old, _ = network_model.apply(params, o)
                logits_new, _ = network_model.apply(new_params, o)
                avg_kl = jnp.mean(
                    distrax.Categorical(logits=logits_old).kl_divergence(
                        distrax.Categorical(logits=logits_new)
                    )
                )
                epoch_kls.append(avg_kl)

                # **always capture the scalar** so it's in scope below
                avg_kl_value = float(avg_kl.item())

                # Early stopping if KL divergence too high
                if avg_kl > config.kl_threshold:
                    logger.info(
                        f"Stopping PPO epochs for agent {agent} at epoch {epoch_idx} "
                        f"due to KL={avg_kl_value:.4f} > {config.kl_threshold:.4f}"
                    )
                    break

                # Update parameters if continuing
                params, opt_state = new_params, new_opt_state
                network_parameters[agent] = params
                optimizer_states[agent] = opt_state
            
            # Store metrics for this agent to log later
            if not args.no_wandb:
                wandb.log(
                    {
                        f"{agent}/ppo_loss_final": epoch_losses[-1],
                        f"{agent}/kl_final": epoch_kls[-1],
                        f"{agent}/kl_max": max(epoch_kls),
                        f"{agent}/kl_mean": sum(epoch_kls) / len(epoch_kls),
                        f"{agent}/reward_mean": np.mean([float(r) for r in traj[agent]["rewards"]]),
                    },
                    step=update_idx,
                )

        # Log metrics
        rewards = [float(r) for r in traj[primary_agent_id]["rewards"]]
        mean_reward = sum(rewards) / len(rewards)

        # Log to W&B if enabled
        if not args.no_wandb:
            # e.g. log the *last* epoch's loss and avg_kl, plus some summary stats
            wandb.log(
                {
                    "global/mean_reward": mean_reward,
                    "global/steps_collected": steps_this_iter,
                    "global/total_steps": total_steps,
                },
                step=update_idx,
            )

    pool.shutdown(wait=True)

    # Process final metrics
    all_zaps = np.concatenate(all_zaps, axis=0)
    zap_through_time = jnp.cumsum(all_zaps, axis=0)
    zap_matrix = zap_through_time[-1]
    all_deaths = np.concatenate(all_deaths, axis=0)
    death_zap_through_time = jnp.cumsum(all_deaths, axis=0)
    death_zap_matrix = death_zap_through_time[-1]

    # 4) Save checkpoints and metrics
    # -------------------------------------------------------------------
    # Create checkpoint directory structure
    ckpt_root = pathlib.Path("checkpoints")
    ckpt_root.mkdir(exist_ok=True)

    # Create a timestamped directory for this training run
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ckpt_root / run_id
    run_dir.mkdir()

    # Save neural network parameters for each agent
    for agent, params in network_parameters.items():
        path = run_dir / f"agent_{agent}_params.msgpack"
        with path.open("wb") as fp:
            fp.write(serialization.to_bytes(params))

    # Save run artifacts to W&B if enabled
    if not args.no_wandb:
        artifact = wandb.Artifact("commons_harvest_models", type="model")
        artifact.add_dir(str(run_dir))
        wandb.log_artifact(artifact)

    # Save reward history for each agent
    # Convert reward_history from dict[str, list[float]] to DataFrame with one column per agent
    df_rewards = pd.DataFrame(reward_history)
    rewards_path = run_dir / "reward_history.csv"
    df_rewards.to_csv(rewards_path, index=False, header=False)
    logger.info("Saved reward history to %s", rewards_path)

    # Save zapping interaction matrices
    # zap_matrix: 2D array showing total zaps between each pair of agents
    df_zap = pd.DataFrame(np.array(zap_matrix))
    zap_path = run_dir / "zap_matrix.csv"
    df_zap.to_csv(zap_path, index=False, header=False)
    logger.info("Saved zap matrix to %s", zap_path)

    # death_zap_matrix: 2D array showing total death zaps between each pair of agents
    df_death_zap = pd.DataFrame(np.array(death_zap_matrix))
    death_zap_path = run_dir / "death_zap_matrix.csv"
    df_death_zap.to_csv(death_zap_path, index=False, header=False)
    logger.info("Saved death-zap matrix to %s", death_zap_path)

    # Save temporal zapping data
    # zap_through_time and death_zap_through_time: 3D arrays of shape [timesteps, n_players, n_players]
    # showing how zapping interactions evolved over the training run
    zap_arr = np.array(zap_through_time)  # shape [T, N, N]
    death_arr = np.array(death_zap_through_time)  # shape [T, N, N]
    np.savez_compressed(
        run_dir / "zap_data_through_time.npz", zap=zap_arr, death=death_arr
    )
    logger.info(
        "Saved zap and death‐zap through time to %s (arrays shapes %s, %s)",
        run_dir / "zap_data_through_time.npz",
        zap_arr.shape,
        death_arr.shape,
    )

    # Save hyperparameters used for this training run
    hyperparams = {
        "DISCOUNT_FACTOR": config.discount_factor,
        "LEARNING_RATE": config.learning_rate,
        "PPO_CLIP_EPSILON": config.ppo_clip_epsilon,
        "BATCH_SIZE": config.batch_size,
        "PPO_EPOCHS": config.ppo_epochs,
        "TOTAL_TRAINING_UPDATES": config.total_training_updates,
        "KL_THRESHOLD": config.kl_threshold,
        "SEED": args.seed,
    }

    # Save hyperparameters as a single-row CSV
    df_hyper = pd.DataFrame([hyperparams])
    hp_path = run_dir / "hyperparameters.csv"
    df_hyper.to_csv(hp_path, index=False)
    logger.info("Saved hyperparameters to %s", hp_path)

    # Clean up W&B
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
