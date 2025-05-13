import argparse
import copy
import datetime
import json
import pathlib
import wandb

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import serialization
from jax import jit, random, value_and_grad
import jax
from jax.lib import xla_bridge
from ml_collections import config_dict

import meltingpot.human_players.level_playing_utils as level_playing_utils
from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.utils.substrates import builder

# ── Hyperparameters ─────────────────────────────────────────────────────────────
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.000238534447933878 #3e-4
PPO_CLIP_EPSILON = 0.1 #0.2
BATCH_SIZE = 256 #128
PPO_EPOCHS = 5
TOTAL_TRAINING_UPDATES = 100
KL_THRESHOLD = 0.04889358402136939 #1e-2
# ────────────────────────────────────────────────────────────────────────────────

# Utils
# -------------------------------------------------------------------

import logging
import sys


def parse_args():
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
        "--settings",
        type=json.loads,
        default="{}",
        help="JSON dict of overrides for the substrate",
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
        "--total‐updates",
        type=int,
        default=5,
        help="number of PPO outer updates",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "COMPLETE"),
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
    p.add_argument("--learning_rate",      type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size",         type=int,   default=BATCH_SIZE)
    p.add_argument("--ppo_clip_epsilon",   type=float, default=PPO_CLIP_EPSILON)
    p.add_argument("--ppo_epochs",         type=int,   default=PPO_EPOCHS)
    p.add_argument("--kl_threshold",       type=float, default=KL_THRESHOLD)
    p.add_argument("--discount_factor",       type=float, default=DISCOUNT_FACTOR)
    p.add_argument("--total_training_updates", type=int, default=TOTAL_TRAINING_UPDATES)
    return p.parse_args()


def configure_logging(level: str) -> None:
    """Configure root and library loggers for reproducible output."""
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence extremely chatty third‑party libraries unless debug is requested
    if level != "COMPLETE":
        for noisy in ("absl", "jaxlib", "jax"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

def log_jax_devices():
    """Log available JAX backend and devices."""
    logger = logging.getLogger(__name__)
    backend = xla_bridge.get_backend()
    logger.info("JAX XLA backend platform: %s", backend.platform)
    devices = jax.devices()
    for dev in devices:
        logger.info(
            "  Device: %s (id=%d, process_index=%d)",
            dev.device_kind, dev.id, dev.process_index
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

            logits, value = ActorCriticNetwork(action_dimension).apply(
                network_parameters[agent], x
            )
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

        # extract all agents’ rewards once, then distribute
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
    network_parameters,      # dict[str → PyTree] of params, keyed by agent ID
    worker_key,              # a single PRNGKey for this worker
    env_config,
    ACTION_SET,
    discount_factor,
    batch_size,
):
    # rebuild env
    with env_config.unlocked() as cfg:
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)
    env = builder.builder(**env_config)

    # optionally mix in the worker_id so keys differ across processes:
    local_key = jr.fold_in(worker_key, worker_id)

    # split into one subkey per agent:
    agent_ids = list(network_parameters.keys())    # e.g. ["1","2","3",...]
    num_agents = len(agent_ids)
    subkeys    = jr.split(local_key, num=num_agents)

    # build the rngs dict
    rngs = {agent_id: subkey for agent_id, subkey in zip(agent_ids, subkeys)}

    steps_per_worker = batch_size // args.num_workers
    return collect_trajectory_batch_per_agent(
        agent_list         = agent_ids,
        env                = env,
        primary_agent_id   = agent_ids[0],
        action_dimension   = len(ACTION_SET),
        network_parameters = network_parameters,
        rng_key_per_agent  = rngs,                # each agent has its own key now
        ACTION_SET         = ACTION_SET,
        steps_per_worker   = steps_per_worker,
        discount_factor    = discount_factor,
    )

def merge_trajs(worker_trajs):
    """Concatenate buffers from num_workers into one full batch."""
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

    args = parse_args()

    # seed the global key
    global_rng_key = random.PRNGKey(0)

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Initialize Weights & Biases or hardcoded parameters
    # -------------------------------------------------------------------
    if not args.no_wandb:
        wandb.init(
          project="commons_harvest_ppo",
          config={
              "discount_factor":      args.discount_factor,
              "learning_rate":        args.learning_rate,
              "ppo_clip_epsilon":     args.ppo_clip_epsilon,
              "batch_size":           args.batch_size,
              "ppo_epochs":           args.ppo_epochs,
              "total_training_updates": args.total_training_updates,
              "kl_threshold":         args.kl_threshold,
              "num_workers":          args.num_workers,
          },
          tags=["jax", "flax", "ppo", args.mode],
          reinit=True,
        )

        config = wandb.config
    else:
        class C: pass
        config = C()
        config.discount_factor      = DISCOUNT_FACTOR
        config.learning_rate        = LEARNING_RATE
        config.ppo_clip_epsilon     = PPO_CLIP_EPSILON
        config.batch_size           = BATCH_SIZE
        config.ppo_epochs           = PPO_EPOCHS
        config.total_training_updates = TOTAL_TRAINING_UPDATES
        config.kl_threshold         = KL_THRESHOLD
        config.num_workers          = args.num_workers

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Build the MeltingPot environment and initialize params
    # -------------------------------------------------------------------
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Launching in %s mode", args.mode)

    # Load the substrate config
    env_config = commons_harvest__open.get_config()
    # Build the `lab2d_settings` key
    with env_config.unlocked() as cfg:
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)

    # instantiate the environment
    env = builder.builder(**env_config)  # returns a `dmlab2d.Environment`

    # determine agents
    agent_list = [str(i + 1) for i in range(len(roles))]
    n_players = len(agent_list)
    primary_agent_id = agent_list[0]

    # All agents share the same discrete action set
    # TODO: Should make these properly global
    ACTION_SET = commons_harvest__open.ACTION_SET
    action_dimension = len(ACTION_SET)

    # Observation shape for the RGB channel
    # We will extract per-agent obs via timestep.observation[f"{agent_id}.RGB"]
    # and reshape into [C,H,W].
    # all agents get the same RGB shape
    dummy_timestep = env.reset()
    rgb = dummy_timestep.observation[f"{primary_agent_id}.RGB"]
    obs_height, obs_width, obs_channels = rgb.shape
    observation_shape = (obs_channels, obs_height, obs_width)

    # instantiate global variables within the main
    zap_matrix = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    death_zap_matrix = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    death_zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    all_zaps = []
    all_deaths = []

    reward_history = {agent: [] for agent in agent_list}
    cum_reward     = {agent: 0.0 for agent in agent_list}

    # instantiate optimizer
    optimizer = optax.adam(config.learning_rate)

    # Defines human mode for debugging
    if args.mode == "human":
        _ACTION_MAP = {
            "move": level_playing_utils.get_direction_pressed,
            "turn": level_playing_utils.get_turn_pressed,
            "fireZap": level_playing_utils.get_space_key_pressed,
            "deathZap": level_playing_utils.get_enter_key_pressed,
        }

        with config_dict.ConfigDict(env_config).unlocked() as env_config:
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

    network_parameters = {}
    optimizer_states = {}
    rng_key_per_agent = {}

    for agent_id in agent_list:
        _, init_rng = random.split(global_rng_key) # throw away the dummy global_rng_key generated here
        dummy_obs = jnp.zeros((1, *observation_shape), jnp.float32)
        params = ActorCriticNetwork(action_dimension).init(init_rng, dummy_obs)
        opt_state = optimizer.init(params)

        network_parameters[agent_id] = params
        optimizer_states[agent_id] = opt_state

    # variable to auxiliate in normalizing return
    running_mean   = {agent: 0.0    for agent in agent_list}
    running_var    = {agent: 1.0    for agent in agent_list}
    running_count  = {agent: 1e-4   for agent in agent_list}

    # 3) Main training loop
    # -------------------------------------------------------------------
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    logger = logging.getLogger("train")
    log_jax_devices()

    # Define PPO loss and update functions that will be used in training
    # action_dimension required in ActorCriticNetwork can't be passed as
    # as argument due to @jax.jit restrictions, it has to come from
    # context
    # -------------------------------------------------------------------
    def compute_ppo_loss(params, obs, acts, old_logp, old_val, rets):
        logits, vals = ActorCriticNetwork(action_dimension).apply(params, obs)
        dist = distrax.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = jnp.exp(logp - old_logp)

        adv = rets - old_val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1 - config.ppo_clip_epsilon, 1 + config.ppo_clip_epsilon) * adv

        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        value_loss = jnp.mean((rets - vals) ** 2)
        return policy_loss + 0.5 * value_loss

    @jit
    def ppo_update_step(params, opt_state, obs, acts, old_logp, old_val, rets):
        loss, grads = value_and_grad(compute_ppo_loss)(
            params, obs, acts, old_logp, old_val, rets
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for update_idx in range(config.total_training_updates):
        # num_workers+1 subkeys for different random initializations
        subkeys = random.split(global_rng_key, num=args.num_workers + 1)
        global_rng_key = subkeys[0]
        worker_keys    = subkeys[1:]  # a list of length num_workers

        # make a one‐off, isolated copy of the current parameters
        # in order to avoid race conditions and overwriting
        # TODO: possibly changing this to 
        # from flax.core import freeze
        #     frozen_params = freeze(network_parameters)
        # will make it faster?
        network_params_snapshot = copy.deepcopy(network_parameters)

        futures = [
            pool.submit(actor_worker,
                        wid,
                        args,
                        network_params_snapshot,
                        worker_keys[wid],  # ← one PRNGKey here
                        env_config,
                        ACTION_SET,
                        config.discount_factor,
                        config.batch_size)
            for wid in range(args.num_workers)
        ]
        worker_trajs = []
        for wid, fut in enumerate(futures):
            try:
                worker_trajs.append(fut.result())
            except Exception as e:
                logger.error(f"actor_worker {wid} failed", exc_info=True)
                #  - raise   # to stop training immediately
                #  - continue  # to skip this worker and merge the rest
                raise
        traj = merge_trajs(worker_trajs)

        # ——— Running‐moment update & normalization ——— #
        for agent in agent_list:
            # raw returns from the buffer (shape [BATCH_SIZE])
            R = jnp.stack(traj[agent]["returns"])

            # batch statistics
            batch_mean  = float(R.mean())
            batch_var   = float(R.var())
            batch_count = R.shape[0]

            # previous running stats
            old_mean  = running_mean[agent]
            old_var   = running_var[agent]
            old_count = running_count[agent]

            # Welford’s algorithm
            delta = batch_mean - old_mean
            tot   = old_count + batch_count
            new_mean = old_mean + delta * batch_count / tot
            m_a = old_var * old_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * old_count * batch_count / tot
            new_var  = M2 / tot

            # store updated running stats
            running_mean[agent]  = new_mean
            running_var[agent]   = new_var
            running_count[agent] = tot

            # normalize returns
            R_norm = (R - new_mean) / (jnp.sqrt(new_var) + 1e-8)

            # replace in the trajectory dict
            traj[agent]["returns"] = R_norm
        # ———————————————————————————————— #

        # debugger logging
        steps_this_iter = len(traj[primary_agent_id]["observations"])
        total_steps     = steps_this_iter * (update_idx + 1)
        logger.debug("Collected %d steps (this iter) and %d steps (total)",
                     steps_this_iter, total_steps)

        # collect per-agent cumulative reward
        for agent in agent_list:
            # traj[agent]['rewards'] is a list of length BATCH_SIZE
            for _, raw_r in enumerate(traj[agent]["rewards"]):
                # convert to float
                r = float(raw_r)

                # update running total
                cum_reward[agent] += r

                # append the _running_ total at this step
                reward_history[agent].append(cum_reward[agent])

        # collect all agents zapping data
        batch_zaps = np.stack(traj[primary_agent_id]["zapped"])
        batch_deaths = np.stack(traj[primary_agent_id]["death_zapped"])
        all_zaps.append(batch_zaps)
        all_deaths.append(batch_deaths)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Training iteration: %d", update_idx)

        # PPO updates for each agent
        for agent in agent_list:
            # wandb logging
            params    = network_parameters[agent]
            opt_state = optimizer_states[agent]

            o = jnp.stack(traj[agent]["observations"])
            a = jnp.stack(traj[agent]["actions"])
            lp = jnp.stack(traj[agent]["logp"])
            v = jnp.stack(traj[agent]["values"])
            R = jnp.stack(traj[agent]["returns"])

            # prepare per‐agent metrics containers
            epoch_losses = []
            epoch_kls    = []

            for epoch_idx in range(config.ppo_epochs):
                # ppo upddate step
                new_params, new_opt_state, loss = ppo_update_step(
                    params, opt_state, o, a, lp, v, R
                )

                # compute avg KL divergence for early stopping
                logits_old, _ = ActorCriticNetwork(action_dimension).apply(params, o)
                logits_new, _ = ActorCriticNetwork(action_dimension).apply(new_params, o)
                avg_kl = jnp.mean(
                    distrax.Categorical(logits=logits_old).kl_divergence(
                        distrax.Categorical(logits=logits_new)
                    )
                )
                epoch_kls.append(avg_kl)

                # **always capture the scalar** so it's in scope below
                avg_kl_value = float(avg_kl.item())

                # if KL exceeds threshold, stop early
                if avg_kl > config.kl_threshold:
                    logger.info(
                        f"Stopping PPO epochs for agent {agent} at epoch {epoch_idx} "
                        f"due to KL={avg_kl_value:.4f} > {config.kl_threshold:.4f}"
                    )
                    break

                params, opt_state = new_params, new_opt_state
                network_parameters[agent] = params
                optimizer_states[agent] = opt_state

        rewards = [float(r) for r in traj[primary_agent_id]["rewards"]]
        mean_reward = sum(rewards) / len(rewards)

        # log per‐epoch metrics to W&B
        if not args.no_wandb:
            # e.g. log the *last* epoch’s loss and avg_kl, plus some summary stats
            wandb.log({
                f"{agent}/ppo_loss_final": epoch_losses[-1],
                f"{agent}/kl_final":        epoch_kls[-1],
                f"{agent}/kl_max":          max(epoch_kls),
                f"{agent}/kl_mean":         sum(epoch_kls) / len(epoch_kls),
            }, step=update_idx)


    pool.shutdown(wait=True)

    # TOTAL_TRAINING_UPDATES is done and all steps can be processed
    all_zaps = np.concatenate(all_zaps, axis=0)
    zap_through_time = jnp.cumsum(all_zaps, axis=0)
    zap_matrix = zap_through_time[-1]
    all_deaths = np.concatenate(all_deaths, axis=0)
    death_zap_through_time = jnp.cumsum(all_deaths, axis=0)
    death_zap_matrix = death_zap_through_time[-1]

    # 4) Save checkpoints
    # -------------------------------------------------------------------
    ckpt_root = pathlib.Path("checkpoints")
    ckpt_root.mkdir(exist_ok=True)

    # Create a run-specific subdirectory using the current date/time
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ckpt_root / run_id
    run_dir.mkdir()

    # save model parameters
    for agent, params in network_parameters.items():
        path = run_dir / f"agent_{agent}_params.msgpack"
        with path.open("wb") as fp:
            fp.write(serialization.to_bytes(params))

    # saves wandb logs as artifact
    if not args.no_wandb:
      artifact = wandb.Artifact("commons_harvest_models", type="model")
      artifact.add_dir(str(run_dir))
      wandb.log_artifact(artifact)

    # reward_history: dict[str, list[float]] → DataFrame with one column per agent
    df_rewards = pd.DataFrame(reward_history)
    rewards_path = run_dir / "reward_history.csv"
    df_rewards.to_csv(rewards_path, index=False, header=False)
    logger.info("Saved reward history to %s", rewards_path)

    # zap_matrix and death_zap_matrix: both are 2D arrays of shape [n_players, n_players]
    df_zap = pd.DataFrame(np.array(zap_matrix))
    zap_path = run_dir / "zap_matrix.csv"
    df_zap.to_csv(zap_path, index=False, header=False)
    logger.info("Saved zap matrix to %s", zap_path)

    df_death_zap = pd.DataFrame(np.array(death_zap_matrix))
    death_zap_path = run_dir / "death_zap_matrix.csv"
    df_death_zap.to_csv(death_zap_path, index=False, header=False)
    logger.info("Saved death-zap matrix to %s", death_zap_path)

    # zap_through_time and death_zap_through_time: 3D arrays of shape [steps, n_players, n_players]
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

    # Collect hyperparameters into a dict
    hyperparams = {
        "DISCOUNT_FACTOR":           config.discount_factor,
        "LEARNING_RATE":             config.learning_rate,
        "PPO_CLIP_EPSILON":          config.ppo_clip_epsilon,
        "BATCH_SIZE":                config.batch_size,
        "PPO_EPOCHS":                config.ppo_epochs,
        "TOTAL_TRAINING_UPDATES":    config.total_training_updates,
        "KL_THRESHOLD":              config.kl_threshold,
    }

    # Create a one‐row DataFrame and write it out
    df_hyper = pd.DataFrame([hyperparams])
    hp_path = run_dir / "hyperparameters.csv"
    df_hyper.to_csv(hp_path, index=False)
    logger.info("Saved hyperparameters to %s", hp_path)

    if not args.no_wandb:
      wandb.finish()

if __name__ == "__main__":
    main()
