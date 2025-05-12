import argparse
import datetime
import json
import pathlib

import distrax
import dm_env
import dmlab2d
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import serialization
from jax import jit, random, value_and_grad
from ml_collections import config_dict

import meltingpot.human_players.level_playing_utils as level_playing_utils
from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.human_players.level_playing_utils import _get_rewards
from meltingpot.utils.substrates import builder

# ── Hyperparameters ─────────────────────────────────────────────────────────────
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 3e-4
PPO_CLIP_EPSILON = 0.2
BATCH_SIZE = 128
PPO_EPOCHS = 3
TOTAL_TRAINING_UPDATES = 1
KL_THRESHOLD = 1e-2
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


def collect_trajectory_batch_per_agent(
    agent_list,
    env,
    primary_agent_id,
    action_dimension,
    network_parameters,
    rng_key_per_agent,
    ACTION_SET,
    steps_per_agent=BATCH_SIZE,
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

    while len(buffer[primary_agent_id]["observations"]) < steps_per_agent:
        action_dict = {}

        # for each agent: compute policy, sample, and store obs/action/logp/value
        for agent in agent_list:
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
            timestep = env.reset

    # compute discounted returns
    for agent in agent_list:
        G, rets = 0.0, []
        for r in reversed(buffer[agent]["rewards"]):
            G = r + DISCOUNT_FACTOR * G
            rets.insert(0, G)
        buffer[agent]["returns"] = rets

    return buffer


# Define PPO loss and update functions that will be used in training
# -------------------------------------------------------------------
def compute_ppo_loss(params, obs, acts, old_logp, old_val, rets):
    logits, vals = ActorCriticNetwork(action_dimension).apply(params, obs)
    dist = distrax.Categorical(logits=logits)
    logp = dist.log_prob(acts)
    ratio = jnp.exp(logp - old_logp)

    adv = rets - old_val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * adv

    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((rets - vals) ** 2)
    return policy_loss + 0.5 * value_loss


@jit
def ppo_update_step(params, opt_state, obs, acts, old_logp, old_val, rets):
    loss, grads = value_and_grad(compute_ppo_loss)(
        params, obs, acts, old_logp, old_val, rets
    )
    updates, new_opt_state = optax.adam(LEARNING_RATE).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


def main():
    # 1) Build the MeltingPot environment and initialize params
    # -------------------------------------------------------------------
    args = parse_args()
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
    zap_increment = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    death_increment = jnp.zeros((n_players, n_players), dtype=jnp.int32)
    zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    death_zap_through_time = jnp.zeros((1, n_players, n_players), dtype=jnp.int32)
    all_zaps = []
    all_deaths = []

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

    global_rng_key = random.PRNGKey(0)
    network_parameters = {}
    optimizer_states = {}
    rng_key_per_agent = {}

    for agent_id in agent_list:
        global_rng_key, init_rng = random.split(global_rng_key)
        dummy_obs = jnp.zeros((1, *observation_shape), jnp.float32)
        params = ActorCriticNetwork(action_dimension).init(init_rng, dummy_obs)
        opt = optax.adam(LEARNING_RATE)
        opt_state = opt.init(params)

        network_parameters[agent_id] = params
        optimizer_states[agent_id] = opt_state
        rng_key_per_agent[agent_id] = init_rng

    # 2) Main training loop
    # -------------------------------------------------------------------
    logger = logging.getLogger("train")

    reward_history = {agent: [] for agent in agent_list}

    for update_idx in range(TOTAL_TRAINING_UPDATES):
        traj = collect_trajectory_batch_per_agent(
            agent_list,
            env,
            primary_agent_id,
            action_dimension,
            network_parameters,
            rng_key_per_agent,
            ACTION_SET,
        )
        logger.debug(
            "Collected %d environment steps",
            len(traj[primary_agent_id]["observations"] * (update_idx + 1)),
        )

        #  per-agent cumulative reward
        for agent in agent_list:
            cum_r = float(jnp.sum(jnp.stack(traj[agent]["rewards"])))
            reward_history[agent].append(cum_r)

        batch_zaps = np.stack(traj[primary_agent_id]["zapped"])
        batch_deaths = np.stack(traj[primary_agent_id]["death_zapped"])

        all_zaps.append(batch_zaps)
        all_deaths.append(batch_deaths)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Training iteration: %d", update_idx)
            logger.debug("zap matrix:\n%s", zap_matrix)
            logger.debug("death_zap matrix:\n%s", death_zap_matrix)

        # PPO updates
        for agent in agent_list:
            o = jnp.stack(traj[agent]["observations"])
            a = jnp.stack(traj[agent]["actions"])
            lp = jnp.stack(traj[agent]["logp"])
            v = jnp.stack(traj[agent]["values"])
            R = jnp.stack(traj[agent]["returns"])

            params, opt_state = network_parameters[agent], optimizer_states[agent]
            for epoch_idx in range(PPO_EPOCHS):
                new_params, new_opt_state, loss = ppo_update_step(
                    params, opt_state, o, a, lp, v, R
                )

                # compute avg KL divergence between old and new policies
                logits_old, _ = ActorCriticNetwork(action_dimension).apply(params, o)
                logits_new, _ = ActorCriticNetwork(action_dimension).apply(
                    new_params, o
                )
                dist_old = distrax.Categorical(logits=logits_old)
                dist_new = distrax.Categorical(logits=logits_new)
                avg_kl = jnp.mean(dist_old.kl_divergence(dist_new))

                if avg_kl > KL_THRESHOLD:
                    avg_kl_value = avg_kl.item()
                    logger.info(
                        f"Stopping PPO epochs for agent {agent} at epoch {epoch_idx} "
                        f"due to KL={avg_kl_value:.4f} > {KL_THRESHOLD:.4f}"
                    )
                    break

                # otherwise accept the update and continue
                params, opt_state = new_params, new_opt_state

            network_parameters[agent] = params
            optimizer_states[agent] = opt_state

    # TOTAL_TRAINING_UPDATES is done and all steps can be processed
    all_zaps = jnp.array(all_zaps)
    all_zaps = np.concatenate(all_zaps, axis=0)
    zap_through_time = jnp.cumsum(all_zaps, axis=0)
    zap_matrix = np.sum(zap_through_time[-1], axis=0)
    all_deaths = jnp.array(all_deaths)
    death_zap_through_time = jnp.cumsum(all_deaths, axis=0)
    death_zap_matrix = np.sum(death_zap_through_time[-1], axis=0)

    # 3) Save checkpoints
    # -------------------------------------------------------------------
    ckpt_root = pathlib.Path("checkpoints")
    ckpt_root.mkdir(exist_ok=True)

    # Create a run-specific subdirectory using the current date/time
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ckpt_root / run_id
    run_dir.mkdir()

    # 7a) Save model parameters
    for agent, params in network_parameters.items():
        path = run_dir / f"agent_{agent}_params.msgpack"
        with path.open("wb") as fp:
            fp.write(serialization.to_bytes(params))

    # reward_history: dict[str, list[float]] → DataFrame with one column per agent
    df_rewards = pd.DataFrame(reward_history)
    rewards_path = run_dir / "reward_history.csv"
    df_rewards.to_csv(rewards_path, index=False)
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
        "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
        "LEARNING_RATE": LEARNING_RATE,
        "PPO_CLIP_EPSILON": PPO_CLIP_EPSILON,
        "BATCH_SIZE": BATCH_SIZE,
        "PPO_EPOCHS": PPO_EPOCHS,
        "TOTAL_TRAINING_UPDATES": TOTAL_TRAINING_UPDATES,
        "KL_THRESHOLD": KL_THRESHOLD,
    }

    # Create a one‐row DataFrame and write it out
    df_hyper = pd.DataFrame([hyperparams])
    hp_path = run_dir / "hyperparameters.csv"
    df_hyper.to_csv(hp_path, index=False)
    logger.info("Saved hyperparameters to %s", hp_path)


if __name__ == "__main__":
    main()
