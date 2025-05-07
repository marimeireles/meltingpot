import argparse
import json
import pathlib
from ml_collections import config_dict

import dmlab2d
import dm_env

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad

import flax.linen as nn
import optax
import distrax

import numpy as np

from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.utils.substrates import builder

from meltingpot.human_players.level_playing_utils import _get_rewards
import meltingpot.human_players.level_playing_utils as level_playing_utils

# ── Hyperparameters ─────────────────────────────────────────────────────────────
DISCOUNT_FACTOR            = 0.99
LEARNING_RATE              = 3e-4
PPO_CLIP_EPSILON           = 0.2
STEPS_PER_UPDATE           = 128
PPO_EPOCHS                 = 4
TOTAL_TRAINING_UPDATES     = 5
EARLY_STOP_DELTA_THRESHOLD = 1e-2
EARLY_STOP_PATIENCE        = 10
# ────────────────────────────────────────────────────────────────────────────────

# 0) Utils

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
        "--render",
        action="store_true",
        help="display a PyGame window during training or human play",
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
    return p.parse_args()

def build_env(args):
    # 1) load config and attach lab2d_settings
    cfg = commons_harvest__open.get_config()
    with cfg.unlocked():
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)
    # 2) instantiate
    return builder.builder(**cfg), roles

def get_multi_rewards(timestep):
    """Returns a dict mapping each 'prefix' → float reward."""
    rewards = {}
    for key, val in timestep.observation.items():
        if key.endswith(".REWARD"):
            prefix, name = key.split(".", 1)
            if name == "REWARD":
                rewards[prefix] = float(val)
    return rewards

# 1) Build the MeltingPot environment directly via DM Lab2D
# -------------------------------------------------------------------
def main():
    args = parse_args()

    # Load the substrate config
    env_config = commons_harvest__open.get_config()
    # Build the `lab2d_settings` key
    with env_config.unlocked() as cfg:
        roles = cfg.default_player_roles
        cfg.lab2d_settings = commons_harvest__open.build(roles, cfg)

    # Instantiate the environment
    env = builder.builder(**env_config)  # returns a `dmlab2d.Environment`

    # Determine agent identifiers (string prefixes "1", "2", …)
    agent_list       = [str(i+1) for i in range(len(roles))]
    primary_agent_id = agent_list[0]

    # All agents share the same discrete action set
    ACTION_SET       = commons_harvest__open.ACTION_SET
    action_dimension = len(ACTION_SET)

    # Observation shape for the RGB channel
    # We will extract per-agent obs via timestep.observation[f"{agent_id}.RGB"]
    # and reshape into [C,H,W].
    # We assume all agents get the same RGB shape
    dummy_timestep = env.reset()
    rgb = dummy_timestep.observation[f"{primary_agent_id}.RGB"]
    obs_height, obs_width, obs_channels = rgb.shape
    observation_shape = (obs_channels, obs_height, obs_width)


    # 2) Define convolutional actor-critic network (Flax)
    # -------------------------------------------------------------------
    class ActorCriticNetwork(nn.Module):
        action_dimension: int

        @nn.compact
        def __call__(self, observations):
            x = observations / 255.0
            x = nn.Conv(32, (8, 8), (4, 4))(x); x = nn.relu(x)
            x = nn.Conv(64, (4, 4), (2, 2))(x); x = nn.relu(x)
            x = nn.Conv(64, (3, 3), (1, 1))(x); x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(512)(x); x = nn.relu(x)

            logits       = nn.Dense(self.action_dimension)(x)
            state_value  = nn.Dense(1)(x)
            return logits, jnp.squeeze(state_value, axis=-1)


    # 3) Initialise parameters, optimisers, RNGs
    # -------------------------------------------------------------------
    global_rng_key    = random.PRNGKey(0)
    network_parameters = {}
    optimizer_states   = {}
    rng_key_per_agent  = {}

    for agent_id in agent_list:
        global_rng_key, init_rng = random.split(global_rng_key)
        dummy_obs = jnp.zeros((1, *observation_shape), jnp.float32)
        params    = ActorCriticNetwork(action_dimension).init(init_rng, dummy_obs)
        opt       = optax.adam(LEARNING_RATE)
        opt_state = opt.init(params)

        network_parameters[agent_id] = params
        optimizer_states[agent_id]   = opt_state
        rng_key_per_agent[agent_id]  = init_rng


    # 4) Data-collection using raw dm_env API
    # -------------------------------------------------------------------
    def collect_trajectory_batch_per_agent(steps_per_agent=STEPS_PER_UPDATE):
        buffer = {
            agent: {
                "observations": [], "actions": [], "logp": [],
                "values": [],       "rewards": [], "metrics": []
            }
            for agent in agent_list
        }

        # start in a fresh episode
        timestep = env.reset()

        while len(buffer[primary_agent_id]["observations"]) < steps_per_agent:
            action_dict = {}

            # 1) for each agent: compute policy, sample, and store obs/action/logp/value
            for agent in agent_list:
                img = timestep.observation[f"{agent}.RGB"]
                x   = jnp.asarray(img, jnp.float32).transpose(2,0,1)[None,...]

                logits, value = ActorCriticNetwork(action_dimension).apply(
                    network_parameters[agent], x
                )
                dist = distrax.Categorical(logits=logits[0])
                rng_key_per_agent[agent], sub = random.split(rng_key_per_agent[agent])
                a    = dist.sample(seed=sub)
                logp = dist.log_prob(a)

                # map to primitive actions
                for name, val in ACTION_SET[int(a)].items():
                    action_dict[f"{agent}.{name}"] = val

                buf = buffer[agent]
                buf["observations"].append(x[0])
                buf["actions"].append(a)
                buf["logp"].append(logp)
                buf["values"].append(value[0])

            # 2) step the environment
            timestep = env.step(action_dict)

            # 3) extract all agents’ rewards once, then distribute
            reward_dict = get_multi_rewards(timestep)  # or _get_rewards(timestep)
            for agent in agent_list:
                r = reward_dict.get(agent, 0.0)
                buffer[agent]["rewards"].append(jnp.asarray(r, dtype=jnp.float32))

            # 4) any global metrics you care about
            buffer[primary_agent_id]["metrics"].append(
                np.array(timestep.observation["WORLD.WHO_ZAPPED_WHO"])
            )

            # 5) if episode ended, reset
            if timestep.last():
                timestep = env.reset()

        # 6) compute discounted returns
        for agent in agent_list:
            G, rets = 0.0, []
            for r in reversed(buffer[agent]["rewards"]):
                G = r + DISCOUNT_FACTOR * G
                rets.insert(0, G)
            buffer[agent]["returns"] = rets

        return buffer


    # 5) PPO loss and update
    # -------------------------------------------------------------------
    def compute_ppo_loss(params, obs, acts, old_logp, old_val, rets):
        logits, vals = ActorCriticNetwork(action_dimension).apply(params, obs)
        dist   = distrax.Categorical(logits=logits)
        logp   = dist.log_prob(acts)
        ratio  = jnp.exp(logp - old_logp)

        adv    = rets - old_val
        adv    = (adv - adv.mean()) / (adv.std() + 1e-8)

        unclipped = ratio * adv
        clipped   = jnp.clip(ratio, 1-PPO_CLIP_EPSILON, 1+PPO_CLIP_EPSILON) * adv

        policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
        value_loss  = jnp.mean((rets - vals)**2)
        return policy_loss + 0.5 * value_loss

    @jit
    def ppo_update_step(params, opt_state, obs, acts, old_logp, old_val, rets):
        loss, grads = value_and_grad(compute_ppo_loss)(
            params, obs, acts, old_logp, old_val, rets
        )
        updates, new_opt_state = optax.adam(LEARNING_RATE).update(
            grads, opt_state, params
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss


    # 6) Main training loop
    # -------------------------------------------------------------------
    reward_history = {agent: [] for agent in agent_list}
    best_reward    = {agent: -jnp.inf for agent in agent_list}
    no_improve_cnt = {agent: 0 for agent in agent_list}

    for update_idx in range(TOTAL_TRAINING_UPDATES):
        traj = collect_trajectory_batch_per_agent()

        # Example: print primary agent’s zap‐history
        hist = traj[primary_agent_id]["metrics"]
        print(f"[Update {update_idx}] {len(hist)} zap‐metric steps collected")
        for t, m in enumerate(hist):
            print(f" Step {t} zap‐matrix:\n{m}")

        # PPO updates
        for agent in agent_list:
            o  = jnp.stack(traj[agent]["observations"])
            a  = jnp.stack(traj[agent]["actions"])
            lp = jnp.stack(traj[agent]["logp"])
            v  = jnp.stack(traj[agent]["values"])
            R  = jnp.stack(traj[agent]["returns"])

            params, opt_state = network_parameters[agent], optimizer_states[agent]
            for _ in range(PPO_EPOCHS):
                params, opt_state, loss = ppo_update_step(
                    params, opt_state, o, a, lp, v, R
                )
            network_parameters[agent] = params
            optimizer_states[agent]   = opt_state

            # logging and early stopping bookkeeping
            avg_r = jnp.mean(jnp.stack(traj[agent]["rewards"]))
            reward_history[agent].append(avg_r)

            if avg_r > best_reward[agent] + EARLY_STOP_DELTA_THRESHOLD:
                best_reward[agent]    = avg_r
                no_improve_cnt[agent] = 0
            else:
                no_improve_cnt[agent] += 1

        if all(no_improve_cnt[a] >= EARLY_STOP_PATIENCE for a in agent_list):
            print(f"Early stopping at update {update_idx}")
            break


    # 7) Save checkpoints
    # -------------------------------------------------------------------
    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    from flax import serialization

    for agent, params in network_parameters.items():
        path = ckpt_dir / f"agent_{agent}_params.msgpack"
        with path.open("wb") as fp:
            fp.write(serialization.to_bytes(params))

if __name__ == "__main__":
    main()
