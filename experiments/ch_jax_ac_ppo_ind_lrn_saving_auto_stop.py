import gymnasium as gym
from shimmy import MeltingPotCompatibilityV0

import jax
import jax.numpy as jnp
from jax import random
from jax import jit, value_and_grad

import flax.linen as nn
import optax
import distrax

# ── Added for checkpointing and logging ─────────────────────────────────────────
import pathlib
from flax import serialization
# ────────────────────────────────────────────────────────────────────────────────

# --- Hyperparameters -----------------------------------------------------------
DISCOUNT_FACTOR            = 0.99
LEARNING_RATE              = 3e-4
PPO_CLIP_EPSILON           = 0.2
STEPS_PER_UPDATE           = 128
PPO_EPOCHS                 = 4
TOTAL_TRAINING_UPDATES     = 100
EARLY_STOP_DELTA_THRESHOLD = 1e-2   # minimum change in average reward to continue
EARLY_STOP_PATIENCE        = 10     # number of updates to wait for improvement
# ----------------------------------------------------------------------------

# 1) Create environment
environment     = MeltingPotCompatibilityV0(
    substrate_name="commons_harvest__open",
    render_mode="human"
)
agent_list       = environment.possible_agents
primary_agent_id = agent_list[0]
action_dimension = environment.action_space(primary_agent_id).n
rgb_observation_space = environment.observation_space(primary_agent_id)["RGB"]
obs_height, obs_width, obs_channels = rgb_observation_space.shape
observation_shape                  = (obs_channels, obs_height, obs_width)

# 2) Define convolutional actor‑critic network using Flax
class ActorCriticNetwork(nn.Module):
    action_dimension: int

    @nn.compact
    def __call__(self, observations):
        # observations: [batch, C, H, W] in 0‒255 range
        x = observations / 255.0
        x = nn.Conv(32, (8, 8), (4, 4))(x); x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x); x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1))(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x); x = nn.relu(x)

        action_logits = nn.Dense(self.action_dimension)(x)
        state_value   = nn.Dense(1)(x)
        return action_logits, jnp.squeeze(state_value, axis=-1)

# 3) Initialise network parameters, optimiser states, and RNG keys
global_rng_key     = random.PRNGKey(0)
network_parameters = {}
optimizer_states   = {}
rng_key_per_agent  = {}
for agent_id in agent_list:
    global_rng_key, init_rng_key = random.split(global_rng_key)
    dummy_input = jnp.zeros((1, *observation_shape), dtype=jnp.float32)
    params       = ActorCriticNetwork(action_dimension).init(init_rng_key, dummy_input)
    optimiser    = optax.adam(learning_rate=LEARNING_RATE)
    opt_state    = optimiser.init(params)

    network_parameters[agent_id] = params
    optimizer_states[agent_id]   = opt_state
    rng_key_per_agent[agent_id]  = init_rng_key

# 4) Data‑collection routine per update
def collect_trajectory_batch_per_agent(steps_per_agent: int = STEPS_PER_UPDATE):
    buffer = {
        agent: {"observations": [], "actions": [], "log_probabilities": [],
                 "value_estimates": [], "rewards": []}
        for agent in agent_list
    }
    obs_dict, _ = environment.reset()
    while len(buffer[primary_agent_id]["observations"]) < steps_per_agent:
        action_dict = {}
        for agent, raw in obs_dict.items():
            img = raw["RGB"]
            x   = jnp.asarray(img, dtype=jnp.float32).transpose(2, 0, 1)[None, ...]
            logits, v = ActorCriticNetwork(action_dimension).apply(
                network_parameters[agent], x
            )
            dist = distrax.Categorical(logits=logits[0])
            rng_key_per_agent[agent], subkey = random.split(rng_key_per_agent[agent])
            a    = dist.sample(seed=subkey)
            logp = dist.log_prob(a)

            action_dict[agent] = int(a)
            buf = buffer[agent]
            buf["observations"].append(x[0])
            buf["actions"].append(a)
            buf["log_probabilities"].append(logp)
            buf["value_estimates"].append(v[0])

        obs_dict_next, r_dict, term, trunc, _ = environment.step(action_dict)
        for agent in agent_list:
            buffer[agent]["rewards"].append(
                jnp.asarray(r_dict.get(agent, 0.0), dtype=jnp.float32)
            )
        if all(term.values()) or all(trunc.values()):
            obs_dict_next, _ = environment.reset()
        obs_dict = obs_dict_next

    # discounted returns
    for agent in agent_list:
        g       = 0.0
        returns = []
        for r in reversed(buffer[agent]["rewards"]):
            g = r + DISCOUNT_FACTOR * g
            returns.insert(0, g)
        buffer[agent]["returns"] = returns

    return buffer

# 5) PPO loss and update step
def compute_ppo_loss(params, obs, acts, old_logp, old_val, rets):
    logits, vals = ActorCriticNetwork(action_dimension).apply(params, obs)
    dist   = distrax.Categorical(logits=logits)
    logp   = dist.log_prob(acts)
    ratio  = jnp.exp(logp - old_logp)

    adv = rets - old_val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped   = jnp.clip(ratio, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * adv

    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss  = jnp.mean((rets - vals) ** 2)
    return policy_loss + 0.5 * value_loss

@jit
def ppo_update_step(params, opt_state, obs, acts, old_logp, old_val, rets):
    loss, grads = value_and_grad(compute_ppo_loss)(params, obs, acts, old_logp, old_val, rets)
    updates, new_opt_state = optax.adam(learning_rate=LEARNING_RATE).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# 6) Main training loop with early stopping and logging
# Initialise history trackers
reward_history = {agent: [] for agent in agent_list}
best_reward    = {agent: -jnp.inf for agent in agent_list}
no_improve_cnt = {agent: 0 for agent in agent_list}

for update_idx in range(TOTAL_TRAINING_UPDATES):
    traj = collect_trajectory_batch_per_agent()

    # Perform PPO updates per agent
    for agent in agent_list:
        o  = jnp.stack(traj[agent]["observations"])
        a  = jnp.stack(traj[agent]["actions"])
        lp = jnp.stack(traj[agent]["log_probabilities"])
        v  = jnp.stack(traj[agent]["value_estimates"])
        R  = jnp.stack(traj[agent]["returns"])

        params, opt_state = network_parameters[agent], optimizer_states[agent]
        for _ in range(PPO_EPOCHS):
            params, opt_state, loss = ppo_update_step(params, opt_state, o, a, lp, v, R)
        network_parameters[agent] = params
        optimizer_states[agent]   = opt_state

        # Compute average reward for this batch and log it
        avg_reward = jnp.mean(jnp.stack(traj[agent]["rewards"]))
        reward_history[agent].append(avg_reward)

        # Early stopping tracking
        if avg_reward > best_reward[agent] + EARLY_STOP_DELTA_THRESHOLD:
            best_reward[agent]    = avg_reward
            no_improve_cnt[agent] = 0
        else:
            no_improve_cnt[agent] += 1

    # Check if all agents have plateaued
    if all(no_improve_cnt[a] >= EARLY_STOP_PATIENCE for a in agent_list):
        print(f"Early stopping at update {update_idx}: no significant improvement for {EARLY_STOP_PATIENCE} updates.")
        break

# 7) Save a separate checkpoint for each trained agent
checkpoint_dir = pathlib.Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
for agent, params in network_parameters.items():
    ckpt_path = checkpoint_dir / f"agent_{agent}_params.msgpack"
    with ckpt_path.open("wb") as fp:
        fp.write(serialization.to_bytes(params))

