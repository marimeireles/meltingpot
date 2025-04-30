import gymnasium as gym
from shimmy import MeltingPotCompatibilityV0

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad

import flax.linen as nn
import optax
import distrax

# ── Added for checkpointing, logging, data export, and metrics ──────────────────
import pathlib
from flax import serialization
from datetime import datetime
import pandas as pd
import numpy as np

from tqdm import trange
# ───────────────────────────────────────────────────────────────────────────────

# --- Hyperparameters -----------------------------------------------------------
DISCOUNT_FACTOR     = 0.99
LEARNING_RATE       = 3e-4
PPO_CLIP_EPSILON    = 0.2
STEPS_PER_UPDATE    = 128
PPO_EPOCHS          = 4
TOTAL_UPDATES       = 100

# KL-based early stopping
KL_THRESHOLD        = 0.005
KL_PATIENCE         = 10
# ----------------------------------------------------------------------------

# 1) Create environment
env = MeltingPotCompatibilityV0(
    substrate_name="commons_harvest__open",
    render_mode="rgb_array"
)
agents        = env.possible_agents
primary_agent = agents[0]
action_dim    = env.action_space(primary_agent).n
rgb_space     = env.observation_space(primary_agent)["RGB"]
H, W, C       = rgb_space.shape
obs_shape     = (C, H, W)

# 2) Define convolutional actor-critic network
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = nn.Conv(32, (8,8), (4,4))(x); x = nn.relu(x)
        x = nn.Conv(64, (4,4), (2,2))(x); x = nn.relu(x)
        x = nn.Conv(64, (3,3), (1,1))(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x); x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        value  = nn.Dense(1)(x)
        return logits, jnp.squeeze(value, -1)

# 3) Initialization
rng                 = random.PRNGKey(0)
params              = {}
optim_states        = {}
rngs                = {}
# metrics histories
gini_history        = []
resource_stock_hist = []
harvest_rate_hist   = {a: [] for a in agents}
cumulative_rewards  = []
spatial_pos_hist    = {a: [] for a in agents}

for a in agents:
    rng, init_rng = random.split(rng)
    dummy         = jnp.zeros((1, *obs_shape), jnp.float32)
    p             = ActorCritic(action_dim).init(init_rng, dummy)
    optim         = optax.adam(LEARNING_RATE)
    state         = optim.init(p)
    params[a]     = p
    optim_states[a] = state
    rngs[a]       = init_rng

# utility: compute Gini coefficient
def compute_gini(array):
    arr = np.array(array, dtype=np.float64)
    if arr.size == 0 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n+1)
    return ((2*index - n - 1) * arr).sum() / (n * arr.sum())

# 4) Trajectory collection with metrics
def collect_batch(n_steps=STEPS_PER_UPDATE):
    buffer = {a: {"obs": [], "acts": [], "logps": [], "vals": [], "rews": []} for a in agents}
    stock_hist = []
    harvests   = {a: [] for a in agents}
    positions  = {a: [] for a in agents}

    obs_dict, _ = env.reset()
    steps = 0
    while steps < n_steps:
        # get global resource stock if available
        info = {}
        try:
            _, _, _, _, info = env.step({})  # dummy to fetch info
            stock = info.get('resources', None)
        except:
            stock = None
        stock_hist.append(stock)

        actions = {}
        for a, obs in obs_dict.items():
            img = jnp.array(obs["RGB"], jnp.float32).transpose(2,0,1)[None]
            logits, v   = ActorCritic(action_dim).apply(params[a], img)
            dist        = distrax.Categorical(logits=logits[0])
            rngs[a], sk = random.split(rngs[a])
            act         = dist.sample(seed=sk)
            actions[a]  = int(act)
            buffer[a]["obs"].append(img[0])
            buffer[a]["acts"].append(act)
            buffer[a]["logps"].append(dist.log_prob(act))
            buffer[a]["vals"].append(v[0])

        nxt, rewards, term, trunc, info = env.step(actions)
        for a in agents:
            r = float(rewards.get(a, 0.0))
            buffer[a]["rews"].append(r)
            harvests[a].append(r)
            pos = obs_dict[a].get('position') if 'position' in obs_dict[a] else None
            positions[a].append(pos)
        steps += 1
        if all(term.values()) or all(trunc.values()):
            nxt, _ = env.reset()
        obs_dict = nxt

    # compute returns
    for a in agents:
        G = 0.0; returns = []
        for r in reversed(buffer[a]["rews"]):
            G = r + DISCOUNT_FACTOR * G
            returns.insert(0, G)
        buffer[a]["rets"] = returns

    return buffer, stock_hist, harvests, positions

# 5) PPO loss and update
def ppo_loss(params, obs, acts, old_logp, old_val, returns):
    logits, vals = ActorCritic(action_dim).apply(params, obs)
    dist   = distrax.Categorical(logits=logits)
    logp   = dist.log_prob(acts)
    ratio  = jnp.exp(logp - old_logp)
    adv    = returns - old_val
    adv    = (adv - adv.mean()) / (adv.std() + 1e-8)
    unclp  = ratio * adv
    clp    = jnp.clip(ratio, 1-PPO_CLIP_EPSILON, 1+PPO_CLIP_EPSILON) * adv
    pol_l  = -jnp.mean(jnp.minimum(unclp, clp))
    val_l  = jnp.mean((returns - vals)**2)
    return pol_l + 0.5 * val_l

@jit
def update_step(params, optim_state, obs, acts, old_logp, old_val, returns):
    loss, grads = value_and_grad(ppo_loss)(params, obs, acts, old_logp, old_val, returns)
    updates, new_state = optax.adam(LEARNING_RATE).update(grads, optim_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, loss

# 6) Training loop with KL-based early stopping, metrics, and progress bar
kl_counter    = {a: 0 for a in agents}
run_id        = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = pathlib.Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Wrap updates loop with tqdm progress bar
for update in trange(TOTAL_UPDATES, desc="PPO updates"):
    batch, stock_hist, harvests, positions = collect_batch()
    resource_stock_hist.extend(stock_hist)

    # record per-agent harvest rates
    for a in agents:
        harvest_rate_hist[a].extend(harvests[a])

    # compute cumulative returns and Gini
    cum_returns = [sum(harvest_rate_hist[a]) for a in agents]
    cumulative_rewards.append(cum_returns)
    gini_history.append(compute_gini(cum_returns))

    for a in agents:
        o       = jnp.stack(batch[a]["obs"])
        ac      = jnp.stack(batch[a]["acts"])
        old_lp  = jnp.stack(batch[a]["logps"])
        old_val = jnp.stack(batch[a]["vals"])
        rets    = jnp.stack(batch[a]["rets"])

        old_params = params[a]
        p, s = params[a], optim_states[a]
        # collect losses over PPO epochs
        losses = []
        for _ in range(PPO_EPOCHS):
            p, s, loss = update_step(p, s, o, ac, old_lp, old_val, rets)
            losses.append(float(loss))
        params[a], optim_states[a] = p, s

        # KL divergence and logging
        old_logits, _ = ActorCritic(action_dim).apply(old_params, o)
        new_logits, _ = ActorCritic(action_dim).apply(p, o)
        dist_old = distrax.Categorical(logits=old_logits)
        dist_new = distrax.Categorical(logits=new_logits)
        kl = jnp.mean(dist_old.kl_divergence(dist_new))
        kl_counter[a] = kl_counter[a] + 1 if kl < KL_THRESHOLD else 0

        print(f"Agent {a} update {update}: mean KL={kl:.6f}, patience={kl_counter[a]}")

    if all(count >= KL_PATIENCE for count in kl_counter.values()):
        print(f"Early stopping at update {update}: KL < {KL_THRESHOLD} for {KL_PATIENCE} updates.")
        break

# 7) Save model and raw data
for a, p in params.items():
    path = checkpoint_dir / f"{run_id}_agent_{a}_params.msgpack"
    with path.open("wb") as f:
        f.write(serialization.to_bytes(p))

# Export CSVs
df_stock = pd.DataFrame({'resource_stock': resource_stock_hist})
df_stock.to_csv(checkpoint_dir / f"{run_id}_resource_stock.csv", index_label='step')

df_harvest = pd.DataFrame(harvest_rate_hist)
df_harvest.to_csv(checkpoint_dir / f"{run_id}_harvest_events.csv", index_label='step')

for a in agents:
    df_pos = pd.DataFrame(spatial_pos_hist[a], columns=['x','y'])
    df_pos.to_csv(checkpoint_dir / f"{run_id}_positions_agent_{a}.csv", index_label='step')
