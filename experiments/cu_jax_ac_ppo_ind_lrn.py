import distrax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from jax import jit, random, value_and_grad
from shimmy import MeltingPotCompatibilityV0

# --- hyperparameters ---------------------------------------------------------
GAMMA = 0.99
LR = 3e-4
CLIP_EPS = 0.2
STEPS_PER_IT = 128
EPOCHS = 4
UPDATES = 100
# ----------------------------------------------------------------------------

# 1) Create environment
env = MeltingPotCompatibilityV0(substrate_name="clean_up", render_mode="human")
agents = env.possible_agents
agent0 = agents[0]
act_dim = env.action_space(agent0).n
img_space = env.observation_space(agent0)["RGB"]
H, W, C = img_space.shape
obs_shape = (C, H, W)


# 2) Define Conv-based actor-critic in Flax
class ConvPolicy(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        # x: [B, C, H, W], values 0–255
        x = x / 255.0
        x = nn.Conv(32, (8, 8), (4, 4))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), (2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), (1, 1))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.act_dim)(x)
        value = nn.Dense(1)(x)
        return logits, jnp.squeeze(value, axis=-1)


# 3) Initialize parameters, optimizers, RNGs
rng = random.PRNGKey(0)
nets = {}
opt_states = {}
rngs = {}

for agt in agents:
    # init params with dummy input [1, C, H, W]
    rng, key = random.split(rng)
    params = ConvPolicy(act_dim).init(key, jnp.zeros((1, *obs_shape), jnp.float32))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    nets[agt] = params
    opt_states[agt] = opt_state
    rngs[agt] = key


# 4) Data-collection routine
def collect_batch_per_agent(steps=STEPS_PER_IT):
    buffers = {
        agt: {"obs": [], "act": [], "logp": [], "val": [], "rew": []} for agt in agents
    }

    obs_dict, _ = env.reset()
    while len(buffers[agent0]["obs"]) < steps:
        action_dict = {}
        for agt, raw in obs_dict.items():
            img = raw["RGB"]  # H×W×C numpy
            x = jnp.array(img, dtype=jnp.float32)
            x = x.transpose(2, 0, 1)[None, ...]  # 1×C×H×W

            # sample action
            logits, v = ConvPolicy(act_dim).apply(nets[agt], x, mutable=False)
            dist = distrax.Categorical(logits=logits[0])
            rngs[agt], subkey = random.split(rngs[agt])
            a = dist.sample(seed=subkey)
            logp = dist.log_prob(a)

            action_dict[agt] = int(a)
            b = buffers[agt]
            b["obs"].append(x[0])
            b["act"].append(a)
            b["logp"].append(logp)
            b["val"].append(v[0])

        obs_dict, r_dict, term, trunc, _ = env.step(action_dict)
        for agt in agents:
            buffers[agt]["rew"].append(jnp.array(r_dict.get(agt, 0.0), jnp.float32))
        if all(term.values()) or all(trunc.values()):
            obs_dict, _ = env.reset()

    # compute returns
    for agt in agents:
        rews = buffers[agt]["rew"]
        returns = []
        G = 0.0
        for r in reversed(rews):
            G = r + GAMMA * G
            returns.insert(0, G)
        buffers[agt]["ret"] = returns

    return buffers


# 5) PPO loss and update step
def ppo_loss(params, obs, acts, old_logp, old_val, rets):
    logits, vals = ConvPolicy(act_dim).apply(params, obs)
    dist = distrax.Categorical(logits=logits)
    logp = dist.log_prob(acts)
    ratio = jnp.exp(logp - old_logp)

    adv = rets - old_val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = jnp.mean((rets - vals) ** 2)

    return policy_loss + 0.5 * value_loss


@jit
def update(params, opt_state, obs, acts, old_logp, old_val, rets):
    loss, grads = value_and_grad(ppo_loss)(params, obs, acts, old_logp, old_val, rets)
    updates, new_opt_state = optax.adam(LR).update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# 6) Main training loop
for _ in range(UPDATES):
    buffers = collect_batch_per_agent()

    for agt in agents:
        # stack into arrays
        o = jnp.stack(buffers[agt]["obs"])
        a = jnp.stack(buffers[agt]["act"])
        lp = jnp.stack(buffers[agt]["logp"])
        v_old = jnp.stack(buffers[agt]["val"])
        R = jnp.stack(buffers[agt]["ret"])

        params, opt_state = nets[agt], opt_states[agt]
        for _ in range(EPOCHS):
            params, opt_state, loss = update(params, opt_state, o, a, lp, v_old, R)
        nets[agt] = params
        opt_states[agt] = opt_state
