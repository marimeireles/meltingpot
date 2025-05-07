from shimmy import MeltingPotCompatibilityV0

env = MeltingPotCompatibilityV0(substrate_name="commons_harvest__open", render_mode="human")

observations = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()