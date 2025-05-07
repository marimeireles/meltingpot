from shimmy import MeltingPotCompatibilityV0
from supersuit import color_reduction_v0, frame_stack_v1

env = MeltingPotCompatibilityV0(substrate_name="commons_harvest__open", render_mode="human")
env = frame_stack_v1(color_reduction_v0(env, 'full'), 4)

observations = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.step(actions)
env.close()