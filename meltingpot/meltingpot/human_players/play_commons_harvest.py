# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for testing `commons_harvest`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `TAB` to switch between players.
Use `ENTER` to fire the death zapper.
"""

import argparse
import json

from ml_collections import config_dict

from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.human_players import level_playing_utils

environment_configs = {
    "commons_harvest__open": commons_harvest__open,
}

_ACTION_MAP = {
    "move": level_playing_utils.get_direction_pressed,
    "turn": level_playing_utils.get_turn_pressed,
    "fireZap": level_playing_utils.get_space_key_pressed,
    "deathZap": level_playing_utils.get_enter_key_pressed,
}


def verbose_fn(timestep, step, player_index):
    obs = timestep.observation
    fire_matrix = obs["WORLD.WHO_ZAPPED_WHO"]
    death_matrix = obs["WORLD.WHO_DEATH_ZAPPED_WHO"]
    print(f"[Frame {step}] FIRE zap matrix:\n{fire_matrix}")
    print(f"[Frame {step}] DEATH zap matrix:\n{death_matrix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level_name",
        type=str,
        default="commons_harvest__open",
        choices=environment_configs.keys(),
        help="Level name to load",
    )
    parser.add_argument(
        "--observation", type=str, default="RGB", help="Observation to render"
    )
    parser.add_argument(
        "--settings", type=json.loads, default={}, help="Settings as JSON string"
    )
    # Activate verbose mode with --verbose=True.
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Print   information"
    )
    # Activate events printing mode with --print_events=True.
    parser.add_argument("--print_events", type=bool, default=False, help="Print events")

    args = parser.parse_args()
    env_module = environment_configs[args.level_name]
    env_config = env_module.get_config()
    with config_dict.ConfigDict(env_config).unlocked() as env_config:
        roles = env_config.default_player_roles
        env_config.lab2d_settings = env_module.build(roles, env_config)
    level_playing_utils.run_episode(
        args.observation,
        args.settings,
        _ACTION_MAP,
        env_config,
        level_playing_utils.RenderType.PYGAME,
        verbose_fn=verbose_fn if args.verbose else None,
        print_events=args.print_events,
    )


if __name__ == "__main__":
    main()
