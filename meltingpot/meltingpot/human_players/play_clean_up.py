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
"""A simple human player for testing `clean_up`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `TAB` to switch between players.
Use `ENTER` to fire the death zapper.
"""

import argparse
import json

from ml_collections import config_dict

from meltingpot.configs.substrates import clean_up
from meltingpot.human_players import level_playing_utils

environment_configs = {
    "clean_up": clean_up,
}

_ACTION_MAP = {
    "move": level_playing_utils.get_direction_pressed,
    "turn": level_playing_utils.get_turn_pressed,
    "deathZap": level_playing_utils.get_enter_key_pressed,
    "fireZap": level_playing_utils.get_key_number_one_pressed,
    "fireClean": level_playing_utils.get_key_number_two_pressed,
}


def verbose_fn(env_timestep, player_index, current_player_index):
    """Print using this function once enabling the option --verbose=True."""
    lua_index = player_index + 1
    cleaned = env_timestep.observation[f"{lua_index}.PLAYER_CLEANED"]
    ate = env_timestep.observation[f"{lua_index}.PLAYER_ATE_APPLE"]
    num_zapped_this_step = env_timestep.observation[
        f"{lua_index}.NUM_OTHERS_PLAYER_ZAPPED_THIS_STEP"
    ]
    num_others_cleaned = env_timestep.observation[
        f"{lua_index}.NUM_OTHERS_WHO_CLEANED_THIS_STEP"
    ]
    num_others_ate = env_timestep.observation[
        f"{lua_index}.NUM_OTHERS_WHO_ATE_THIS_STEP"
    ]
    # Only print observations from current player.
    if player_index == current_player_index:
        print(
            f"player: {player_index} --- player_cleaned: {cleaned} --- "
            + f"player_ate_apple: {ate} --- num_others_cleaned: "
            + f"{num_others_cleaned} --- num_others_ate: {num_others_ate} "
            + f"---num_others_player_zapped_this_step: {num_zapped_this_step}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level_name",
        type=str,
        default="clean_up",
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
        "--verbose", type=bool, default=False, help="Print debug information"
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
