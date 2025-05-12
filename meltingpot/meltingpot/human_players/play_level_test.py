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
"""Tests for human_players."""

from unittest import mock

import pygame
from absl.testing import absltest, parameterized
from ml_collections import config_dict

from meltingpot.configs.substrates import (
    allelopathic_harvest__open,
    boat_race__eight_races,
    chemistry__three_metabolic_cycles,
    chemistry__three_metabolic_cycles_with_plentiful_distractors,
    chemistry__two_metabolic_cycles,
    chemistry__two_metabolic_cycles_with_distractors,
    clean_up,
    coins,
    collaborative_cooking__asymmetric,
    commons_harvest__closed,
    coop_mining,
    daycare,
    externality_mushrooms__dense,
    factory_commons__either_or,
    fruit_market__concentric_rivers,
    gift_refinements,
    paintball__capture_the_flag,
    paintball__king_of_the_hill,
    predator_prey__alley_hunt,
    predator_prey__orchard,
    prisoners_dilemma_in_the_matrix__arena,
    territory__rooms,
)
from meltingpot.human_players import (
    level_playing_utils,
    play_allelopathic_harvest,
    play_anything_in_the_matrix,
    play_boat_race,
    play_chemistry,
    play_clean_up,
    play_coins,
    play_collaborative_cooking,
    play_commons_harvest,
    play_coop_mining,
    play_daycare,
    play_externality_mushrooms,
    play_factory_commons,
    play_fruit_market,
    play_gift_refinements,
    play_paintball,
    play_predator_and_prey,
    play_territory,
)


class PlayLevelTest(parameterized.TestCase):

    @parameterized.named_parameters(
        (
            "allelopathic_harvest__open",
            allelopathic_harvest__open,
            play_allelopathic_harvest,
        ),
        ("boat_race__eight_races", boat_race__eight_races, play_boat_race),
        (
            "chemistry__three_metabolic_cycles",
            chemistry__three_metabolic_cycles,
            play_chemistry,
        ),
        (
            "chemistry__three_metabolic_cycles_with_plentiful_distractors",
            chemistry__three_metabolic_cycles_with_plentiful_distractors,
            play_chemistry,
        ),
        (
            "chemistry__two_metabolic_cycles",
            chemistry__two_metabolic_cycles,
            play_chemistry,
        ),
        (
            "chemistry__two_metabolic_cycles_with_distractors",
            chemistry__two_metabolic_cycles_with_distractors,
            play_chemistry,
        ),
        ("clean_up", clean_up, play_clean_up),
        ("coins", coins, play_coins),
        (
            "collaborative_cooking__asymmetric",
            collaborative_cooking__asymmetric,
            play_collaborative_cooking,
        ),
        (
            "commons_harvest__closed",
            commons_harvest__closed,
            play_commons_harvest,
        ),
        ("coop_mining", coop_mining, play_coop_mining),
        ("daycare", daycare, play_daycare),
        (
            "externality_mushrooms__dense",
            externality_mushrooms__dense,
            play_externality_mushrooms,
        ),
        (
            "factory_commons__either_or",
            factory_commons__either_or,
            play_factory_commons,
        ),
        (
            "fruit_market__concentric_rivers",
            fruit_market__concentric_rivers,
            play_fruit_market,
        ),
        ("gift_refinements", gift_refinements, play_gift_refinements),
        (
            "paintball__capture_the_flag",
            paintball__capture_the_flag,
            play_paintball,
        ),
        (
            "paintball__king_of_the_hill",
            paintball__king_of_the_hill,
            play_paintball,
        ),
        (
            "predator_prey__alley_hunt",
            predator_prey__alley_hunt,
            play_predator_and_prey,
        ),
        (
            "predator_prey__orchard",
            predator_prey__orchard,
            play_predator_and_prey,
        ),
        (
            "prisoners_dilemma_in_the_matrix__arena",
            prisoners_dilemma_in_the_matrix__arena,
            play_anything_in_the_matrix,
        ),
        ("territory__rooms", territory__rooms, play_territory),
    )
    @mock.patch.object(pygame, "key")
    @mock.patch.object(pygame, "display")
    @mock.patch.object(pygame, "event")
    @mock.patch.object(pygame, "time")
    def test_run_level(
        self, config_module, play_module, unused_k, unused_d, unused_e, unused_t
    ):
        env_module = config_module
        env_config = env_module.get_config()

        with config_dict.ConfigDict(env_config).unlocked() as env_config:
            roles = env_config.default_player_roles
            env_config.lab2d_settings = env_module.build(roles, env_config)

        env_config["lab2d_settings"]["maxEpisodeLengthFrames"] = 10
        level_playing_utils.run_episode("RGB", {}, play_module._ACTION_MAP, env_config)


if __name__ == "__main__":
    absltest.main()
