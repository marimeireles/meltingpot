# Copyright 2020 DeepMind Technologies Limited.
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
"""Melting Pot."""

import sys

from meltingpot import bot, scenario, substrate

try:
    # Keep `import meltingpot.python` working for external meltingpot.
    # TODO: b/292470900 - Remove in v3.0
    sys.modules["meltingpot.python"] = sys.modules["meltingpot"]
except KeyError:
    pass  # Internal version of meltingpot.
