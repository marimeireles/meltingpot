# Conan

Reimplementation of general-sum environments in meltingpot that allows for agents to be killed by agents who retain less resources than them.

This repository also provides an implementation of agents and the supporting functions to run experiments.

## Installation

Install python 3.11 with your favorite package manager and create an uv environment within it.

Install meltingpot locally with:

```bash
uv pip install -e .
```

Install the rest of the dependencies:

```bash
uv pip install tqdm matplotlib shimmy gymnasium flax distrax wandb
```

## Organizational

Linting the repository:

```bash
uv pip install pyink isort
```

```bash
pyink . && isort .
```

---

*Mongol General: Hao! Dai ye! We won again! This is good, but what is best in life?*

*Mongol: The open steppe, fleet horse, falcons at your wrist, and the wind in your hair.*

*Mongol General: Wrong! Conan! What is best in life?*

*Conan: To crush your enemies, see them driven before you, and to hear the lamentations of their women.*

*Mongol General: That is good! That is good.*
