# CM3070 Rocket Lander Project

[![CI](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml/badge.svg?branch=showcase-rebuild)](https://github.com/adityaanilshukla/CM3070-Final-Project/actions/workflows/ci.yml)

This repository contains the code for a rocket landing simulation using both Finite State Machine (FSM) and Proximal Policy Optimization (PPO) models. The project demonstrates the implementation and comparison of these models in controlling the landing of a rocket on a platform within a simulated environment.

## Landings

One fixed-seed episode per controller, same start condition, rendered with `rocketrl/record_gifs.py`.

<table>
  <tr>
    <th align="center">SAC</th>
    <th align="center">PPO</th>
    <th align="center">Classical (cascaded PID)</th>
  </tr>
  <tr>
    <td align="center" width="33%"><em>TODO: sac.gif</em></td>
    <td align="center" width="33%"><em>TODO: ppo.gif</em></td>
    <td align="center" width="33%"><img src="assets/gifs/classical.gif" width="220" alt="Classical cascaded-PID controller landing"></td>
  </tr>
  <tr>
    <td align="center">Not yet trained to reliable landings</td>
    <td align="center">Not yet trained to reliable landings</td>
    <td align="center">seed 6, a clean landing (this cascaded-PID controller lands ~48% of episodes)</td>
  </tr>
</table>

> **TODO:** SAC and PPO are still training; their GIFs will be added here once those agents land reliably. Do not confuse the `assets/gifs/random.gif` baseline (generated alongside `classical.gif` to smoke-test the recorder) for a trained controller.

Regenerate (or add the RL controllers once trained) with:

```bash
# classical + random baselines (what is shown above)
uv run python -m rocketrl.record_gifs --classical --random --seed 3 --out-dir assets/gifs

# once SAC / PPO checkpoints exist, e.g. under runs/sac_s0 and runs/ppo_s0:
uv run python -m rocketrl.record_gifs --sac-run runs/sac_s0 --ppo-run runs/ppo_s0 --seed 3 --out-dir assets/gifs
```

## Table of Contents

- [Landings](#landings)
- [Prerequisites](#prerequisites)
- [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Linux/macOS](#linuxmacos)
  - [Windows](#windows)
- [Installing Dependencies](#installing-dependencies)
- [Running the Project](#running-the-project)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7** is installed on your system.
- **pip** (Python package manager) is installed.

## Setting Up the Virtual Environment

### Linux/macOS

1. **Open a terminal**.

2. **Navigate to the project directory**:

    ```bash
    cd /path/to/project
    ```

3. **Create a virtual environment**:

    ```bash
    python3.7 -m venv rocket-lander-env
    ```

4. **Activate the virtual environment**:

    ```bash
    source rocket-lander-env/bin/activate
    ```

### Windows

1. **Open Command Prompt**.

2. **Navigate to the project directory**:

    ```cmd
    cd \path\to\project
    ```

3. **Create a virtual environment**:

    ```cmd
    python -m venv rocket-lander-env-windows
    ```

4. **Activate the virtual environment**:

    ```cmd
    rocket-lander-env-windows\Scripts\activate
    ```

## Installing Dependencies

Once the virtual environment is activated, you need to install the required packages. 

### Linux/macOS/Windows

1. **Ensure you are in the project directory**.

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    If you encounter issues related to `gym`, you may need to downgrade the `setuptools`:

    ```bash
    pip install setuptools==65.5.0
    pip install -r requirements.txt
    ```

## Running the Project

### Running the FSM Model

To run the Finite State Machine (FSM) model:

```bash
python run_fsm_model.py
```
### Running the RL model 

```bash
python run_ppo_model.py
```
