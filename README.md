# CM3070 Rocket Lander Project

This repository contains the code for a rocket landing simulation using both Finite State Machine (FSM) and Proximal Policy Optimization (PPO) models. The project demonstrates the implementation and comparison of these models in controlling the landing of a rocket on a platform within a simulated environment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting Up the Virtual Environment](#setting-up-the-virtual-environment)
  - [Linux/macOS](#linuxmacos)
  - [Windows](#windows)
- [Installing Dependencies](#installing-dependencies)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

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
python run_fsm.py
