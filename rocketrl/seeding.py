"""Global seeding so every run is reproducible from a single integer."""

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch. Env seeding is handled separately by
    passing the seed into env.reset(seed=...) / the SB3 model seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
