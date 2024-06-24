"""SINN - Scale Invariant Neural Network for spg identification."""

__version__ = "1.0.0"

# model - neural network
from . import model

# datasets
from . import dataset

# training
from . import train

# layers
from . import layers

# graph
from . import graph


# cache directory
from pathlib import Path
import platformdirs

CACHE = platformdirs.user_cache_dir("sinn", "sinn")
