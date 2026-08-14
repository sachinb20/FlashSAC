"""Simulator-agnostic Go2 environment shared by the Genesis and IsaacLab backends.

Import here must stay free of any engine import -- see ``sim_backend.make_backend``.
"""

from .go2_env import Go2WalkEnv, get_cfgs, get_env
from .sim_backend import Go2SimBackend, make_backend

__all__ = ["Go2WalkEnv", "Go2SimBackend", "get_cfgs", "get_env", "make_backend"]
