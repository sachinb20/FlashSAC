from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

try:
    import jax.numpy as jnp

    _JaxArray = jnp.ndarray
except ImportError:
    # jax is not installed in the isaaclab conda env (see flash_rl/envs/isaaclab_go2.py) --
    # it is only used here for a type-alias member, never imported by the torch-only path
    # (flash_rl/common, flash_rl/buffers, flash_rl/agents/flashSAC).
    _JaxArray = Any

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, _JaxArray, torch.Tensor]
