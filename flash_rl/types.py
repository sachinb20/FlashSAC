from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

# JAX is only needed to widen the `Tensor` alias; nothing on the torch path calls into it.
# Keeping the import optional lets torch-only setups (e.g. an IsaacSim environment where
# installing JAX would drag numpy along with it) run without pulling JAX in.
try:
    import jax.numpy as jnp

    _JAX_ARRAY: Any = jnp.ndarray
except ImportError:  # pragma: no cover - depends on the environment, not the code path
    _JAX_ARRAY = np.ndarray

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, _JAX_ARRAY, torch.Tensor]
