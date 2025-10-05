#Handles the output from the files
from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax.numpy as jnp
import numpy as np

@dataclass(frozen=True)
class Mechanism:
    # Only JAX-friendly leaves: jnp.ndarray, floats, ints, tuples, dicts of those.
    parameters: Dict[str, Any]   # geometry, masses, etc. as JAX arrays / Python scalars
    layout: Dict[str, Any]       # any indices/mappings as ints/tuples
    q0: jnp.ndarray              # (3N,)
    report: Dict[str, Any]       # metadata (counts, naming)

@dataclass(frozen=True)
class SolverConfig:
    newton_tol: float = 1e-10
    newton_max_iter: int = 50
    step_method: str = "kinematic"  # room to extend

@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray             # (T,)
    Qs: np.ndarray                # (T, 3N)
    Qdots: Optional[np.ndarray]   # (T, 3N) or None
    Qddots: Optional[np.ndarray]  # (T, 3N) or None
