# runner.py
# Glue code: build -> solve -> return a Trajectory for plotting.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from system_solver import make_kinematic_solver

# We keep a lightweight wrapper so main.py doesn't juggle tuples
@dataclass(frozen=True)
class MechanismPack:
    parameters: dict
    layout: dict
    q0: jnp.ndarray
    report: dict

@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray         # (T,)
    Qs: np.ndarray            # (T, 3N)
    Qdots: Optional[np.ndarray] = None
    Qddots: Optional[np.ndarray] = None

def _to_numpy(x):
    return np.asarray(x) if x is not None else None

def run_kinematics(build_mechanism: Callable[[], "Mechanism"],  # returns types.Mechanism dataclass
                   times: np.ndarray,
                   *,
                   solver_kwargs: Optional[dict] = None):
    """
    Build the mechanism, run the kinematic simulation, and return (mpack, traj, info).
    Compatible with the new KinematicSolver instance returned by make_kinematic_solver().
    """
    solver_kwargs = solver_kwargs or {}

    # 1) Build mechanism (dataclass)
    mech = build_mechanism()  # types.Mechanism
    mpack = MechanismPack(parameters=mech.parameters, layout=mech.layout, q0=mech.q0, report=mech.report)

    # 2) Build solver instance
    solver = make_kinematic_solver(mpack.parameters, mpack.layout)

    # 3) Assemble at t0 (for info) — safe even if simulate also assembles
    t0 = float(times[0])
    q0_corr, info = solver.assemble(mpack.q0, t0)

    # 4) Simulate whole trajectory
    Qs, Qdots, Qddots = solver.simulate(jnp.asarray(times), q0_corr)

    traj = Trajectory(
        times=_to_numpy(times),
        Qs=_to_numpy(Qs),
        Qdots=_to_numpy(Qdots),
        Qddots=_to_numpy(Qddots),
    )
    return mpack, traj, info
