# system_solver.py
# High-level kinematics runner using JAX.
# - Newton assembly: lax.while_loop (pure JAX)
# - Time marching:   lax.scan (pure JAX)
# - Methods are jit-compiled with self marked static.
#
# Depends on `system_functions.py` providing:
#   SystemFunctions -> returns jitted (Phi, J, Nu, Gamma) closures and extras.

from __future__ import annotations
from functools import partial
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jax import lax

from system_functions import SystemFunctions


jax.config.update("jax_enable_x64", True)


class KinematicSolver:
    """
    Kinematic solver that wraps constraint functions (Φ, J, Nu, Γ) and
    provides:
      - assemble(q_init, t): Newton solve Φ(q,t)=0
      - velocities(q, t):    solve J q̇ = Nu
      - accelerations(q,q̇,t): solve J q̈ = Γ
      - step(carry, t):      one time sample (assemble, then q̇, q̈)
      - simulate(times, q0): run over time vector with lax.scan

    All public methods are JIT-compiled with `static_argnums=0` so `self`
    is static and not traced.
    """

    def __init__(self,
                 parameters: Dict[str, Any],
                 layout: Dict[str, Any],
                 config: Dict[str, Any] | None = None):
        sf = SystemFunctions(parameters, layout)
        self._Phi, self._J, self._Nu, self._Gamma, self.extras = sf.as_tuple()

        self.num_coord = int(self.extras["num_coord"])

        # Config (Python scalars captured as static in jitted methods)
        cfg = {} if config is None else dict(config)
        self.newton_tol: float = float(cfg.get("newton_tol", 1e-10))
        self.newton_max: int   = int(cfg.get("newton_max", 25))

    # ----------------- linear least squares helper (pure JAX) -----------------
    @staticmethod
    def _lstsq(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """
        Solve min_x ||A x - b||_2. Works for non-square/ill-conditioned J too.
        """
        x, *_ = jnp.linalg.lstsq(A, b, rcond=None)
        return x

    # ----------------- Newton assembly (pure JAX) -----------------
    @partial(jax.jit, static_argnums=0)
    def assemble(self, q_init: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.int32, jnp.ndarray]]:
        """
        Solve Φ(q, t) = 0 starting from q_init using Newton's method.
        Uses lax.while_loop, so it is traceable & JIT-friendly.

        Returns:
            qN         : assembled coordinates (3N,)
            (kN, resN) : iteration count and final residual vector (m,)
        """
        tol   = self.newton_tol
        kmax  = jnp.int32(self.newton_max)

        def cond_fun(carry):
            q, k = carry
            res  = self._Phi(q, t)
            infn = jnp.linalg.norm(res, ord=jnp.inf)
            return jnp.logical_and(infn > tol, k < kmax)

        def body_fun(carry):
            q, k = carry
            Phi  = self._Phi(q, t)                      # (m,)
            J    = self._J(q, jnp.zeros_like(q), t)     # (m, 3N)
            dq   = self._lstsq(J, -Phi)
            return (q + dq, k + jnp.int32(1))

        q0 = q_init
        k0 = jnp.int32(0)
        qN, kN = lax.while_loop(cond_fun, body_fun, (q0, k0))
        resN = self._Phi(qN, t)
        return qN, (kN, resN)

    # ----------------- velocities / accelerations (pure JAX) -----------------
    @partial(jax.jit, static_argnums=0)
    def velocities(self, q: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Solve J(q,t) q̇ = Nu(q,t).
        """
        J   = self._J(q, jnp.zeros_like(q), t)  # (m,3N)
        Nu  = self._Nu(q, t)                    # (m,)
        qdot = self._lstsq(J, Nu)
        return qdot

    @partial(jax.jit, static_argnums=0)
    def accelerations(self, q: jnp.ndarray, qdot: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Solve J(q,t) q̈ = Γ(q,q̇,t).
        """
        J     = self._J(q, qdot, t)     # (m,3N)
        Gamma = self._Gamma(q, qdot, t) # (m,)
        qddot = self._lstsq(J, Gamma)
        return qddot

    # ----------------- one step at time t (pure JAX) -----------------
    @partial(jax.jit, static_argnums=0)
    def step(self,
             carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
             t: jnp.ndarray) -> Tuple[
                 Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
             ]:
        """
        carry = (q_prev, qdot_prev, qddot_prev)  (previous results; not strictly required)
        returns new_carry, (q, qdot, qddot) at time t
        """
        q_prev, qdot_prev, qddot_prev = carry
        q, _   = self.assemble(q_prev, t)
        qdot   = self.velocities(q, t)
        qddot  = self.accelerations(q, qdot, t)
        new_c  = (q, qdot, qddot)
        return new_c, (q, qdot, qddot)

    # ----------------- full simulate over time vector (pure JAX) -----------------
    @partial(jax.jit, static_argnums=0)
    def simulate(self, times: jnp.ndarray, q0: jnp.ndarray):
        """
        times: (T,) JAX array
        q0   : (3N,)
        Returns: Qs, Qdots, Qddots with shape (T, 3N)
        """
        times = jnp.asarray(times)
        t0    = times[0]

        # Assemble and compute kinematics at initial time
        q0_corr, _ = self.assemble(q0, t0)
        qdot0      = self.velocities(q0_corr, t0)
        qddot0     = self.accelerations(q0_corr, qdot0, t0)

        carry0 = (q0_corr, qdot0, qddot0)

        # Pure JAX scan over the remaining times; works even if times[1:] has length 0
        def scan_step(carry, t):
            return self.step(carry, t)

        carryN, (Qs_tail, Qdots_tail, Qddots_tail) = lax.scan(
            scan_step, carry0, times[1:]
        )

        # Prepend initial sample
        Qs     = jnp.vstack([q0_corr[None, :], Qs_tail])
        Qdots  = jnp.vstack([qdot0 [None, :], Qdots_tail])
        Qddots = jnp.vstack([qddot0[None, :], Qddots_tail])

        return Qs, Qdots, Qddots


# --------- small convenience factory used by runner.py ----------
def make_kinematic_solver(parameters: Dict[str, Any],
                          layout: Dict[str, Any],
                          config: Dict[str, Any] | None = None) -> KinematicSolver:
    """
    Factory that matches the call site in runner.py:
        solver = make_kinematic_solver(mpack.parameters, mpack.layout)
    """
    return KinematicSolver(parameters, layout, config)
