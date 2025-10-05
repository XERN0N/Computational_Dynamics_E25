# system_solver.py
import jax
import jax.numpy as jnp
from jax import lax

from system_functions import system_setup

jax.config.update("jax_enable_x64", True)


def _linear_solve(Jmat, rhs):
    """Solve J x = rhs. If square use solve; else min-norm with pinv."""
    m, n = Jmat.shape
    return jnp.linalg.solve(Jmat, rhs) if m == n else (jnp.linalg.pinv(Jmat) @ rhs)


def _newton_raphson_jit(function, jacobian, q0, t,
                        rtol=1e-8, atol=1e-12, max_steps=100, step_tol=None):
    """
    JIT-able Newton–Raphson (no line search).
    Uses lax.while_loop; returns (q, info_dict).
    """
    q0 = jnp.asarray(q0).reshape(-1)
    F0 = function(q0, t)
    res0 = jnp.linalg.norm(F0, jnp.inf)
    if step_tol is None:
        step_tol = rtol * (1.0 + jnp.linalg.norm(q0, jnp.inf))

    # Carry: (k, q, F, res0, step_norm, converged)
    carry0 = (jnp.array(0, dtype=jnp.int32),
              q0,
              F0,
              res0,
              jnp.array(jnp.inf, dtype=F0.dtype),
              jnp.array(False))

    def cond(carry):
        k, q, F, res0_, step_norm, converged = carry
        res_norm = jnp.linalg.norm(F, jnp.inf)
        stop_res = res_norm <= atol + rtol * (res0_ + 1.0)
        stop_step = step_norm <= jnp.maximum(step_tol, atol)
        done = jnp.logical_or(stop_res, stop_step)
        keep_iter = jnp.logical_and(~done, k < max_steps)
        return keep_iter

    def body(carry):
        k, q, F, res0_, _, _ = carry
        Jmat = jacobian(q, jnp.zeros_like(q), t)
        step = -_linear_solve(Jmat, F)
        q_new = q + step
        F_new = function(q_new, t)
        step_norm = jnp.linalg.norm(step, jnp.inf)

        # recompute convergence flags for next cond()
        res_norm_new = jnp.linalg.norm(F_new, jnp.inf)
        stop_res = res_norm_new <= atol + rtol * (res0_ + 1.0)
        stop_step = step_norm <= jnp.maximum(step_tol, atol)
        converged = jnp.logical_or(stop_res, stop_step)

        return (k + 1, q_new, F_new, res0_, step_norm, converged)

    k, q, F, res0, step_norm, converged = lax.while_loop(cond, body, carry0)
    info = {
        "converged": converged,                          # jnp.bool_
        "n_steps": k,                                    # jnp.int32
        "res_norm": jnp.linalg.norm(F, jnp.inf),         # jnp.float64
        "step_norm": step_norm,                          # jnp.float64
    }
    return q, info


def make_kinematic_solver(parameters, layout):
    """
    Build position/velocity/acceleration solvers using Φ, J, Nu, Γ,
    plus a JIT-able single-step and a simulate(times, q0).
    Returns: assemble, velocities, accelerations, step_kinematics, simulate, extras
    """
    Phi, J, Nu, Gamma, extras = system_setup(parameters, layout)

    # JIT-able assemble (Newton on Φ(q,t)=0)
    def assemble(q0, t, rtol=1e-8, atol=1e-12, max_steps=100, step_tol=None):
        return _newton_raphson_jit(Phi, J, q0, t, rtol=rtol, atol=atol,
                                   max_steps=max_steps, step_tol=step_tol)

    # JIT-able velocity/acceleration solves
    def velocities(q, t):
        Jmat = J(q, jnp.zeros_like(q), t)
        return _linear_solve(Jmat, Nu(q, t))

    def accelerations(q, qdot, t):
        Jmat = J(q, qdot, t)
        return _linear_solve(Jmat, Gamma(q, qdot, t))

    # One predictor–corrector step: (q,qdot,qddot,t, t_next) -> (q1,qdot1,qddot1)
    def step_kinematics(state, t_next):
        q, qdot, qddot, t = state
        dt = t_next - t
        q_pred = q + dt * qdot + 0.5 * (dt ** 2) * qddot
        q1, _info = assemble(q_pred, t_next)                # correct with Newton
        qdot1 = velocities(q1, t_next)
        qddot1 = accelerations(q1, qdot1, t_next)
        return (q1, qdot1, qddot1, t_next), (q1, qdot1, qddot1)

    # Simulate across an array of times using lax.scan
    def simulate(times, q0, *,
                 rtol=1e-8, atol=1e-12, max_steps=100, step_tol=None):
        """
        times: shape (T,), monotonically increasing
        q0: initial configuration consistent with Φ(q0, times[0]) ≈ 0
        returns (Qs, Qdots, Qddots)
        """
        t0 = times[0]
        q0_corr, _ = assemble(q0, t0, rtol=rtol, atol=atol, max_steps=max_steps, step_tol=step_tol)
        qdot0 = velocities(q0_corr, t0)
        qddot0 = accelerations(q0_corr, qdot0, t0)

        init_state = (q0_corr, qdot0, qddot0, t0)
        # scan over t[1:], stepping from each previous time to next
        def _scan_fn(state, t_next):
            return step_kinematics(state, t_next)

        _, traj = lax.scan(_scan_fn, init_state, times[1:])
        # prepend initial state to outputs
        q_init = q0_corr[None, :]
        qdot_init = qdot0[None, :]
        qddot_init = qddot0[None, :]
        Qs = jnp.vstack([q_init, traj[0]])
        Qdots = jnp.vstack([qdot_init, traj[1]])
        Qddots = jnp.vstack([qddot_init, traj[2]])
        return Qs, Qdots, Qddots

    # Optional JIT wrappers
    assemble_jit = jax.jit(assemble, static_argnames=("rtol", "atol", "max_steps", "step_tol"))
    velocities_jit = jax.jit(velocities)
    accelerations_jit = jax.jit(accelerations)
    step_kinematics_jit = jax.jit(step_kinematics)
    simulate_jit = jax.jit(simulate, static_argnames=("rtol", "atol", "max_steps", "step_tol"))

    # You can return the jitted versions by default; switch to non-jit if debugging prints
    return assemble_jit, velocities_jit, accelerations_jit, step_kinematics_jit, simulate_jit, extras
