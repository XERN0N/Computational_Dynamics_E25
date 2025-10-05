# system_functions.py
# Planar multibody constraint assembly following Haug (Φ, J, Nu, Γ).
# Compatible with MechanismBuilder.compile() from preprocessor.py.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Any
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ------------------------------------------------------------
# Core builder: returns jittable closures (Phi, J, Nu, Gamma)
# ------------------------------------------------------------
def system_setup(
    parameters: Dict,
    layout: Dict,
    *,
    jacobian_mode: str = "autodiff",
) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    """
    Build pure-JAX closures describing the mechanism constraints.

    Returns:
        Phi(q,t)              -> (m,)
        J(q,qdot,t)=∂Φ/∂q     -> (m, 3N)
        Nu(q,t)   = -Φ_t      -> (m,)
        Gamma(q,qdot,t)       -> (m,)
        extras: dict with sizes + pack/unpack helpers
    """

    # ===== Unpack arrays from preprocessor =====
    joint_type_id = jnp.asarray(parameters["joint_type_id"])
    joint_i       = jnp.asarray(parameters["joint_i"])
    joint_j       = jnp.asarray(parameters["joint_j"])
    si_xy         = jnp.asarray(parameters["si_xy"])
    sj_xy         = jnp.asarray(parameters["sj_xy"])

    types = parameters.get("types", {})
    rev_c_xy   = jnp.asarray(types.get("revolute",    {}).get("c_xy",     jnp.zeros_like(si_xy)))
    dist_d     = jnp.asarray(types.get("distance",    {}).get("d",        jnp.zeros((joint_type_id.shape[0],))))
    trans_ui_l = jnp.asarray(types.get("translation", {}).get("ui_local", jnp.zeros_like(si_xy)))
    trans_uj_l = jnp.asarray(types.get("translation", {}).get("uj_local", jnp.zeros_like(si_xy)))

    # ===== Drivers =====
    drv = parameters.get("drivers", {})
    drv_target_id = jnp.asarray(drv.get("target_id",   jnp.zeros((0,), dtype=jnp.int32)))   # 0:coord, 1:angle
    drv_i         = jnp.asarray(drv.get("i",           jnp.zeros((0,), dtype=jnp.int32)))
    drv_j         = jnp.asarray(drv.get("j",           jnp.zeros((0,), dtype=jnp.int32)))
    drv_coord_idx = jnp.asarray(drv.get("coord_index", jnp.zeros((0,), dtype=jnp.int32)))

    sig           = drv.get("signal", {})
    # 0: sin, 1: cos, 2: linear
    sig_kind_id   = jnp.asarray(sig.get("kind_id",     jnp.zeros((0,), dtype=jnp.int32)))
    # params layout: (A, omega, phi, bias); for linear, A := rate
    float_dtype   = layout.get("float_dtype", jnp.float64)
    sig_params    = jnp.asarray(sig.get("params",      jnp.zeros((sig_kind_id.shape[0], 4), dtype=float_dtype)))
    D             = int(sig_kind_id.shape[0])

    # ===== Layout and sizes =====
    row_counts  = jnp.asarray(layout["row_counts"])
    row_offsets = jnp.asarray(layout["row_offset"])
    num_rows    = int(layout["num_rows"])
    num_coord   = int(layout["num_coord"])

    joint_ids = layout.get("joint_ids", {"revolute": 0, "distance": 1, "translation": 2})
    REV = int(joint_ids.get("revolute", 0))
    DIS = int(joint_ids.get("distance", 1))
    TRA = int(joint_ids.get("translation", 2))

    K = int(joint_type_id.shape[0])

    joint_row_counts = row_counts[:K] if K > 0 else jnp.zeros((0,), dtype=row_counts.dtype)
    num_joint_rows   = int(jnp.sum(joint_row_counts)) if K > 0 else 0
    max_row_count    = int(jnp.max(joint_row_counts)) if K > 0 else 0

    # Row indexing helpers for packing joint blocks
    row_index = (jnp.cumsum(joint_row_counts) - joint_row_counts)[:, None] + jnp.arange(max_row_count)[None, :]
    row_mask  = (jnp.arange(max_row_count)[None, :] < joint_row_counts[:, None])

    # Rotation (for translational joint normal)
    Rotation_90deg = jnp.array([[0.0, -1.0],
                                [1.0,  0.0]], dtype=float_dtype)

    def A(phi):
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([[c, -s],
                          [s,  c]], dtype=float_dtype)

    def _get_q_comp(q, idx, off):
        """Return q[3*idx+off] if idx>=0 else 0.0 (JAX-friendly)."""
        idx = jnp.int32(idx)
        return jax.lax.cond(idx >= 0,
                            lambda _: jnp.take(q, 3*idx + off),
                            lambda _: jnp.array(0.0, dtype=q.dtype),
                            operand=None)

    def slice_pose(q, idx):
        """(x,y,phi) for body idx; zeros for world (idx==-1)."""
        x   = _get_q_comp(q, idx, 0)
        y   = _get_q_comp(q, idx, 1)
        phi = _get_q_comp(q, idx, 2)
        return x, y, phi

    def world_point(q, idx, s_local_or_world):
        """If idx>=0: r + A(phi) s_local; if idx==-1: s_local is world anchor."""
        x, y, p = slice_pose(q, idx)
        r   = jnp.array([x, y], dtype=float_dtype)
        p_body = r + A(p) @ s_local_or_world
        return jax.lax.cond(idx >= 0,
                            lambda _: p_body,
                            lambda _: s_local_or_world,
                            operand=None)

    # ----- Drivers -----
    def _signal_value(k, t):
        """f_k(t): sin/cos/linear signal."""
        Aamp, w, ph, b = sig_params[k]
        return jax.lax.switch(
            sig_kind_id[k],
            (
                lambda _: b + Aamp * jnp.sin(w * t + ph),  # 0
                lambda _: b + Aamp * jnp.cos(w * t + ph),  # 1
                lambda _: b + Aamp * t,                    # 2 (linear: A=rate)
            ),
            operand=None
        )

    def _phi_drivers(q, t):
        """(D,) one scalar equation per driver row."""
        def row(k):
            tgt = jnp.int32(drv_target_id[k])   # 0:coord, 1:angle
            ii  = jnp.int32(drv_i[k])
            jj  = jnp.int32(drv_j[k])
            val = _signal_value(k, t)

            def coord_eq(_):
                # q_comp - f(t) = 0
                return _get_q_comp(q, ii, drv_coord_idx[k]) - val

            def angle_eq(_):
                # (phi_j - phi_i) - f(t) = 0
                _, _, phii = slice_pose(q, ii)
                _, _, phij = slice_pose(q, jj)
                return (phij - phii) - val

            return jax.lax.cond(tgt == 0, coord_eq, angle_eq, operand=None)

        return jax.vmap(row)(jnp.arange(D, dtype=jnp.int32)) if D > 0 else jnp.zeros((0,), dtype=float_dtype)

    # ===== Φ(q,t): joint blocks (+ driver rows) =====
    def Phi(q, t):
        def per_joint(k, acc):
            jt = jnp.int32(joint_type_id[k])
            ii = jnp.int32(joint_i[k])
            jj = jnp.int32(joint_j[k])

            # Endpoints in world
            pi = world_point(q, ii, si_xy[k])   # (2,)
            pj = world_point(q, jj, sj_xy[k])   # (2,)

            # Revolute: (pi - pj - c) = 0  → 2 rows
            rev_vec2 = (pi - pj - rev_c_xy[k])

            # Distance: ||pi - pj||^2 - d^2 = 0  → 1 row
            dvec   = pi - pj
            dis_sc = jnp.dot(dvec, dvec) - dist_d[k] ** 2

            # Translation: incidence + parallelism → 2 rows
            xi, yi, phii = slice_pose(q, ii)
            xj, yj, phij = slice_pose(q, jj)
            ui = A(phii) @ trans_ui_l[k]  # unit vector (preprocessor normalizes)
            uj = A(phij) @ trans_uj_l[k]
            wi = Rotation_90deg @ ui                   # normal to ui
            inc = wi @ (pj - pi)                       # point-on-line
            par = wi @ uj                              # axes parallel
            tra_vec2 = jnp.stack([inc, par])

            # Pad to max_row_count for easy packing
            pad2 = max(0, max_row_count - 2)
            pad1 = max(0, max_row_count - 1)
            rev_vec  = jnp.pad(rev_vec2, (0, pad2))
            dis_vec  = jnp.pad(dis_sc[None], (0, pad1))
            tra_vec  = jnp.pad(tra_vec2, (0, pad2))
            zero_vec = jnp.zeros((max_row_count,), dtype=float_dtype)

            vec = jax.lax.cond(jt == REV, lambda _: rev_vec,
                  lambda _: jax.lax.cond(jt == DIS, lambda __: dis_vec,
                  lambda __: jax.lax.cond(jt == TRA, lambda ___: tra_vec,
                                          lambda ___: zero_vec, None), None), None)

            # Mask off unused trailing positions for this joint
            vec = jnp.where(jnp.arange(max_row_count) < row_counts[k], vec, 0.0)
            acc = acc.at[k, :].set(vec)
            return acc

        phi_block0 = jnp.zeros((K, max_row_count), dtype=float_dtype)
        phi_block  = jax.lax.fori_loop(0, K, per_joint, phi_block0)

        # Pack into flat (num_rows,)
        flat = jnp.zeros((num_rows,), dtype=float_dtype)
        flat = flat.at[row_index[row_mask]].set(phi_block[row_mask])

        # Drivers follow joint rows
        if D > 0:
            phi_drv = _phi_drivers(q, t)   # (D,)
            idx_d   = num_joint_rows + jnp.arange(D, dtype=jnp.int32)
            flat = flat.at[idx_d].set(phi_drv)

        return flat

    # ===== Jacobians & RHS terms =====
    if jacobian_mode.lower() == "autodiff":
        # Φ_q and Φ_t via forward-mode
        Phi_q = jax.jacfwd(lambda qq, tt: Phi(qq, tt), argnums=0)
        Phi_t = jax.jacfwd(lambda qq, tt: Phi(qq, tt), argnums=1)
    else:
        raise NotImplementedError("Only 'autodiff' implemented for Jacobian.")

    def J(q, q_dot, t):
        return Phi_q(q, t)

    def Nu(q, t):
        # Explicit time dependence (drivers): Nu = -Φ_t
        return -Phi_t(q, t)

    def Gamma(q, q_dot, t):
        # Γ = -[ Φ_qq(q̇,q̇) + 2 Φ_qt q̇ + Φ_tt ]
        def J_of_q(qq):
            return Phi_q(qq, t)  # (m,3N)
        _, Jdot_from_q = jax.jvp(J_of_q, (q,), (q_dot,))   # (m,3N)
        Phi_qt = jax.jacfwd(lambda tt: Phi_q(q, tt))(t)    # (m,3N)
        Phi_tt = jax.jacfwd(lambda tt: Phi_t(q, tt))(t)    # (m,)
        return -(Jdot_from_q @ q_dot + 2.0 * (Phi_qt @ q_dot) + Phi_tt)

    # Small helpers
    def pack(x, y, phi):
        return jnp.ravel(jnp.column_stack([x, y, phi]))

    def unpack(q):
        N = num_coord // 3
        return q[0::3], q[1::3], q[2::3]

    extras = dict(
        num_rows=num_rows,
        num_coord=num_coord,
        row_counts=row_counts,
        row_offsets=row_offsets,
        pack=pack,
        unpack=unpack,
    )

    # JIT the core functions before returning
    Phi_jit   = jax.jit(Phi)
    J_jit     = jax.jit(J)
    Nu_jit    = jax.jit(Nu)
    Gamma_jit = jax.jit(Gamma)

    return Phi_jit, J_jit, Nu_jit, Gamma_jit, extras


# ------------------------------------------------------------
# Small typed container + ergonomic wrapper
# ------------------------------------------------------------
@dataclass(frozen=True)
class Extras:
    num_rows: int
    num_coord: int
    row_counts: Any
    row_offsets: Any
    pack: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    unpack: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


class SystemFunctions:
    """
    Lightweight, JAX-friendly organizer for the mechanism constraint functions.
    """

    def __init__(self,
                 parameters: Dict,
                 layout: Dict,
                 *,
                 jacobian_mode: str = "autodiff"):
        Phi, J, Nu, Gamma, extras = system_setup(parameters, layout, jacobian_mode=jacobian_mode)

        # Store callables
        self._phi:    Callable[[jnp.ndarray, float], jnp.ndarray] = Phi
        self._J:      Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray] = J
        self._nu:     Callable[[jnp.ndarray, float], jnp.ndarray] = Nu
        self._gamma:  Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray] = Gamma

        self.extras: Extras = Extras(
            num_rows=int(extras["num_rows"]),
            num_coord=int(extras["num_coord"]),
            row_counts=extras["row_counts"],
            row_offsets=extras["row_offsets"],
            pack=extras["pack"],
            unpack=extras["unpack"],
        )

        # Pre-JIT entry points (handy for callers wanting jitted versions)
        self.phi_jit     = jax.jit(self._phi)
        self.jacobian_jit= jax.jit(self._J)
        self.nu_jit      = jax.jit(self._nu)
        self.gamma_jit   = jax.jit(self._gamma)

    # Non-jitted accessors
    def phi(self, q: jnp.ndarray, t: float) -> jnp.ndarray:
        return self._phi(q, t)

    def jacobian(self, q: jnp.ndarray, qdot: jnp.ndarray, t: float) -> jnp.ndarray:
        return self._J(q, qdot, t)

    def nu(self, q: jnp.ndarray, t: float) -> jnp.ndarray:
        return self._nu(q, t)

    def gamma(self, q: jnp.ndarray, qdot: jnp.ndarray, t: float) -> jnp.ndarray:
        return self._gamma(q, qdot, t)

    # Legacy shape for plumbing
    def as_tuple(self) -> Tuple[
        Callable[[jnp.ndarray, float], jnp.ndarray],
        Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
        Callable[[jnp.ndarray, float], jnp.ndarray],
        Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray],
        Dict[str, Any],
    ]:
        extras = dict(
            num_rows=self.extras.num_rows,
            num_coord=self.extras.num_coord,
            row_counts=self.extras.row_counts,
            row_offsets=self.extras.row_offsets,
            pack=self.extras.pack,
            unpack=self.extras.unpack,
        )
        return self._phi, self._J, self._nu, self._gamma, extras


def make_system_functions(parameters: Dict,
                          layout: Dict,
                          *,
                          jacobian_mode: str = "autodiff") -> SystemFunctions:
    return SystemFunctions(parameters, layout, jacobian_mode=jacobian_mode)
