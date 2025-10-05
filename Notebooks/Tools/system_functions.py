# System_solver.py
# Planar multibody constraint assembly following Haug (Φ, J, Nu, Γ).
# Compatible with MechanismBuilder.compile() from preprocessor.py.

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple

jax.config.update('jax_enable_x64', True)

def system_setup(
    parameters: Dict,
    layout: Dict,
    *,
    jacobian_mode: str = "autodiff",
) -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    """
    This function returns pure jittable jax-functions that can be used to describe the system.
    The inputs should be the output of the class MechanismBuilder.

    Returns:

    Phi     (q,t)         -> (m,)           - Constraint equations
    J       (q, q_dot, t) -> (m, 3*N)       - Constraint jacobian
    Nu      (q, t)        -> (m,)           - time derivative of phi
    Gamma   (q, q_dot, t) -> (m,)           - Used to get the accelerations of the system

    Extras: {num_row, num_coord, row_count, row_offset, pack, unpack}
    
    
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


    # ===== Drivers (compact sine/cos) =====
    drv = parameters.get("drivers", {})
    drv_target_id = jnp.asarray(drv.get("target_id",   jnp.zeros((0,), dtype=jnp.int32)))   # 0:coord, 1:angle
    drv_i         = jnp.asarray(drv.get("i",          jnp.zeros((0,), dtype=jnp.int32)))
    drv_j         = jnp.asarray(drv.get("j",          jnp.zeros((0,), dtype=jnp.int32)))
    drv_coord_idx = jnp.asarray(drv.get("coord_index",jnp.zeros((0,), dtype=jnp.int32)))

    sig           = drv.get("signal", {})
    sig_kind_id   = jnp.asarray(sig.get("kind_id",    jnp.zeros((0,), dtype=jnp.int32)))    # 0:sin, 1:cos
    # params = (A, omega, phi, bias)
    sig_params    = jnp.asarray(sig.get("params",     jnp.zeros((sig_kind_id.shape[0], 4), dtype=layout.get("float_dtype", jnp.float64))))

    D = int(sig_kind_id.shape[0])  # number of drivers


    # ===== Layout and sizes =====
    row_counts  = jnp.asarray(layout["row_counts"])   
    row_offsets = jnp.asarray(layout["row_offset"])   
    num_rows    = int(layout["num_rows"])
    num_coord   = int(layout["num_coord"])
    float_dtype = layout.get("float_dtype", jnp.float64)

    joint_ids = layout.get("joint_ids", {"revolute": 0, "distance": 1, "translation": 2})
    REV = int(joint_ids.get("revolute", 0))
    DIS = int(joint_ids.get("distance", 1))
    TRA = int(joint_ids.get("translation", 2))

    K = joint_type_id.shape[0]
    
    joint_row_counts = row_counts[:K] if K > 0 else jnp.zeros((0,), dtype=row_counts.dtype)
    num_joint_rows   = int(jnp.sum(joint_row_counts)) if K > 0 else 0
    max_row_count = int(jnp.max(joint_row_counts)) if K > 0 else 0

    # Create row index matrix and as a mask
    row_index = (jnp.cumsum(joint_row_counts) - joint_row_counts)[:, None] + jnp.arange(max_row_count)[None, :]
    row_mask  = (jnp.arange(max_row_count)[None, :] < joint_row_counts[:, None])

    # Rotation matrix for translational joint
    Rotation_90deg = jnp.array([[0.0, -1.0],
                                [1.0,  0.0]], dtype=float_dtype)

    def A(phi):
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([[c, -s],
                          [s,  c]], dtype=float_dtype)
    
    def B(phi):
        c, s = jnp.cos(phi), jnp.sin(phi)
        return jnp.array([[-s, -c],
                        [c,  -s]], dtype=float_dtype)

    # JAX-safe handling int type
    def _get_q_comp(q, idx, off):
        """Return q[3*idx+off] if idx>=0 else 0.0, using JAX control-flow."""
        idx = jnp.int32(idx)
        return jax.lax.cond(idx >= 0,
                            lambda _: jnp.take(q, 3*idx + off),
                            lambda _: jnp.array(0.0, dtype=q.dtype),
                            operand=None)

    def slice_pose(q, idx):
        """Return (x,y,phi) for body idx; zeros for world (idx==-1)."""
        x   = _get_q_comp(q, idx, 0)
        y   = _get_q_comp(q, idx, 1)
        phi = _get_q_comp(q, idx, 2)
        return x, y, phi

    def world_point(q, idx, s_local_or_world):
        """If idx>=0: r + A(phi) s_local; if idx==-1: s_local is world anchor."""
        idx = jnp.int32(idx)
        x, y, p = slice_pose(q, idx)
        r   = jnp.array([x, y], dtype=float_dtype)
        p_body = r + A(p) @ s_local_or_world
        return jax.lax.cond(idx >= 0,
                            lambda _: p_body,
                            lambda _: s_local_or_world,
                            operand=None)

    # Driver helpers
    def _signal_value(k, t):
        """f_k(t) = bias + A*sin(w t + phi) / cos(...) / (linear: A*t)"""
        Aamp, w, ph, b = sig_params[k]
        return jax.lax.switch(
            sig_kind_id[k],
            (
                lambda _: b + Aamp * jnp.sin(w * t + ph),   # 0: sin
                lambda _: b + Aamp * jnp.cos(w * t + ph),   # 1: cos
                lambda _: b + Aamp * t,                     # 2: linear (A=rate)
            ),
            operand=None
        )

    def _phi_drivers(q, t):
        """Vector (D,) with one scalar equation per driver."""
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

    # ===== Constraint assembly =====
    def Phi(q, t):
        # Build per-joint padded vectors, shape (K, max_rc)
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
            inc = wi @ (pj - pi)           # point-on-line
            par = wi @ uj                  # axes parallel
            tra_vec2 = jnp.stack([inc, par])

            # Padded vectors
            pad2 = max(0, max_row_count - 2)
            pad1 = max(0, max_row_count - 1)
            rev_vec = jnp.pad(rev_vec2, (0, pad2))
            dis_vec = jnp.pad(dis_sc[None], (0, pad1))
            tra_vec = jnp.pad(tra_vec2, (0, pad2))
            zero_vec = jnp.zeros((max_row_count,), dtype=float_dtype)

            # JAX branching on joint type
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
        
        if D > 0:
            phi_drv = _phi_drivers(q, t)   # (D,)
            idx_d   = num_joint_rows + jnp.arange(D, dtype=jnp.int32)
            flat = flat.at[idx_d].set(phi_drv) 

        return flat

    # Jacobian via autodiff
    if jacobian_mode.lower() == "autodiff":
        Phi_q = jax.jacfwd(lambda qq, tt: Phi(qq, tt), argnums=0)
        Phi_t = jax.jacfwd(lambda qq, tt: Phi(qq, tt), argnums=1)
    else:
        raise NotImplementedError("Only 'autodiff' implemented for Jacobian.")

    def J(q, q_dot, t):
        # For convenience keep same signature as solvers expect
        return Phi_q(q, t)

    def Nu(q, t):
        # With drivers, Φ has explicit time dependence: Nu = -Φ_t
        return -Phi_t(q, t)

    def Gamma(q, q_dot, t):
        # Full form with explicit-time dependence:
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

    # JIT the core functions
    Phi_jit   = jax.jit(Phi)
    J_jit     = jax.jit(J)
    Nu_jit    = jax.jit(Nu)
    Gamma_jit = jax.jit(Gamma)

    return Phi_jit, J_jit, Nu_jit, Gamma_jit, extras



if __name__ == "__main__":
    # Minimal end-to-end sanity check for driver rows in Nu
    from preprocessor import MechanismBuilder  # same folder assumed

    # Build a single link pinned to ground
    mb = MechanismBuilder("drv_demo")
    mb.add_world_point("O", (0.0, 0.0))
    mb.add_body("link1", {"A": (0.0, 0.0)}, initial_configuration=(0.0, 0.0, 0.0),
                mass=1.0, inertia_zz=0.1)
    mb.add_joint(
        name="J_rev",
        joint_type="revolute",
        endpoint_1={"body": "link1", "point": "A"},
        endpoint_2={"body": "world", "point": "O"},
        constraint_arguments={"c": (0.0, 0.0)},
    )

    # Two drivers to exercise multiple rows:
    # 1) Absolute angle φ_link1(t) = 0.5 * sin(2 t + 0.1)
    mb.add_driver(
        "phi_sin",
        target={"type": "coord", "body": "link1", "coord": "phi"},
        signal={"kind": "sin", "A": 0.5, "omega": 2.0, "phi": 0.1, "bias": 0.0},
    )

    params, layout, q0, report = mb.compile()
    Phi, J, Nu, Gamma, extras = system_setup(params, layout)

    # Evaluate Nu and compare its driver rows to df/dt analytically
    t = 0.3
    nu = Nu(q0, t)

    # Joint rows are first; drivers follow
    K = int(params["joint_type_id"].shape[0])
    num_joint_rows = int(jnp.sum(layout["row_counts"][:K])) if K > 0 else 0
    D = int(params["drivers"]["signal"]["kind_id"].shape[0])

    nu_drv = nu[num_joint_rows:num_joint_rows + D]

    kinds = params["drivers"]["signal"]["kind_id"]           # 0: sin, 1: cos
    pars  = params["drivers"]["signal"]["params"]            # (A, omega, phi, bias)
    expected = []
    for k in range(D):
        Aamp, w, ph, b = pars[k]
        # d/dt [bias + A sin(w t + phi)] = A w cos(w t + phi)
        # d/dt [bias + A cos(w t + phi)] = -A w sin(w t + phi)
        if int(kinds[k]) == 0:
            expected.append(Aamp * w * jnp.cos(w * t + ph))
        else:
            expected.append(-Aamp * w * jnp.sin(w * t + ph))
    expected = jnp.asarray(expected, dtype=nu.dtype)

    print("\n=== Driver Nu check ===")
    print("Nu driver rows:", nu_drv)
    print("Expected df/dt :", expected)
    print("max |diff|     :", float(jnp.max(jnp.abs(nu_drv - expected))))

    # (Optional) also show sizes for sanity
    print("\nLayout/report:")
    print("num_joints:", layout["num_joints"], " num_drivers:", D,
          " num_rows:", layout["num_rows"], " DoFs:", report["DoFs"])


