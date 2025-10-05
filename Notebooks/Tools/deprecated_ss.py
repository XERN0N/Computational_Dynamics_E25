# MechanismBuilder.compile() → (parameters, layout, q0, report)
#
# Symbols:
#   N = number of dynamic bodies
#   K = number of joints
#   n_coord = 3*N              (planar: [x, y, φ] per body)
#   n_rows  = sum(row_counts)  (total scalar constraint equations)
#
# A) parameters  (all arrays are JAX arrays; floats typically float64, ids int32)
#   Bodies:
#     mass         : shape (N,)       # m_i per body (kg)
#     inertia_zz   : shape (N,)       # Izz_i per body (kg·m²)
#     gravity      : shape (2,)       # [gx, gy] (m/s²)
#
#   Joints (monolithic, aligned by joint index k = 0..K-1):
#     joint_type_id: shape (K,)  int  # e.g., revolute=0, distance=1
#     joint_i      : shape (K,)  int  # endpoint-1 body id, or -1 if world
#     joint_j      : shape (K,)  int  # endpoint-2 body id, or -1 if world
#     si_xy        : shape (K,2) float# endpoint-1 point: local [sx,sy] if body_i; world [x,y] if -1
#     sj_xy        : shape (K,2) float# endpoint-2 point: local if body_j; world if -1
#     dist_d OR distance : shape (K,) float  # distance value for distance joints; 0 for others
#
#   Conventions:
#     - If joint_* == -1, that side is ground; corresponding s*_xy row is the world XY anchor.
#     - If joint_* >= 0, s*_xy is the attachment point in that body’s LOCAL frame.
#
# B) layout (sizes, indices, and maps for assembly)
#     num_bodies   : int       # = N
#     num_joints   : int       # = K
#     num_coord    : int       # = 3*N
#     body_id_by_name : dict[str -> int]       # dynamic bodies only
#     coord_slices    : dict[str -> (start,stop)]  # start=3*id, stop=start+3
#     row_counts   : shape (K,)  int  # rows per joint (e.g., revolute=2, distance=1)
#     row_offset   : shape (K,)  int  # start row index for each joint (prefix sum of row_counts)
#     num_rows     : int             # = n_rows = sum(row_counts)
#
#   Indexing helpers (derived from layout):
#     - Body i state slice in q: [3*i : 3*i+3] → (x_i, y_i, φ_i)
#     - Joint k row slice in Φ: [row_offset[k] : row_offset[k] + row_counts[k]]
#
# C) q0 / initial_configuration
#     q0           : shape (3*N,) float  # stacked [x0, y0, φ0] in body-id order (0..N-1)
#
# Report (informational counts)
#     num_bodies, num_joints, num_coord, num_rows, DoFs (= 3*N - n_rows)

import jax
import jax.numpy as jnp
import kinematic_functions
from preprocessor import MechanismBuilder as MB
from typing import Callable, Dict, Tuple
import numpy as np



def system_setup(parameters: Dict, layout: Dict, *, jacobian_mode: str = "autodiff") -> Tuple[Callable, Callable, Callable, Callable, Dict]:
    
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

    joint_type_id =jnp.asarray(parameters.get("joint_type_id"))
    joint_i = jnp.asarray(parameters.get("joint_i"))
    joint_j = jnp.asarray(parameters.get("joint_j"))
    si_xy = jnp.asarray(parameters.get("si_xy"))
    sj_xy = jnp.asarray(parameters.get("sj_xy"))

    row_counts = jnp.asarray(layout.get("row_counts"))
    row_offsets = jnp.asarray(layout.get("row_offset"))
    num_rows = jnp.asarray(layout.get("num_rows"))
    num_coord = jnp.asarray(layout.get("num_coord"))
    max_row_count = int(jnp.max(row_counts))
    float_dtype = layout.get("float_dtype", jnp.float64)

    row_index = row_offsets[:, None] + jnp.arange(max_row_count)[None, :]
    row_mask = jnp.arange(max_row_count)[None, :] < row_counts[:, None]

    def A(phi):
        c, s = jnp.cos(phi), jnp.sin(phi)

        return jnp.array([[c, -s], [s, c]])

    def B(phi):
        c, s = jnp.cos(phi), jnp.sin(phi)

        return jnp.array([[-s, -c], [c, -s]])
    
    def slice_pose(q, idx):
        is_body = idx >= 0
        
        x = jnp.where(is_body, q[3*idx], 0.0)
        y = jnp.where(is_body, q[3*idx+1], 0.0)
        p = jnp.where(is_body, q[3*idx+2], 0.0)
    
        return x, y, p

    def world_point(q, idx, s_xy):
        x, y, p = slice_pose(q, idx)
        r = jnp.array([x, y])
        p_body = r + A(p) @ s_xy

        return  jnp.where(idx >=0, p_body, s_xy)
    
    revolute = 0

    def phi_rev(q, t):




if __name__ == "__main__":
    test = MB("test")
    
    point_dict_body_1 = {"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [1.0, 1.0], "D": [2.0, 1.0]}
    point_dict_body_2 = {"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [1.0, 1.0], "D": [2.0, 1.0]}
    test.add_body("Body_1", point_dict_body_1, [0,0,0], 3, 0.5)
    test.add_body("Body_2", point_dict_body_2, [0,0,np.pi/2], 25, 4.5)
    test.add_joint("JB1_W", "revolute",
                {"body":"Body_1","point":"A"},
                {"body":"world","point":"O"},
                {})
    test.add_joint("JB1_B2", "revolute",
                {"body":"Body_1","point":"B"},
                {"body":"Body_2","point":"A"},
                {"distance": 5})
    test.add_joint("JB2_W", "revolute",
                {"body":"Body_2","point":"C"},
                {"body":"world","point":"O"},
                {"distance": 5})


    params, layout, initial_configuration, report = test.compile()

    sys_test = system_setup(params, layout, initial_configuration)

    print(sys_test)