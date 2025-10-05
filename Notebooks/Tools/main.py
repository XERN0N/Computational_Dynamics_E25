# main.py
import numpy as np
import jax.numpy as jnp
from preprocessor import MechanismBuilder
from system_solver import make_kinematic_solver


# ---------------- Simulation variables ----------------
start_time = 0.0
stop_time  = 10.0
time_step  = 0.01

# ---------------- Mechanism constants -----------------
L1 = 10.0
L2 = 26.0
L3 = 18.0
G  = 20.0
omega = 1.5  # rad/s (constant angular velocity of link1)


# ---------------- Build mechanism ---------------------
mb = MechanismBuilder("assignment_three_link")

# Ground anchor points
mb.add_world_point("G1", (0.0, 0.0))
mb.add_world_point("G2", (G,   0.0))

# Bodies with local points (±L/2 along x-axis)
mb.add_body(
    "link1",
    {"A": (-L1/2, 0.0), "B": ( L1/2, 0.0)},
    initial_configuration=(5.0, 0.0, 0.0),  # guess like your IG
    mass=1.0, inertia_zz=0.1,
)
mb.add_body(
    "link2",
    {"B2": (-L2/2, 0.0), "C2": ( L2/2, 0.0)},
    initial_configuration=(20.0, 10.0, np.pi/4),
    mass=1.0, inertia_zz=0.1,
)
mb.add_body(
    "link3",
    {"C3": (-L3/2, 0.0), "D3": ( L3/2, 0.0)},
    initial_configuration=(20.0, 9.0, np.pi + np.pi/2),
    mass=1.0, inertia_zz=0.1,
)

# Revolute joints
mb.add_joint(
    "J1_ground",
    "revolute",
    {"body": "link1", "point": "A"},
    {"body": "world", "point": "G1"},
    {"c": (0.0, 0.0)},
)
mb.add_joint(
    "J12",
    "revolute",
    {"body": "link1", "point": "B"},
    {"body": "link2", "point": "B2"},
    {"c": (0.0, 0.0)},
)
mb.add_joint(
    "J23",
    "revolute",
    {"body": "link2", "point": "C2"},
    {"body": "link3", "point": "C3"},
    {"c": (0.0, 0.0)},
)
mb.add_joint(
    "J3_ground",
    "revolute",
    {"body": "link3", "point": "D3"},
    {"body": "world", "point": "G2"},
    {"c": (0.0, 0.0)},
)

# Driver: φ1(t) = ω·t
mb.add_driver(
    "phi1_driver",
    target={"type": "coord", "body": "link1", "coord": "phi"},
    signal={"kind": "linear", "rate": omega, "bias": 0.0},
)

parameters, layout, q0, report = mb.compile()


# ---------------- Build solvers -----------------------
assemble, velocities, accelerations, step_kinematics, simulate, extras = make_kinematic_solver(parameters, layout)


# ---------------- Single solve at t0 ------------------
t0 = start_time
q, info = assemble(q0, t0)
print("converged:", bool(info["converged"]), "steps:", int(info["n_steps"]), "res:", float(info["res_norm"]))
print("q0 corrected:", np.array(q))


# ---------------- Trajectory simulation ---------------
times = jnp.arange(start_time, stop_time + 1e-12, time_step)
Qs, Qdots, Qddots = simulate(times, q0)

print("Qs:", np.array(Qs).shape, "Qdots:", np.array(Qdots).shape, "Qddots:", np.array(Qddots).shape)


# ---------------- Optional: Matplotlib plots ----------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.asarray(times)
    Q   = np.asarray(Qs)
    Qd  = np.asarray(Qdots)
    Qdd = np.asarray(Qddots)

    # Generalized coordinates
    plt.figure()
    for i, label in enumerate(["x1","y1","phi1","x2","y2","phi2","x3","y3","phi3"]):
        plt.plot(t, Q[:, i], label=label)
    plt.xlabel("time [s]"); plt.ylabel("q"); plt.title("Generalized coordinates")
    plt.legend(ncol=3); plt.grid(ls=":")

    # Velocities
    plt.figure()
    for i, label in enumerate(["x1dot","y1dot","phi1dot","x2dot","y2dot","phi2dot","x3dot","y3dot","phi3dot"]):
        plt.plot(t, Qd[:, i], label=label)
    plt.xlabel("time [s]"); plt.ylabel("qdot"); plt.title("Generalized velocities")
    plt.legend(ncol=3); plt.grid(ls=":")

    # Accelerations
    plt.figure()
    for i, label in enumerate(["x1ddot","y1ddot","phi1ddot","x2ddot","y2ddot","phi2ddot","x3ddot","y3ddot","phi3ddot"]):
        plt.plot(t, Qdd[:, i], label=label)
    plt.xlabel("time [s]"); plt.ylabel("qddot"); plt.title("Generalized accelerations")
    plt.legend(ncol=3); plt.grid(ls=":")

    # Driver check: φ1 vs ωt
    plt.figure()
    plt.plot(t, Q[:, 2], label="phi1 (sim)")
    plt.plot(t, omega * t, "--", label="phi1 (ref = ωt)")
    plt.xlabel("time [s]"); plt.ylabel("angle [rad]"); plt.title("Driver tracking")
    plt.legend(); plt.grid(ls=":")

    plt.show()
