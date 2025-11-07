# main.py — Inverse Slider-Crank (with simple hand-picked initial configuration)

from __future__ import annotations
import numpy as np

from preprocessor import MechanismBuilder
from runner import run_kinematics
from plotter import animate_from_traj, joint_plots_from_traj

# ----- time vector -----
START, STOP, DT = 0.0, 10.0, 0.01
times = np.arange(START, STOP + 1e-12, DT)

# ----- geometry -----
BODY_3, BODY_1, BODY_2 = 16.0, 4.0, 2.0
O = (0.0, 0.0)       # ground pin for small arm (A)
D_pt = (-8.0, 0.0)   # ground pin for long element (D)

# ------------------------------------------------------------------
# Initial configuration (just a sensible guess, not exactly consistent)
# Chosen to resemble the sketch: long guide slightly inclined,
# small link up-and-right, slider roughly under B on the guide.
# ------------------------------------------------------------------

# Small link (link1) angle ~ 1.2 rad (~69°). Center near the midpoint of AB.
phi1_0 = np.rad2deg(80)  # rad
x1_0   = 0.724000000000  # ≈ (L_SMALL/2)*cos(phi1_0)
y1_0   = 1.864000000000  # ≈ (L_SMALL/2)*sin(phi1_0)

# Slider (link2) roughly aligned with the guide; S somewhere near C.
phi2_0 = np.rad2deg(15)  # rad ≈ 12°
x2_0   = 1.450000000000
y2_0   = 2.000000000000

# Long guide (link3) inclined ~12°, positioned so D is roughly near (-8,0)
phi3_0 = np.rad2deg(15)  # rad ≈ 12°
x3_0   = -0.174821119075  # ≈ -8 + (L_LONG/2)*cos(12°)
y3_0   =  1.663292392775  # ≈      (L_LONG/2)*sin(12°)

# ----- helpers -----
def slender_rod_inertia(m, L):
    return (1.0 / 12.0) * m * L * L

def build_mechanism():
    mb = MechanismBuilder("inverse_slider_crank")

    # world points
    mb.add_world_point("O", O)
    mb.add_world_point("D", D_pt)

    # bodies
    m1, m2, m3 = 40.0, 20.0, 160.0
    mb.add_body(
        "link1",
        {"A": (-BODY_1/2, 0.0), "B": (BODY_1/2, 0.0)},
        initial_configuration=(x1_0, y1_0, phi1_0),
        mass=m1,
        inertia_zz=slender_rod_inertia(m1, BODY_1),
    )
    mb.add_body(
        "slider",
        {"S": (0.0, 0.0), "B": (0.0, -BODY_2)},
        initial_configuration=(x2_0, y2_0, phi2_0),
        mass=m2,
        inertia_zz=slender_rod_inertia(m2, 2.0*BODY_2),
    )
    mb.add_body(
        "link3",
        {"D": (-BODY_3/2, 0.0), "C": (0.0, 0.0), "E": (BODY_3/2, 0.0)},
        initial_configuration=(x3_0, y3_0, phi3_0),
        mass=m3,
        inertia_zz=slender_rod_inertia(m3, BODY_3),
    )

    # joints
    mb.add_joint(
        "rev_A_ground", "revolute",
        {"body": "link1", "point": "A"},
        {"body": "world", "point": "O"},
        {}
    )
    mb.add_joint(
        "rev_B", "revolute",
        {"body": "link1", "point": "B"},
        {"body": "slider", "point": "B"},
        {}
    )
    mb.add_joint(
        "trans_C", "translation",
        {"body": "slider", "point": "S"},
        {"body": "link3", "point": "E"},
        {"ui_local": (1.0, 0.0), "uj_local": (1.0, 0.0)}  # axes aligned with +x
    )
    mb.add_joint(
        "rev_D_ground", "revolute",
        {"body": "link3", "point": "D"},
        {"body": "world", "point": "D"},
        {}
    )

    # driver: link1 angle about A with constant speed 1.5 rad/s
    mb.add_driver(
        "crank_speed",
        target={"type": "coord", "body": "link1", "coord": "phi"},
        signal={"kind": "linear", "rate": 1.5, "bias": float(phi1_0)},
    )

    return mb.compile()

def main():
    mpack, traj, info = run_kinematics(build_mechanism, times)

    #Get inverse kinematics
    g = 9.81

    
    from system_functions import SystemFunctions

    def inverse_driver_torque(mpack, traj):
        """
        Driving torque(s) from Haug Ch. 6:
        Phi_q(q,t)^T * lambda = Q_A(q,qd,t) - M(q) * qdd
        Returns (times, tau_hist) where tau_hist has shape (T, num_drivers),
        or (T,) if there is only one driver.
        """
        params = mpack.parameters
        layout = mpack.layout

        # sizes
        N = int(layout["num_bodies"]) if "num_bodies" in layout else int(params["mass"].shape[0])
        m = int(layout["num_rows"])
        K = int(layout["num_joints"])  # joint count

        # first sum(row_counts[:K]) rows are joints; the rest are drivers
        row_counts = np.asarray(layout["row_counts"])
        num_joint_rows = int(np.sum(row_counts[:K])) if K > 0 else 0
        num_driver_rows = m - num_joint_rows
        assert num_driver_rows >= 1, "No driver rows found. Add a driver constraint."

        # mass matrix (planar blocks)
        mass = np.asarray(params["mass"])
        Jz   = np.asarray(params["inertia_zz"])
        M = np.zeros((3*N, 3*N))
        for i in range(N):
            M[3*i+0, 3*i+0] = mass[i]
            M[3*i+1, 3*i+1] = mass[i]
            M[3*i+2, 3*i+2] = Jz[i]

        # applied generalized forces: gravity only (use both gx and gy just in case)
        gx, gy = map(float, params.get("gravity", (0.0, -9.81)))
        QA = np.zeros((3*N,))
        for i in range(N):
            QA[3*i+0] += mass[i] * gx
            QA[3*i+1] += mass[i] * gy
            # no direct gravitational torque term

        # Jacobian from your JAX system functions
        sf = SystemFunctions(params, layout)
        Jfun = sf.jacobian  # (q, qdot, t) -> (m, 3N)

        times = np.asarray(traj.times)
        Qs    = np.asarray(traj.Qs)
        Qds   = np.asarray(traj.Qdots)  if traj.Qdots  is not None else np.zeros_like(Qs)
        Qdds  = np.asarray(traj.Qddots) if traj.Qddots is not None else np.zeros_like(Qs)

        lam_drv_hist = np.zeros((times.shape[0], num_driver_rows))
        for k, t in enumerate(times):
            q, qd, qdd = Qs[k], Qds[k], Qdds[k]
            Phi_q = np.asarray(Jfun(q, qd, float(t)))         # (m, 3N)
            rhs   = QA - M @ qdd                              # (3N,)

            # Solve Phi_q^T lambda = rhs (lstsq for robustness)
            lam, *_ = np.linalg.lstsq(Phi_q.T, rhs, rcond=None)  # (m,)
            lam_drv_hist[k, :] = lam[num_joint_rows:]            # keep driver rows only

        # if just one driver, return a 1D vector
        if num_driver_rows == 1:
            lam_drv_hist = lam_drv_hist[:, 0]

        return times, lam_drv_hist

    


    animate_from_traj(mpack.parameters, traj, frame_step=10, save_path="inverse_slider.gif")
    joint_plots_from_traj(mpack.parameters, traj, save_path="inverse_slider_trajs.png")
    print("Saved: inverse_slider.gif, inverse_slider_trajs.png")

    t, tau = inverse_driver_torque(mpack, traj)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, tau)
    plt.xlabel("time [s]")
    plt.ylabel("driver torque τ_φ [N·m]")
    plt.title("Required torque for φ-driver (Haug Ch. 6 inverse dynamics)")
    plt.grid(True)
    plt.savefig("driver_torque.png", dpi=150, bbox_inches="tight")
    print("Saved: driver_torque.png")


if __name__ == "__main__":
    main()
