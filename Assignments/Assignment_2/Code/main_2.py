# main.py — Inverse Slider-Crank (with simple hand-picked initial configuration)

from __future__ import annotations
import numpy as np

from preprocessor import MechanismBuilder
from runner import run_kinematics
from plotter import animate_from_traj, joint_plots_from_traj

# ----- time vector -----
START, STOP, DT = 0.0, 6.0, 0.01
times = np.arange(START, STOP + 1e-12, DT)

# ----- geometry -----
L_LONG, L_SMALL, OFFSET = 16.0, 4.0, 2.0
O = (0.0, 0.0)       # ground pin for small arm (A)
D_pt = (-8.0, 0.0)   # ground pin for long element (D)

# ------------------------------------------------------------------
# Initial configuration (just a sensible guess, not exactly consistent)
# Chosen to resemble the sketch: long guide slightly inclined,
# small link up-and-right, slider roughly under B on the guide.
# ------------------------------------------------------------------

# Small link (link1) angle ~ 1.2 rad (~69°). Center near the midpoint of AB.
phi1_0 = 1.200000000000  # rad
x1_0   = 0.724000000000  # ≈ (L_SMALL/2)*cos(phi1_0)
y1_0   = 1.864000000000  # ≈ (L_SMALL/2)*sin(phi1_0)

# Slider (link2) roughly aligned with the guide; S somewhere near C.
phi2_0 = 0.209439510239  # rad ≈ 12°
x2_0   = 1.450000000000
y2_0   = 2.000000000000

# Long guide (link3) inclined ~12°, positioned so D is roughly near (-8,0)
phi3_0 = 0.209439510239  # rad ≈ 12°
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
        {"A": (-L_SMALL/2, 0.0), "B": (L_SMALL/2, 0.0)},
        initial_configuration=(x1_0, y1_0, phi1_0),
        mass=m1,
        inertia_zz=slender_rod_inertia(m1, L_SMALL),
    )
    mb.add_body(
        "slider",
        {"S": (0.0, 0.0), "B": (0.0, +OFFSET)},
        initial_configuration=(x2_0, y2_0, phi2_0),
        mass=m2,
        inertia_zz=slender_rod_inertia(m2, 2.0*OFFSET),
    )
    mb.add_body(
        "link3",
        {"D": (-L_LONG/2, 0.0), "C": (0.0, 0.0), "E": (L_LONG/2, 0.0)},
        initial_configuration=(x3_0, y3_0, phi3_0),
        mass=m3,
        inertia_zz=slender_rod_inertia(m3, L_LONG),
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
        {"body": "link3", "point": "C"},
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
    animate_from_traj(mpack.parameters, traj, frame_step=2, save_path="inverse_slider.gif")
    joint_plots_from_traj(mpack.parameters, traj, save_path="inverse_slider_trajs.png")
    print("Saved: inverse_slider.gif, inverse_slider_trajs.png")

if __name__ == "__main__":
    main()
