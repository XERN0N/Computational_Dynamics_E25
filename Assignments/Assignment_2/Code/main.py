# main.py
import numpy as np

from preprocessor import MechanismBuilder
from runner import run_kinematics
from plotter import animate_from_traj, joint_plots_from_traj


# ---------------- Simulation knobs ----------------
START = 0.0
STOP  = 10.0
DT    = 0.01

# ---------------- Mechanism constants -----------------
L1 = 10.0
L2 = 26.0
L3 = 18.0
G  = 20.0
OMEGA = 1.5  # phi1(t) = OMEGA * t


def build_mechanism():
    """
    Returns (parameters, layout, q0, report) via MechanismBuilder.compile()
    """
    mb = MechanismBuilder("assignment_three_link")

    # World points
    mb.add_world_point("G1", (0.0, 0.0))
    mb.add_world_point("G2", (G,   0.0))

    # Bodies (names, point dictionaries, initial poses, mass, inertia)
    mb.add_body(
        "link1",
        {"A": (-L1 / 2, 0.0), "B": ( L1 / 2, 0.0)},
        initial_configuration=(5.0, 0.0, 0.0),
        mass=1.0, inertia_zz=0.1
    )
    mb.add_body(
        "link2",
        {"B2": (-L2 / 2, 0.0), "C2": ( L2 / 2, 0.0)},
        initial_configuration=(20.0, 10.0, np.pi / 4),
        mass=1.0, inertia_zz=0.1
    )
    mb.add_body(
        "link3",
        {"C3": (-L3 / 2, 0.0), "D3": ( L3 / 2, 0.0)},
        initial_configuration=(20.0, 9.0, np.pi + np.pi / 2),
        mass=1.0, inertia_zz=0.1
    )

    # Joints
    mb.add_joint("J1_ground", "revolute",
                 {"body": "link1", "point": "A"},
                 {"body": "world", "point": "G1"},
                 {"c": (0.0, 0.0)})

    mb.add_joint("J12", "revolute",
                 {"body": "link1", "point": "B"},
                 {"body": "link2", "point": "B2"},
                 {"c": (0.0, 0.0)})

    mb.add_joint("J23", "revolute",
                 {"body": "link2", "point": "C2"},
                 {"body": "link3", "point": "C3"},
                 {"c": (0.0, 0.0)})

    mb.add_joint("J3_ground", "revolute",
                 {"body": "link3", "point": "D3"},
                 {"body": "world", "point": "G2"},
                 {"c": (0.0, 0.0)})

    # Driver: φ1(t) = OMEGA · t   (requires your preprocessor/system_functions linear driver support)
    mb.add_driver(
        "phi1_driver",
        target={"type": "coord", "body": "link1", "coord": "phi"},
        signal={"kind": "linear", "rate": OMEGA, "bias": 0.0},
    )

    return mb.compile()  # (parameters, layout, q0, report)


def main():
    # time grid
    times = np.arange(START, STOP + 1e-12, DT)

    # run
    mpack, traj, info = run_kinematics(build_mechanism, times,
                                       solver_kwargs=dict(rtol=1e-8, atol=1e-12, max_steps=100))
    try:
        print(f"assemble@t0 converged={bool(info['converged'])}, "
              f"steps={int(info['n_steps'])}, "
              f"res={float(info['res_norm']):.2e}, "
              f"step_norm={float(info['step_norm']):.2e}")
    except Exception:
        pass  # if info missing or non-dict

    # visualize
    animate_from_traj(mpack.parameters, traj, frame_step=10, save_path="fourbar.gif")
    joint_plots_from_traj(mpack.parameters, traj, save_path="joint_trajs.png")

    print("Saved: fourbar.gif, joint_trajs.png")


if __name__ == "__main__":
    main()
