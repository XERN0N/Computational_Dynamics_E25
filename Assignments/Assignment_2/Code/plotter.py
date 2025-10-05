# plotter.py
# Clean, NumPy-only plotting utilities for planar multibody kinematics,
# with bodies rendered as segments between their incident joint endpoints.

from __future__ import annotations

from typing import Iterable, Optional, Tuple, List

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------
# Low-level geometry utilities
# ----------------------------
def _num_bodies_from_params(parameters: dict, fallback_N: Optional[int] = None) -> int:
    ji = np.asarray(parameters["joint_i"], dtype=int)
    jj = np.asarray(parameters["joint_j"], dtype=int)
    mx = -1
    if ji.size:
        mx = max(mx, int(np.max(ji)))
    if jj.size:
        mx = max(mx, int(np.max(jj)))
    N = (mx + 1) if mx >= 0 else 0
    if fallback_N is not None:
        # prefer shape from trajectory/q if provided
        N = max(N, int(fallback_N))
    return N

def _q_split(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (3N,) -> (x(N,), y(N,), phi(N,))."""
    q = np.asarray(q, dtype=float).reshape(-1)
    assert q.size % 3 == 0, "q must have length 3N"
    N = q.size // 3
    x = q[0::3]
    y = q[1::3]
    phi = q[2::3]
    return x, y, phi

def _rot(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def _world_point(q: np.ndarray, body_index: int, local_xy: np.ndarray) -> np.ndarray:
    """Map a local point (on body_index) to world coordinates given q. If body_index == -1,
    local_xy is already a world coordinate."""
    if body_index < 0:
        return np.asarray(local_xy, dtype=float)
    x, y, phi = _q_split(q)
    R = _rot(phi[body_index])
    p0 = np.array([x[body_index], y[body_index]])
    return p0 + R @ np.asarray(local_xy, dtype=float)

def _joint_endpoints_at_q(parameters: dict, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return joint endpoint world coords Pi, Pj for a single configuration q.
    Shapes: Pi: (K,2), Pj: (K,2)."""
    ji = np.asarray(parameters["joint_i"], dtype=int)
    jj = np.asarray(parameters["joint_j"], dtype=int)
    si = np.asarray(parameters["si_xy"], dtype=float)
    sj = np.asarray(parameters["sj_xy"], dtype=float)
    K = int(ji.shape[0])
    Pi = np.zeros((K, 2), dtype=float)
    Pj = np.zeros((K, 2), dtype=float)
    for k in range(K):
        Pi[k] = _world_point(q, int(ji[k]), si[k])
        Pj[k] = _world_point(q, int(jj[k]), sj[k])
    return Pi, Pj

def _compute_joint_positions_over_time(parameters: dict, Qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized over frames T (loop in Python but fast enough for plotting).
    Returns Pi, Pj with shape (T, K, 2)."""
    Qs = np.asarray(Qs, dtype=float)
    T = Qs.shape[0]
    ji = np.asarray(parameters["joint_i"], dtype=int)
    K = int(ji.shape[0])
    Pi = np.zeros((T, K, 2), dtype=float)
    Pj = np.zeros((T, K, 2), dtype=float)
    for t in range(T):
        Pi[t], Pj[t] = _joint_endpoints_at_q(parameters, Qs[t])
    return Pi, Pj

def _infer_axes_limits(Pi: np.ndarray, Pj: np.ndarray, pad: float = 0.1) -> Tuple[float, float, float, float]:
    """Compute nice padded limits from two (T,K,2) arrays."""
    pts = np.concatenate([Pi.reshape(-1,2), Pj.reshape(-1,2)], axis=0)
    xmin, ymin = np.nanmin(pts, axis=0)
    xmax, ymax = np.nanmax(pts, axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    half = 0.5 * (1.0 + pad) * max(dx, dy)
    return cx - half, cx + half, cy - half, cy + half

def _body_joint_lookup(parameters: dict, N_hint: Optional[int] = None) -> List[List[Tuple[int, int]]]:
    """For each body b in 0..N-1, collect incident joint endpoints as (k, side).
    side = 0 means use Pi[k]; side = 1 means use Pj[k]."""
    N = _num_bodies_from_params(parameters, fallback_N=N_hint)
    ji = np.asarray(parameters["joint_i"], dtype=int)
    jj = np.asarray(parameters["joint_j"], dtype=int)
    lookup: List[List[Tuple[int,int]]] = [[] for _ in range(N)]
    for k, (a, b) in enumerate(zip(ji, jj)):
        if int(a) >= 0:
            lookup[int(a)].append((k, 0))
        if int(b) >= 0:
            lookup[int(b)].append((k, 1))
    return lookup

def _pick_body_segment(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given Mx2 points on a body, choose two that are farthest apart."""
    if points.shape[0] < 2:
        return None, None  # type: ignore
    # pairwise distances
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    return points[i], points[j]

# ----------------------------
# Public plotting API
# ----------------------------
def animate_mechanism(parameters: dict,
                      times: Optional[np.ndarray],
                      Qs: np.ndarray,
                      *,
                      frame_step: int = 1,
                      traj_joint_indices: Iterable[int] = (),
                      save_path: Optional[str] = None,
                      dpi: int = 120,
                      show: bool = True,
                      figsize: Tuple[float, float] = (7.5, 6.0)) -> FuncAnimation:
    """
    Animate joint endpoints and draw *bodies* as segments between their incident joint points.
    If a body has more than two incident joints, the segment uses the two farthest points.

    parameters : dict with keys: joint_i, joint_j, si_xy, sj_xy
    times : optional (T,) array for title
    Qs : (T, 3N) array of generalized coordinates
    """
    if frame_step < 1:
        frame_step = 1

    Pi, Pj = _compute_joint_positions_over_time(parameters, Qs)
    T, K, _ = Pi.shape
    N = Qs.shape[1] // 3 if Qs.ndim == 2 else _num_bodies_from_params(parameters)

    # precompute per-body incidence
    body_lookup = _body_joint_lookup(parameters, N_hint=N)

    # Axes limits once (stable, equal aspect)
    xmin, xmax, ymin, ymax = _infer_axes_limits(Pi, Pj, pad=0.15)

    # figure setup
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, ls=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    title = ax.set_title("Mechanism animation")

    # plot handles
    (joint_scatter_i,) = ax.plot([], [], "o", ms=4, label="joint i")
    (joint_scatter_j,) = ax.plot([], [], "o", ms=4, label="joint j")

    # one line per body
    body_lines = [ax.plot([], [], "-", lw=3)[0] for _ in range(len(body_lookup))]

    # background path overlays (optional)
    traj_lines = []
    traj_joint_indices = tuple(int(k) for k in traj_joint_indices if 0 <= int(k) < K)
    for k in traj_joint_indices:
        (ln,) = ax.plot(Pi[:, k, 0], Pi[:, k, 1], "--", alpha=0.35, lw=1.5)
        traj_lines.append(ln)

    ax.legend(loc="upper right")

    # Update function
    def _update(frame_idx: int):
        t = int(frame_idx * frame_step)
        if t >= T:
            t = T - 1

        pi = Pi[t]  # (K,2)
        pj = Pj[t]

        # scatter endpoints
        joint_scatter_i.set_data(pi[:, 0], pi[:, 1])
        joint_scatter_j.set_data(pj[:, 0], pj[:, 1])

        # draw bodies
        for line, incidences in zip(body_lines, body_lookup):
            if len(incidences) < 2:
                line.set_data([], [])
                continue
            pts = np.array([pi[k] if side == 0 else pj[k] for (k, side) in incidences], dtype=float)
            a, b = _pick_body_segment(pts)
            if a is None:
                line.set_data([], [])
            else:
                line.set_data([a[0], b[0]], [a[1], b[1]])

        if times is not None:
            title.set_text(f"t = {times[t]:.3f} s")
        return [joint_scatter_i, joint_scatter_j, *body_lines, *traj_lines, title]

    anim = FuncAnimation(fig, _update, frames=math.ceil(T / frame_step), blit=False, interval=30, repeat=False)

    if save_path:
        save_path = str(save_path)
        ext = os.path.splitext(save_path)[1].lower()
        try:
            if ext == ".gif":
                writer = PillowWriter(fps=24)
                anim.save(save_path, writer=writer)
            else:
                # try mp4 via ffmpeg if available, else fallback to GIF
                from matplotlib.animation import FFMpegWriter  # noqa: WPS433
                writer = FFMpegWriter(fps=24, bitrate=1800)
                anim.save(save_path, writer=writer)
        except Exception:
            # always guarantee something is written if requested
            writer = PillowWriter(fps=24)
            anim.save(os.path.splitext(save_path)[0] + ".gif", writer=writer)

    if show:
        plt.show(block=False)
    return anim

def plot_joint_trajectories(parameters: dict,
                            Qs: np.ndarray,
                            *,
                            joints: Iterable[int] = (),
                            show: bool = True,
                            save_path: Optional[str] = None,
                            dpi: int = 120,
                            figsize: Tuple[float, float] = (8.0, 6.0)) -> None:
    """
    Plot path of selected joint first-endpoint positions over time (x-y path).
    If 'joints' is empty, plot all.
    """
    Pi, Pj = _compute_joint_positions_over_time(parameters, Qs)
    T, K, _ = Pi.shape

    if not joints:
        joints = tuple(range(K))
    joints = tuple(int(k) for k in joints if 0 <= int(k) < K)

    plt.figure(figsize=figsize, dpi=dpi)
    for k in joints:
        plt.plot(Pi[:, k, 0], Pi[:, k, 1], label=f"joint {k}")
    plt.axis("equal")
    plt.grid(True, ls=":")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Joint trajectories (first endpoints)")
    if joints:
        plt.legend(loc="best")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show(block=False)

def snapshot(parameters: dict,
             q: np.ndarray,
             *,
             annotate: bool = False,
             dpi: int = 120,
             figsize: Tuple[float, float] = (7.0, 5.5),
             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Draw a single configuration with joints and *bodies* (as segments).
    Returns the Matplotlib Axes used for drawing.
    """
    Pi, Pj = _joint_endpoints_at_q(parameters, q)
    Pi = Pi[None, ...]
    Pj = Pj[None, ...]
    xmin, xmax, ymin, ymax = _infer_axes_limits(Pi, Pj, pad=0.15)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, ls=":")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # draw joints
    ax.plot(Pi[0, :, 0], Pi[0, :, 1], "o", ms=4, label="joint i")
    ax.plot(Pj[0, :, 0], Pj[0, :, 1], "o", ms=4, label="joint j")

    # draw bodies
    N = len(q) // 3
    body_lookup = _body_joint_lookup(parameters, N_hint=N)
    for incidences in body_lookup:
        if len(incidences) < 2:
            continue
        pts = np.array([Pi[0, k] if side == 0 else Pj[0, k] for (k, side) in incidences], dtype=float)
        a, b = _pick_body_segment(pts)
        if a is None:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", lw=3)

    if annotate:
        for k in range(Pi.shape[1]):
            ax.annotate(f"{k}", (Pi[0, k, 0], Pi[0, k, 1]), xytext=(4, 4), textcoords="offset points")

    ax.legend(loc="upper right")
    return ax

# ----------------------------
# Convenience wrappers
# ----------------------------
def animate_from_traj(parameters: dict, traj, **kwargs):
    """
    Traj must provide .times (optional) and .Qs (T,3N). Extra kwargs go to animate_mechanism.
    """
    times = getattr(traj, "times", None)
    Qs = getattr(traj, "Qs", None)
    if Qs is None:
        raise ValueError("Trajectory-like object must have attribute 'Qs' with shape (T, 3N).")
    return animate_mechanism(parameters, times, Qs, **kwargs)

def joint_plots_from_traj(parameters: dict, traj, **kwargs):
    """
    Traj must provide .Qs (T,3N). Extra kwargs go to plot_joint_trajectories.
    """
    Qs = getattr(traj, "Qs", None)
    if Qs is None:
        raise ValueError("Trajectory-like object must have attribute 'Qs' with shape (T, 3N).")
    return plot_joint_trajectories(parameters, Qs, **kwargs)
