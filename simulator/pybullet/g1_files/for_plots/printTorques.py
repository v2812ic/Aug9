import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pb
from config.g1.sim.pybullet.ihwbc.pybullet_params import G1JointIdx

# ==========
#  ESTADO GLOBAL (logs)
# ==========
_JOINT_INFO_READY = False
_JOINT_ORDER = [
    G1JointIdx.left_hip_pitch_joint,
    G1JointIdx.left_hip_roll_joint,
    G1JointIdx.left_hip_yaw_joint,
    G1JointIdx.left_knee_joint,
    G1JointIdx.left_ankle_pitch_joint,
    G1JointIdx.left_ankle_roll_joint,
    G1JointIdx.right_hip_pitch_joint,
    G1JointIdx.right_hip_roll_joint,
    G1JointIdx.right_hip_yaw_joint,
    G1JointIdx.right_knee_joint,
    G1JointIdx.right_ankle_pitch_joint,
    G1JointIdx.right_ankle_roll_joint,
    G1JointIdx.waist_yaw_joint,
    G1JointIdx.left_shoulder_pitch_joint,
    G1JointIdx.left_shoulder_roll_joint,
    G1JointIdx.left_shoulder_yaw_joint,
    G1JointIdx.left_elbow_joint,
    G1JointIdx.left_wrist_roll_joint,
    G1JointIdx.left_wrist_pitch_joint,
    G1JointIdx.left_wrist_yaw_joint,
    G1JointIdx.right_shoulder_pitch_joint,
    G1JointIdx.right_shoulder_roll_joint,
    G1JointIdx.right_shoulder_yaw_joint,
    G1JointIdx.right_elbow_joint,
    G1JointIdx.right_wrist_roll_joint,
    G1JointIdx.right_wrist_pitch_joint,
    G1JointIdx.right_wrist_yaw_joint,
]
_JOINT_LIMITS_DICT = {}      # {pb_idx: (name, max_torque)}
_JOINT_NAMES_ORDER = []      # lista de nombres en el orden de rpc_trq_command
_JOINT_MAX_ORDER = None      # np.array(max_torque) en el mismo orden

_TIME_LOG = []               # [t]
_USAGE_LOG = []              # [(%usage_j for all j in order)]


def _ensure_joint_info(g1_humanoid):
    """Inicializa nombres y límites (una sola vez) en el orden de rpc_trq_command."""
    global _JOINT_INFO_READY, _JOINT_LIMITS_DICT, _JOINT_NAMES_ORDER, _JOINT_MAX_ORDER
    if _JOINT_INFO_READY:
        return

    num_joints = pb.getNumJoints(g1_humanoid)
    # Construir dict con límites por índice de PyBullet
    for j in range(num_joints):
        info = pb.getJointInfo(g1_humanoid, j)
        name = info[1].decode("utf-8")
        max_force = float(info[10])  # límite de motor reportado por PyBullet
        _JOINT_LIMITS_DICT[j] = (name, max_force)

    # Reordenar al orden del comando RPC
    names = []
    maxs = []
    for pb_idx in _JOINT_ORDER:
        name, m = _JOINT_LIMITS_DICT.get(pb_idx, (f"joint_{pb_idx}", 0.0))
        names.append(name)
        maxs.append(m)
    _JOINT_NAMES_ORDER = names
    _JOINT_MAX_ORDER = np.array(maxs, dtype=float)

    _JOINT_INFO_READY = True


def printTorques(g1_humanoid, t, rpc_trq_command, flag_torque_limit,
                 warn_frac=0.05, verbose=True, log_usage=True):
    """
    - Imprime líneas cuando una articulación supera 'warn_frac' del límite.
    - Registra %uso por articulación para graficar luego con plotTorques().
    """
    _ensure_joint_info(g1_humanoid)

    # %uso por articulación en el orden de rpc_trq_command
    rpc_trq_command = np.asarray(rpc_trq_command, dtype=float).reshape(-1)
    max_limits = _JOINT_MAX_ORDER
    usage_pct = np.zeros_like(rpc_trq_command)

    for i, trq in enumerate(rpc_trq_command):
        m = max_limits[i]
        if m <= 1e-9:
            usage_pct[i] = np.nan  # sin límite válido -> evita división por 0
        else:
            usage_pct[i] = 100.0 * abs(trq) / m

    # Logging temporal
    if log_usage:
        _TIME_LOG.append(float(t))
        _USAGE_LOG.append(usage_pct.copy())

    # Impresión de advertencias
    if verbose:
        names = _JOINT_NAMES_ORDER
        max_torque_limit = 0.0
        for i, pct in enumerate(usage_pct):
            if np.isnan(pct):
                continue
            if pct > warn_frac * 100.0:
                name = names[i]
                trq = rpc_trq_command[i]
                if abs(pct) > max_torque_limit:
                    max_torque_limit = abs(pct)
                print(f"{name:25s}: {trq:7.2f} Nm / {pct:6.1f}% usage")
                if pct >= 100.0 - 1e-5 and not flag_torque_limit:
                    print("** Limite de par alcanzado **")
                    flag_torque_limit = True
        if max_torque_limit > 0.0:
            print("")  # línea en blanco estética

import itertools

STYLE_GUIDE = {
    'colors': {
        'blue': '#4169E1',
        'green': '#3CB371',
        'orange': '#FF8C00',
        'red': '#DC143C',
        'purple': '#BA55D3',
        'text': '#333333',
        'grid': '#CCCCCC',
        'line_light': '#D3D3D3'
    },
    'fonts': {
        'title': 16,
        'label': 12,
        'ticks': 10,
        'legend': 10,
        'suptitle': 18
    },
    'lines': {
        'width': 1.8,
        'grid_style': '--'
    },
    'figure': {
        'size_wide': (10, 5),
        'size_tall': (12, 10)
    }
}

def plotTorques(outdir="plots/torques", topk=8, include_heatmap=True,
                plot_from_zero=True, plot_from_time=None):
    """
    Generate torque usage plots with unified style:
      1) Time series of Top-K joints by peak usage.
      2) Bar chart of peak usage.
      3) Curve of maximum usage over time.
      4) (Optional) Heatmap of usage.

    Args:
        outdir (str): output folder
        topk (int): number of top joints to plot
        include_heatmap (bool): include heatmap plot
        plot_from_zero (bool): if False, start plotting from t >= 2.0s
        plot_from_time (float|None): si no es None, se usa como tiempo inicial.
                                     Tiene prioridad sobre plot_from_zero.
    """
    if len(_TIME_LOG) == 0 or len(_USAGE_LOG) == 0:
        print("[plotTorques] No torque data logged.")
        return

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # --- data ---
    t_all = np.asarray(_TIME_LOG, dtype=float).reshape(-1)
    U_all = np.asarray(_USAGE_LOG, dtype=float)  # (T, D)
    names = _JOINT_NAMES_ORDER

    # --- time window (solo desde el inicio) ---
    if plot_from_time is not None:
        t0 = float(plot_from_time)
    else:
        t0 = 0.0 if plot_from_zero else 2.0

    mask = (t_all >= t0)
    if not np.any(mask):
        print(f"[plotTorques] No samples with t >= {t0:.2f}s; using all.")
        mask = np.ones_like(t_all, dtype=bool)

    t = t_all[mask]
    U = U_all[mask]
    T, D = U.shape
    U_plot = np.nan_to_num(U, nan=0.0)

    # Top-K joints by peak usage
    peaks = np.nanmax(U, axis=0)
    idx_sorted = np.argsort(peaks)[::-1]
    k = min(topk, D)
    idx_top = idx_sorted[:k]
    names_top = [names[i] for i in idx_top]

    # --- 1a) Time series Top-K ---
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    color_cycle = itertools.cycle([
        STYLE_GUIDE['colors']['blue'],
        STYLE_GUIDE['colors']['green'],
        STYLE_GUIDE['colors']['orange'],
        STYLE_GUIDE['colors']['red'],
        STYLE_GUIDE['colors']['purple']
    ])
    for i, j in enumerate(idx_top):
        ax.plot(t, U_plot[:, j],
                lw=STYLE_GUIDE['lines']['width'],
                label=f"{names_top[i]} (max {peaks[j]:.1f}%)",
                color=next(color_cycle))
    ax.set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Torque usage [%]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Top-{k} torque usage",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    ax.legend(fontsize=STYLE_GUIDE['fonts']['legend'], loc="upper right")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"torque_usage_top{k}_timeseries.pdf"))
    plt.close(fig)

    # --- 1b) Bar chart ---
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(np.arange(k), peaks[idx_top], color=STYLE_GUIDE['colors']['blue'])
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(names_top, rotation=40, ha="right")
    ax.set_ylabel("Peak usage [%]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Top-{k} peak torque usage",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(axis="y", linestyle=":", color=STYLE_GUIDE['colors']['grid'])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"torque_usage_top{k}_peaks.pdf"))
    plt.close(fig)

    # --- 2) Max over time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, np.nanmax(U, axis=1),
            lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['red'])
    ax.set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Max joint usage [%]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title("Maximum torque usage over joints",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "torque_usage_max_over_time.pdf"))
    plt.close(fig)

    # --- 3) Heatmap (optional) ---
    if include_heatmap:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(U_plot.T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], -0.5, D - 0.5],
                       cmap="viridis")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Torque usage [%]", fontsize=STYLE_GUIDE['fonts']['label'])
        ax.set_yticks(np.arange(D))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
        ax.set_ylabel("Joint", fontsize=STYLE_GUIDE['fonts']['label'])
        ax.set_title("Torque usage heatmap",
                     fontsize=STYLE_GUIDE['fonts']['title'],
                     color=STYLE_GUIDE['colors']['text'])
        fig.tight_layout()
        plt.savefig(os.path.join(outdir, "torque_usage_heatmap.pdf"))
        plt.close(fig)
