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
    - Registra %uso por articulación para graficar luego con plot_torques().
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


def plotTorques(outdir="plots", topk=8, include_heatmap=True, start_time=1.0):
    """
    Genera:
      1) Series temporales de las Top-K articulaciones por pico de uso (t >= start_time).
      2) Barras con picos de uso de esas Top-K (t >= start_time).
      3) Curva del máximo global en cada instante (t >= start_time).
      4) (opcional) Heatmap T x D del % de uso (t >= start_time).

    Llama a esta función al final de la simulación (o en tu finalize()).
    """
    if len(_TIME_LOG) == 0 or len(_USAGE_LOG) == 0:
        print("[plotTorques] No hay datos de par registrados.")
        return

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # --- datos completos ---
    t_all = np.asarray(_TIME_LOG, dtype=float).reshape(-1)
    U_all = np.asarray(_USAGE_LOG, dtype=float)  # (T, D)
    names = _JOINT_NAMES_ORDER

    # --- recorte por tiempo ---
    start_time = float(start_time)
    mask = t_all >= start_time
    if not np.any(mask):
        print(f"[plotTorques] No hay muestras con t >= {start_time:.2f}s; se usan todas.")
        mask = np.ones_like(t_all, dtype=bool)

    t = t_all[mask]
    U = U_all[mask]            # (T_cut, D)
    T, D = U.shape

    # Reemplaza NaN por 0 para gráficos (solo visual)
    U_plot = np.nan_to_num(U, nan=0.0)

    # 1) Top-K por pico (solo en t>=start_time)
    peaks = np.nanmax(U, axis=0)  # pico por articulación
    idx_sorted = np.argsort(peaks)[::-1]
    k = min(topk, D)
    idx_top = idx_sorted[:k]
    names_top = [names[i] for i in idx_top]

    # 1a) Series temporales Top-K
    plt.figure(figsize=(11, 5))
    for i, j in enumerate(idx_top):
        plt.plot(t, U_plot[:, j], lw=1.8, label=f"{names_top[i]} (max {peaks[j]:.1f}%)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque usage (%)")
    plt.title(f"Top-{k} torque usage (t >= {start_time:.2f}s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"torque_usage_top{k}_timeseries.png"), dpi=300)
    plt.close()

    # 1b) Barras de picos Top-K
    plt.figure(figsize=(11, 4))
    x = np.arange(k)
    plt.bar(x, peaks[idx_top])
    plt.xticks(x, names_top, rotation=40, ha="right")
    plt.ylabel("Peak usage (%)")
    plt.title(f"Top-{k} peak torque usage (t >= {start_time:.2f}s)")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"torque_usage_top{k}_peaks.png"), dpi=300)
    plt.close()

    # 2) Máximo global por instante
    max_over_time = np.nanmax(U, axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(t, max_over_time, lw=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Max joint usage (%)")
    plt.title(f"Maximum torque usage over joints (t >= {start_time:.2f}s)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "torque_usage_max_over_time.png"), dpi=300)
    plt.close()

    # 3) Heatmap (opcional)
    if include_heatmap:
        plt.figure(figsize=(12, 6))
        im = plt.imshow(U_plot.T, aspect="auto", origin="lower",
                        extent=[t[0], t[-1], -0.5, D - 0.5])
        plt.colorbar(im, label="Torque usage (%)")
        plt.yticks(np.arange(D), names, fontsize=7)
        plt.xlabel("Time (s)")
        plt.ylabel("Joint")
        plt.title(f"Torque usage heatmap (t >= {start_time:.2f}s)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "torque_usage_heatmap.png"), dpi=300)
        plt.close()