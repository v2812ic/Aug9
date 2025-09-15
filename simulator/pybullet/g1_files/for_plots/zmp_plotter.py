import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.path import Path

# -----------------------------
# Style Guide (Based on LaTeX tcolorbox)
# -----------------------------
STYLE_GUIDE = {
    'colors': {
        'blue': '#003B80',    # theorem: blue!75!black
        'green': '#006442',   # definition: green!60!black
        'orange': '#BF5700',  # lemma: orange!75!black
        'red': '#992323',     # proposition: red!60!black
        'text': '#333333',
        'grid': '#CCCCCC',
        'polygon': '#404040'
    },
    'fonts': {
        'title': 14,
        'label': 12,
        'ticks': 10,
        'legend': 10,
        'suptitle': 16
    },
    'lines': {
        'grid_style': '--',
    },
    'figure': {
        'size_zmp': (14, 7),   # A bit wider for the colorbar
        'size_timeline': (10, 3)
    },
    'colormap': 'viridis'     # Colormap for time gradient
}

# -----------------------------
# Helpers
# -----------------------------
def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])

def _sum_wrenches_per_timestep(list_of_lists):
    # Suma por timestep; si la lista interna está vacía -> zeros(6)
    return np.array([np.sum(np.asarray(L), axis=0) if L else np.zeros(6) for L in list_of_lists])

def _apply_time_window_seq(seq, mask):
    """Aplica la misma máscara temporal a secuencias indexables (listas)"""
    if seq is None:
        return None
    return [item for item, keep in zip(seq, mask) if keep]

def _nan_minmax(arr):
    """Devuelve (min, max) ignorando NaNs; si todo es NaN, devuelve (0, 1)"""
    a = np.asarray(arr)
    finite = np.isfinite(a)
    if not np.any(finite):
        return 0.0, 1.0
    return float(np.nanmin(a[finite])), float(np.nanmax(a[finite]))

# -----------------------------
# Main
# -----------------------------
def plot_zmp(time, cf_left_log, cf_right_log,
             pos_left_log, ori_left_log, pos_right_log, ori_right_log,
             outdir="plots/zmp", plot_from_time=None, plot_from_zero=True):
    """
    Genera los plots de ZMP en el marco local de cada pie con gradiente temporal.

    Ventana temporal (desde el inicio):
      - Si plot_from_time es not None, se usa ese instante (s).
      - En caso contrario: 0.0 s si plot_from_zero=True, 2.0 s si False.
    """
    print("--- Generating ZMP plots (Time-Encoded Version) ---")
    os.makedirs(outdir, exist_ok=True)

    # ---- Tiempo y ventana ----
    time_arr_full = np.asarray(time, dtype=float).reshape(-1)
    if time_arr_full.size == 0:
        print("[plot_zmp] Empty time array.")
        return

    t0 = float(plot_from_time) if plot_from_time is not None else (0.0 if plot_from_zero else 2.0)
    mask = time_arr_full >= t0
    if not np.any(mask):
        print(f"[plot_zmp] No samples with t >= {t0:.3f}s; using all.")
        mask = np.ones_like(time_arr_full, dtype=bool)

    time_arr = time_arr_full[mask]
    cf_left_log  = _apply_time_window_seq(cf_left_log,  mask)
    cf_right_log = _apply_time_window_seq(cf_right_log, mask)
    pos_left_log  = _apply_time_window_seq(pos_left_log,  mask)
    pos_right_log = _apply_time_window_seq(pos_right_log, mask)
    ori_left_log  = _apply_time_window_seq(ori_left_log,  mask)
    ori_right_log = _apply_time_window_seq(ori_right_log, mask)

    # ---- Wrenches sumados por timestep ----
    cfL_W = _sum_wrenches_per_timestep(cf_left_log)
    cfR_W = _sum_wrenches_per_timestep(cf_right_log)

    # Polígono de soporte (local)
    foot_poly = np.array([
        [ 0.09,  0.02],
        [ 0.09, -0.02],
        [-0.05, -0.02],
        [-0.05,  0.02],
        [ 0.09,  0.02]
    ])

    T = len(time_arr)
    zmpL, zmpR = np.full((T, 2), np.nan), np.full((T, 2), np.nan)

    # ---- ZMP por pie (en marco local) ----
    for i in range(T):
        R_LW = _quat_to_R(np.array(ori_left_log[i])).T
        R_RW = _quat_to_R(np.array(ori_right_log[i])).T

        fL_L, mL_L = R_LW @ cfL_W[i, :3], R_LW @ cfL_W[i, 3:6]
        fR_R, mR_R = R_RW @ cfR_W[i, :3], R_RW @ cfR_W[i, 3:6]

        if abs(fL_L[2]) > 1e-3:
            zmpL[i] = [-mL_L[1] / fL_L[2],  mL_L[0] / fL_L[2]]
        if abs(fR_R[2]) > 1e-3:
            zmpR[i] = [-mR_R[1] / fR_R[2],  mR_R[0] / fR_R[2]]

    # ---- Límites de ejes (incluyen pie + ZMP, ignorando NaNs) ----
    x_min_L, x_max_L = _nan_minmax(zmpL[:, 0])
    y_min_L, y_max_L = _nan_minmax(zmpL[:, 1])
    x_min_R, x_max_R = _nan_minmax(zmpR[:, 0])
    y_min_R, y_max_R = _nan_minmax(zmpR[:, 1])

    x_lim_min = min(foot_poly[:, 0].min(), x_min_L, x_min_R) - 0.01
    x_lim_max = max(foot_poly[:, 0].max(), x_max_L, x_max_R) + 0.01
    y_lim_min = min(foot_poly[:, 1].min(), y_min_L, y_min_R) - 0.01
    y_lim_max = max(foot_poly[:, 1].max(), y_max_L, y_max_R) + 0.01

    # ---- Figura ZMP con gradiente temporal ----
    fig, axes = plt.subplots(1, 2, figsize=STYLE_GUIDE['figure']['size_zmp'], sharey=True)
    fig.suptitle('ZMP Trajectory in Local Foot Frame', fontsize=STYLE_GUIDE['fonts']['suptitle'], color=STYLE_GUIDE['colors']['text'])

    # Pie izquierdo
    axes[0].plot(foot_poly[:, 0], foot_poly[:, 1], color=STYLE_GUIDE['colors']['polygon'], label='Support Polygon')
    scL = axes[0].scatter(zmpL[:, 0], zmpL[:, 1], s=15, c=time_arr, cmap=STYLE_GUIDE['colormap'],
                          vmin=time_arr[0], vmax=time_arr[-1])
    axes[0].set_title('Left Foot', fontsize=STYLE_GUIDE['fonts']['title'])
    axes[0].set_xlabel('Local X [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[0].set_ylabel('Local Y [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    axes[0].legend(loc='upper left', fontsize=STYLE_GUIDE['fonts']['legend'])
    axes[0].set_xlim(x_lim_min, x_lim_max)
    axes[0].set_ylim(y_lim_min, y_lim_max)

    # Pie derecho
    axes[1].plot(foot_poly[:, 0], foot_poly[:, 1], color=STYLE_GUIDE['colors']['polygon'])
    scR = axes[1].scatter(zmpR[:, 0], zmpR[:, 1], s=15, c=time_arr, cmap=STYLE_GUIDE['colormap'],
                          vmin=time_arr[0], vmax=time_arr[-1])
    axes[1].set_title('Right Foot', fontsize=STYLE_GUIDE['fonts']['title'])
    axes[1].set_xlabel('Local X [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    axes[1].set_xlim(x_lim_min, x_lim_max)
    axes[1].set_ylim(y_lim_min, y_lim_max)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(scR, cax=cbar_ax, label='Time [s]')

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # espacio para la colorbar
    fig.savefig(f"{outdir}/zmp_trajectory_local.pdf")
    plt.close(fig)

    # (opcional) Timeline de estabilidad -> comentar/añadir si vuelves a calcular insideL/insideR

    print(f" -> ZMP plots saved in '{outdir}'.")
