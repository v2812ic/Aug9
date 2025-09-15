import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow
import matplotlib as mpl


# -----------------------------
# Style Guide (Consistent with other plotters)
# -----------------------------
STYLE_GUIDE = {
    'colors': {
        'blue': '#4169E1',      # RoyalBlue
        'green': '#3CB371',     # MediumSeaGreen
        'orange': '#FF8C00',    # DarkOrange
        'red': '#DC143C',       # Crimson
        'purple': '#BA55D3',    # MediumOrchid
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

# LaTeX symbol for tau, for use in plots
LATEX_TAU = r"$\tau$"
TAU_EXT = r"$\tau_{\mathrm{ext}}$"

# -----------------------------
# Global Labels and Units
# -----------------------------
LABELS_FORCE = [r"$F_x$", r"$F_y$", r"$F_z$"]
LABELS_MOMENT = [r"$M_x$", r"$M_y$", r"$M_z$"]
LABELS_WRENCH = [
    r"$F_x$ [N]", r"$F_y$ [N]", r"$F_z$ [N]",
    r"$M_x$ [N·m]", r"$M_y$ [N·m]", r"$M_z$ [N·m]"
]
LABELS_ERROR_ABS = [
    r"$|\Delta F_x|$", r"$|\Delta F_y|$", r"$|\Delta F_z|$",
    r"$|\Delta M_x|$", r"$|\Delta M_y|$", r"$|\Delta M_z|$"
]
LABELS_ERROR_REL = [
    r"$|\Delta F_x|$ rel", r"$|\Delta F_y|$ rel", r"$|\Delta F_z|$ rel",
    r"$|\Delta M_x|$ rel", r"$|\Delta M_y|$ rel", r"$|\Delta M_z|$ rel"
]

COMP_COLORS = {
    # Fuerzas
    "Fx": STYLE_GUIDE['colors']['blue'],
    "Fy": STYLE_GUIDE['colors']['green'],
    "Fz": STYLE_GUIDE['colors']['orange'],
    # Momentos
    "Mx": STYLE_GUIDE['colors']['blue'],
    "My": STYLE_GUIDE['colors']['green'],
    "Mz": STYLE_GUIDE['colors']['orange'],
}

# === Paletas específicas para errores ===
ERROR_COLORS = [
    STYLE_GUIDE['colors']['blue'],
    STYLE_GUIDE['colors']['green'],
    STYLE_GUIDE['colors']['orange'],
    STYLE_GUIDE['colors']['purple'],  # ΔMx
    STYLE_GUIDE['colors']['red'],     # ΔMy
    '#8A8A8A',                        # ΔMz (gris medio)
]
ERROR_LINESTYLES_ABS = ['-', '-', '-', '-', '-', '-']
ERROR_LINESTYLES_REL = ['-', '-', '-', '-', '-', '-'] # Ahora todas sólidas para claridad en log


def _plot_real_vs_est(t, real, est, kind, filename, title,
                      ylabel, outdir,
                      labels_real=(r"$F_x^{real}$", r"$F_y^{real}$", r"$F_z^{real}$"),
                      labels_est=(r"$F_x^{est}$",  r"$F_y^{est}$",  r"$F_z^{est}$")):
    """
    """
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt"))
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    # Paletas de colores separadas para mayor claridad
    COLORS_REAL = {
        "Fx": '#00008B', "Fy": '#006400', "Fz": '#A52A2A', # Dark Blue, Green, Brown
        "Mx": '#00008B', "My": '#006400', "Mz": '#A52A2A',
    }
    COLORS_EST = COMP_COLORS # La paleta original más brillante

    comps = ["x", "y", "z"]
    for i, c in enumerate(comps):
        base = (('F' if kind=='F' else 'M') + c)
        # Real: línea sólida, color oscuro
        ax.plot(t, real[:, i],
                label=labels_real[i],
                color=COLORS_REAL[base],
                linewidth=STYLE_GUIDE['lines']['width'])
        # Estimado: línea sólida, color más brillante
        ax.plot(t, est[:, i],
                label=labels_est[i],
                color=COLORS_EST[base],
                linewidth=STYLE_GUIDE['lines']['width'])

    ax.set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel(ylabel, fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(title, fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])

    ax.legend(loc='best', fontsize=STYLE_GUIDE['fonts']['legend'], ncol=2)
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close(fig)

# -----------------------------
# Utilities
# -----------------------------
def _as_np(x):
    if x is None: return None
    try: return np.asarray(x)
    except Exception: return np.array(list(x))

def _ensure_2d(series):
    if series is None: return None
    arr = _as_np(series)
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    elif arr.ndim > 2: arr = arr.reshape(arr.shape[0], -1)
    return arr

def _resolve_time(time, length, dt=None):
    if time is not None:
        t = _as_np(time).astype(float)
        if len(t) != length: raise ValueError(f"Time vector length {len(t)} != data length {length}")
        return t
    if dt is None: return np.arange(length, dtype=float)
    return np.arange(length, dtype=float) * float(dt)

def _safe_makedirs_for(filename):
    d = os.path.dirname(filename) or "."
    os.makedirs(d, exist_ok=True)
# -----------------------------
# Quaternions / Orientation
# -----------------------------
def _quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0.0] = 1.0
    return q / n

def _quat_to_euler_zyx(q_xyzw, degrees=True):
    q = _quat_normalize(q_xyzw)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (w * z + x * y); cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    sinr_cosp = 2.0 * (w * x + y * z); cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    if degrees: k = 180.0 / np.pi; return yaw * k, pitch * k, roll * k
    return yaw, pitch, roll

def _quat_series_to_euler_zyx(quat_series, degrees=True):
    Q = _ensure_2d(quat_series)
    if Q is None or Q.shape[1] != 4: return None
    YPR = np.zeros((Q.shape[0], 3))
    for i in range(Q.shape[0]):
        YPR[i] = np.array(_quat_to_euler_zyx(Q[i], degrees=degrees))
    return YPR

# -----------------------------
# Base Plotting Functions
# -----------------------------
def _plot_vector_series(t, Y, title, xlabel, ylabel, labels, filename, yscale=None,
                        colors=None, linestyles=None):
    Y = _ensure_2d(Y)
    if Y is None or Y.size == 0: return
    T, D = Y.shape
    t = np.asarray(t).reshape(-1)
    if t.size != T:
        if Y.T.shape[0] == t.size: Y = Y.T; T, D = Y.shape
        else: raise ValueError(f"len(t)={t.size} vs Y.shape={Y.shape}")
    if labels is None: labels = [f"Dim {i}" for i in range(D)]
    elif isinstance(labels, str): labels = [labels]
    if len(labels) < D: labels += [f"Dim {i}" for i in range(len(labels), D)]
    elif len(labels) > D: labels = labels[:D]
    _safe_makedirs_for(filename)
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    from matplotlib import cm
    default_colors = [cm.get_cmap('tab20')(i) for i in range(20)]
    default_linestyles = ['-', '--', '-.', ':']
    for i in range(D):
        c = colors[i] if colors is not None and i < len(colors) else default_colors[i % 20]
        ls = linestyles[i] if linestyles is not None and i < len(linestyles) else default_linestyles[(i // 20) % len(default_linestyles)]
        ax.plot(t, Y[:, i], label=labels[i], color=c, linestyle=ls, linewidth=STYLE_GUIDE['lines']['width'])
    ax.set_xlabel(xlabel, fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel(ylabel, fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(title, fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    if D > 1 or (D == 1 and labels[0] is not None):
        ncol = 1 if D <= 6 else 2 if D <= 12 else 3
        ax.legend(loc='best', fontsize=STYLE_GUIDE['fonts']['legend'], ncol=ncol)
    if yscale: ax.set_yscale(yscale)
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# -----------------------------
# Contact logs: forces and moments
# -----------------------------
def _extract_wrench_component(contact_log, start_idx):
    N = len(contact_log)
    component = np.zeros((4, N, 3))
    for i, frame in enumerate(contact_log):
        for j in range(min(4, len(frame))):
            item = frame[j]
            try: vec = np.asarray(item, dtype=float)
            except Exception: continue
            if vec.ndim == 1 and vec.size >= start_idx + 3:
                component[j, i, :] = vec[start_idx:start_idx+3]
    return component
def _extract_forces_from_contact_log(contact_log): return _extract_wrench_component(contact_log, 0)
def _extract_moments_from_contact_log(contact_log): return _extract_wrench_component(contact_log, 3)
def _apply_time_window(t, arr_like, start_time=0.0):
    t = np.asarray(t).reshape(-1)
    mask = t >= float(start_time)
    if arr_like is None: return t[mask], None
    arr = np.asarray(arr_like)
    if arr.ndim == 2 and arr.shape[0] == t.size: return t[mask], arr[mask]
    if arr.ndim == 3 and arr.shape[1] == t.size: return t[mask], arr[:, mask, :]
    return t, arr_like

# -----------------------------
# Plot foot contact wrenches
# -----------------------------
def _plot_foot_contact_wrench(time_arr, wrench_data, component_type,
                              foot_name, outdir, plot_from_zero=True):
    start_time = 0.0 if plot_from_zero else 2.0
    t, W = _apply_time_window(time_arr, wrench_data, start_time=start_time)
    is_force = component_type.lower() == 'force'
    unit = "[N]" if is_force else "[N·m]"
    labels = [r"$F_x$", r"$F_y$", r"$F_z$"] if is_force else [r"$M_x$", r"$M_y$", r"$M_z$"]
    fig = plt.figure(figsize=STYLE_GUIDE['figure']['size_tall'])
    gs = fig.add_gridspec(4, 2, width_ratios=[4, 1], hspace=0.35)
    fig.suptitle(f"Contact {component_type}s – {foot_name} Foot", fontsize=STYLE_GUIDE['fonts']['suptitle'], color=STYLE_GUIDE['colors']['text'])
    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    color_cycle_base = [STYLE_GUIDE['colors']['blue'], STYLE_GUIDE['colors']['green'], STYLE_GUIDE['colors']['orange']]
    for i in range(4):
        for c in range(3):
            axs[i].plot(t, W[i, :, c], label=labels[c], lw=STYLE_GUIDE['lines']['width'], color=color_cycle_base[c])
        axs[i].set_ylabel(f"{component_type} [{i + 1}] {unit}", fontsize=STYLE_GUIDE['fonts']['label'])
        if i == 0: axs[i].legend(loc='upper right', fontsize=STYLE_GUIDE['fonts']['legend'])
        axs[i].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
        axs[i].tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
        axs[i].set_xlim(t[0], t[-1])
    axs[-1].set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])
    ax_schematic = fig.add_subplot(gs[:, 1])
    hx, hy = 0.1, 0.04
    rect = patches.Rectangle((-hy, -hx), 2*hy, 2*hx, linewidth=1, edgecolor=STYLE_GUIDE['colors']['text'], facecolor=STYLE_GUIDE['colors']['line_light'])
    ax_schematic.add_patch(rect)
    poc_positions = [(-hy, -hx), (-hy, hx), (hy, hx), (hy, -hx)]
    if foot_name == "Left": poc_positions = [(hy, -hx), (hy, hx), (-hy, hx), (-hy, -hx)]
    for i, (x, y) in enumerate(poc_positions):
        ax_schematic.plot(x, y, 'o', color=STYLE_GUIDE['colors']['blue'])
        
        x_offset = 0.020 if x >= 0 else -0.020
        y_offset = 0.015 if y >= 0 else -0.015
        ha = 'left' if x >= 0 else 'right'

        ax_schematic.text(
            x + x_offset, y + y_offset,
            f'{i+1}',
            fontsize=12,
            color=STYLE_GUIDE['colors']['text'],
            va='center',
            ha=ha
        )

    arrow = FancyArrow(0, 0, 0, 0.15, width=0.005, head_width=0.015, head_length=0.015, color=STYLE_GUIDE['colors']['blue'])
    ax_schematic.add_patch(arrow)
    ax_schematic.text(0.02, 0.18, '+X (Front)', color=STYLE_GUIDE['colors']['text'], fontsize=10, ha='center')
    ax_schematic.set_xlim(-0.08, 0.08); ax_schematic.set_ylim(-0.15, 0.22)
    ax_schematic.set_aspect('equal'); ax_schematic.set_title("Points of Contact\n(Top View)", fontsize=STYLE_GUIDE['fonts']['label']); ax_schematic.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt"))
    plt.savefig(os.path.join(outdir, f'contact_{component_type.lower()}_{foot_name.lower()}.pdf'))
    plt.close(fig)

# -----------------------------
# Total GRF/GRM and per-foot plots
# -----------------------------
def _plot_total_grf_xyz(time_arr, forces_left, forces_right, outdir, total_mass_hint=35.1151):
    t = np.asarray(time_arr).reshape(-1); F_L = forces_left.sum(axis=0); F_R = forces_right.sum(axis=0); F_T = F_L + F_R
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    ax.plot(t, F_T[:, 0], label=r"$F_x$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, F_T[:, 1], label=r"$F_y$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, F_T[:, 2], label=r"$F_z$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['orange'])
    weight = float(total_mass_hint) * 9.81
    ax.plot(t, np.full_like(t, weight), STYLE_GUIDE['lines']['grid_style'], label="Weight", linewidth=1.2, color=STYLE_GUIDE['colors']['red'])
    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label']); ax.set_ylabel("Force [N]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title("Total Ground Reaction Force", fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks']); ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend']); fig.tight_layout()
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt")); plt.savefig(os.path.join(outdir, "grf_total_xyz.pdf")); plt.close(fig)

def _plot_total_grm_xyz(time_arr, moments_left, moments_right, outdir):
    t = np.asarray(time_arr).reshape(-1); M_L = moments_left.sum(axis=0); M_R = moments_right.sum(axis=0); M_T = M_L + M_R
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    ax.plot(t, M_T[:, 0], label=r"$M_x$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, M_T[:, 1], label=r"$M_y$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, M_T[:, 2], label=r"$M_z$", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['orange'])
    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label']); ax.set_ylabel("Moment [N·m]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title("Total Ground Reaction Moment", fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks']); ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend']); fig.tight_layout()
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt")); plt.savefig(os.path.join(outdir, "grm_total_xyz.pdf")); plt.close(fig)

def _plot_grf_xyz_one_foot(time_arr, forces_foot, foot_name, outdir, total_mass_hint=35.1151):
    t = np.asarray(time_arr).reshape(-1); F_FOOT = forces_foot.sum(axis=0)
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    ax.plot(t, F_FOOT[:, 0], label=rf"$F_x$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, F_FOOT[:, 1], label=rf"$F_y$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, F_FOOT[:, 2], label=rf"$F_z$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['orange'])
    weight = float(total_mass_hint) * 9.81
    ax.plot(t, np.full_like(t, weight/2.0), STYLE_GUIDE['lines']['grid_style'], label="Weight/2", linewidth=1.2, color=STYLE_GUIDE['colors']['red'])
    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label']); ax.set_ylabel("Force [N]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Ground Reaction Force – {foot_name} Foot", fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks']); ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend']); fig.tight_layout()
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt")); plt.savefig(os.path.join(outdir, f"grf_{foot_name.lower()}_xyz.pdf")); plt.close(fig)

def _plot_grm_xyz_one_foot(time_arr, moments_foot, foot_name, outdir):
    t = np.asarray(time_arr).reshape(-1); M_FOOT = moments_foot.sum(axis=0)
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])
    ax.plot(t, M_FOOT[:, 0], label=rf"$M_x$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, M_FOOT[:, 1], label=rf"$M_y$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, M_FOOT[:, 2], label=rf"$M_z$ {foot_name}", lw=STYLE_GUIDE['lines']['width'], color=STYLE_GUIDE['colors']['orange'])
    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label']); ax.set_ylabel("Moment [N·m]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Ground Reaction Moment – {foot_name} Foot", fontsize=STYLE_GUIDE['fonts']['title'], color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks']); ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend']); fig.tight_layout()
    _safe_makedirs_for(os.path.join(outdir, "dummy.txt")); plt.savefig(os.path.join(outdir, f"grm_{foot_name.lower()}_xyz.pdf")); plt.close(fig)

# -----------------------------
# Public API
# -----------------------------
def make_plots(
    outdir="plots/general", *, time=None, dt=None, tau_ext=None, cf_left_log=None, cf_right_log=None,
    pos_left_log=None, pos_right_log=None, ori_left_log=None, ori_right_log=None, f_ext_log=None,
    f_real_log=None, f_A_r_log=None, f_D_r_log=None, plot_from_zero=True, plot_from_time=None, **_ignore):

    os.makedirs(outdir, exist_ok=True)
    print("--- Generating all plots...")

    # ---------- τ_ext ----------
    if tau_ext is not None and len(tau_ext) > 0:
        tau_ext_arr = _ensure_2d(tau_ext)
        T = tau_ext_arr.shape[0]
        t = _resolve_time(time, T, dt)
        # ventana
        t, tau_ext_arr = _apply_time_window(t, tau_ext_arr, start_time=plot_from_time)

        try:
            tau_norm = np.linalg.norm(tau_ext_arr, axis=1)
            _plot_vector_series(
                t, tau_norm.reshape(-1, 1),
                title=f"External Torque Norm ({TAU_EXT})",
                xlabel="Time [s]", ylabel=r"$\|\tau_{\mathrm{ext}}\|$ [SI]",
                labels=[r"$\|\tau_{\mathrm{ext}}\|$"],
                filename=os.path.join(outdir, "tau_ext_norm.pdf")
            )
        except Exception as e:
            print(f"[plots] Error plotting τ_ext norm: {e}")

        _plot_vector_series(
            t, tau_ext_arr,
            title=f"External Torque Components ({TAU_EXT})",
            xlabel="Time [s]", ylabel="Torque [N·m]",
            labels=None, filename=os.path.join(outdir, "tau_ext_components.pdf")
        )
        if tau_ext_arr.shape[1] >= 3:
            _plot_vector_series(
                t, tau_ext_arr[:, :3],
                title=f"{TAU_EXT} – Forces",
                xlabel="Time [s]", ylabel="Force [N]",
                labels=[r"$F_x$", r"$F_y$", r"$F_z$"],
                filename=os.path.join(outdir, "tau_ext_forces_first3.pdf")
            )
        if tau_ext_arr.shape[1] >= 6:
            _plot_vector_series(
                t, tau_ext_arr[:, 3:6],
                title=f"{TAU_EXT} – Moments",
                xlabel="Time [s]", ylabel="Moment [N·m]",
                labels=[r"$M_x$", r"$M_y$", r"$M_z$"],
                filename=os.path.join(outdir, "tau_ext_moments_next3.pdf")
            )

    # ---------- Contactos (fuerzas y momentos) ----------
    if cf_left_log is not None and cf_right_log is not None:
        T_forces = len(cf_left_log)
        t_forces = _resolve_time(time, T_forces, dt)

        cf_left  = _extract_forces_from_contact_log(cf_left_log)
        cf_right = _extract_forces_from_contact_log(cf_right_log)
        # ventana (nota: mismas máscaras para ambos usando el mismo t)
        t_fw, cf_left  = _apply_time_window(t_forces, cf_left,  start_time=plot_from_time)
        _ ,   cf_right = _apply_time_window(t_forces, cf_right, start_time=plot_from_time)

        # ya no recortamos dentro; les pasamos series ya recortadas
        _plot_foot_contact_wrench(t_fw, cf_left,  "Force", "Left",  outdir, plot_from_zero=True)
        _plot_foot_contact_wrench(t_fw, cf_right, "Force", "Right", outdir, plot_from_zero=True)
        _plot_total_grf_xyz(t_fw, cf_left, cf_right, outdir)
        _plot_grf_xyz_one_foot(t_fw, cf_left,  "Left",  outdir)
        _plot_grf_xyz_one_foot(t_fw, cf_right, "Right", outdir)

        cm_left  = _extract_moments_from_contact_log(cf_left_log)
        cm_right = _extract_moments_from_contact_log(cf_right_log)
        t_mw, cm_left  = _apply_time_window(t_forces, cm_left,  start_time=plot_from_time)
        _ ,   cm_right = _apply_time_window(t_forces, cm_right, start_time=plot_from_time)

        _plot_foot_contact_wrench(t_mw, cm_left,  "Moment", "Left",  outdir, plot_from_zero=True)
        _plot_foot_contact_wrench(t_mw, cm_right, "Moment", "Right", outdir, plot_from_zero=True)
        _plot_total_grm_xyz(t_mw, cm_left, cm_right, outdir)
        _plot_grm_xyz_one_foot(t_mw, cm_left,  "Left",  outdir)
        _plot_grm_xyz_one_foot(t_mw, cm_right, "Right", outdir)

    # ---------- Pose pies ----------
    def _plot_pose_series(t, pos_log, ori_log, foot_name):
        P = _ensure_2d(pos_log)
        Q = _ensure_2d(ori_log)
        # ventana para P y Q (si existen)
        if P is not None:
            tP, P = _apply_time_window(t, P, start_time=plot_from_time)
            _plot_vector_series(
                tP, P,
                title=f"{foot_name} Foot Position",
                xlabel="Time (s)", ylabel="Position [m]",
                labels=[r"$x$", r"$y$", r"$z$"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_pos.pdf")
            )
        if Q is not None and Q.shape[1] == 4:
            tQ, Q = _apply_time_window(t, Q, start_time=plot_from_time)
            _plot_vector_series(
                tQ, Q,
                title=f"{foot_name} Foot Quaternion",
                xlabel="Time [s]", ylabel="Component",
                labels=[r"$q_x$", r"$q_y$", r"$q_z$", r"$q_w$"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_quat.pdf")
            )
            YPR = _quat_series_to_euler_zyx(Q, degrees=True)
            if YPR is not None:
                _plot_vector_series(
                    tQ, YPR,
                    title=f"{foot_name} Foot Orientation (Euler ZYX)",
                    xlabel="Time [s]", ylabel="Angle [deg]",
                    labels=[r"Yaw (Z)", r"Pitch (Y)", r"Roll (X)"],
                    filename=os.path.join(outdir, f"{foot_name.lower()}_foot_euler_zyx.pdf")
                )

    if pos_left_log is not None or ori_left_log is not None:
        T_L = len(pos_left_log) if pos_left_log is not None else len(ori_left_log)
        tL = _resolve_time(time, T_L, dt)
        _plot_pose_series(tL, pos_left_log, ori_left_log, "Left")

    if pos_right_log is not None or ori_right_log is not None:
        T_R = len(pos_right_log) if pos_right_log is not None else len(ori_right_log)
        tR = _resolve_time(time, T_R, dt)
        _plot_pose_series(tR, pos_right_log, ori_right_log, "Right")

    # ---------- Wrenches externos / Archimedes / Drag ----------
    if f_ext_log is not None:
        Fext = _ensure_2d(f_ext_log)
        if Fext is not None and Fext.shape[1] >= 6:
            T_F = Fext.shape[0]
            tF = _resolve_time(time, T_F, dt)
            tF, Fext = _apply_time_window(tF, Fext, start_time=plot_from_time)
            _plot_vector_series(tF, Fext[:, :3], title="External Wrench – Forces",
                                xlabel="Time [s]", ylabel="Force [N]",
                                labels=[r"$F_x$", r"$F_y$", r"$F_z$"],
                                filename=os.path.join(outdir, "f_ext_forces.pdf"))
            _plot_vector_series(tF, Fext[:, 3:6], title="External Wrench – Moments",
                                xlabel="Time [s]", ylabel="Moment [N·m]",
                                labels=[r"$M_x$", r"$M_y$", r"$M_z$"],
                                filename=os.path.join(outdir, "f_ext_moments.pdf"))

    if f_A_r_log is not None:
        FAR = _ensure_2d(f_A_r_log)
        if FAR is not None and FAR.shape[1] >= 6:
            T_A = FAR.shape[0]
            tA = _resolve_time(time, T_A, dt)
            tA, FAR = _apply_time_window(tA, FAR, start_time=plot_from_time)
            _plot_vector_series(tA, FAR[:, :3], title="Archimedes Forces",
                                xlabel="Time [s]", ylabel="Force [N]",
                                labels=[r"$F_x$", r"$F_y$", r"$F_z$"],
                                filename=os.path.join(outdir, "archimedes_forces.pdf"))
            _plot_vector_series(tA, FAR[:, 3:6], title="Archimedes Moments",
                                xlabel="Time [s]", ylabel="Moment [N·m]",
                                labels=[r"$M_x$", r"$M_y$", r"$M_z$"],
                                filename=os.path.join(outdir, "archimedes_moments.pdf"))

    if f_D_r_log is not None:
        FDR = _ensure_2d(f_D_r_log)
        if FDR is not None and FDR.shape[1] >= 6:
            T_D = FDR.shape[0]
            tD = _resolve_time(time, T_D, dt)
            tD, FDR = _apply_time_window(tD, FDR, start_time=plot_from_time)
            _plot_vector_series(tD, FDR[:, :3], title="Drag Forces",
                                xlabel="Time [s]", ylabel="Force [N]",
                                labels=[r"$F_x$", r"$F_y$", r"$F_z$"],
                                filename=os.path.join(outdir, "drag_forces.pdf"))
            _plot_vector_series(tD, FDR[:, 3:6], title="Drag Moments",
                                xlabel="Time [s]", ylabel="Moment [N·m]",
                                labels=[r"$M_x$", r"$M_y$", r"$M_z$"],
                                filename=os.path.join(outdir, "drag_moments.pdf"))

    # ---------- Comparativas Real vs Estimado + Errores ----------
    if f_real_log is not None and f_ext_log is not None:
        Fext = _ensure_2d(f_ext_log)
        Freal = _ensure_2d(f_real_log)
        if (Fext is not None and Freal is not None and
            Fext.shape[0] == Freal.shape[0] and
            Fext.shape[1] >= 6 and Freal.shape[1] >= 6):

            T_FR = Fext.shape[0]
            tFR_full = _resolve_time(time, T_FR, dt)

            # Recorta AMBAS usando el MISMO eje original y el MISMO start_time
            tFR, Fext  = _apply_time_window(tFR_full, Fext,  start_time=plot_from_time)
            _,   Freal = _apply_time_window(tFR_full, Freal, start_time=plot_from_time)

            _plot_real_vs_est(
                tFR, real=Freal[:, :3], est=Fext[:, :3], kind="F",
                filename="forces_real_vs_estimated.pdf", title="Forces: Real vs Estimated",
                ylabel="Force [N]", outdir=outdir,
                labels_real=(r"$F_x^{real}$", r"$F_y^{real}$", r"$F_z^{real}$"),
                labels_est =(r"$F_x^{est}$",  r"$F_y^{est}$",  r"$F_z^{est}$")
            )
            _plot_real_vs_est(
                tFR, real=Freal[:, 3:6], est=Fext[:, 3:6], kind="M",
                filename="moments_real_vs_estimated.pdf", title="Moments: Real vs Estimated",
                ylabel="Moment [N·m]", outdir=outdir,
                labels_real=(r"$M_x^{real}$", r"$M_y^{real}$", r"$M_z^{real}$"),
                labels_est =(r"$M_x^{est}$",  r"$M_y^{est}$",  r"$M_z^{est}$")
            )

            Err_abs = Freal[:, :6] - Fext[:, :6]
            Err_rel = np.divide(
                Err_abs,
                np.where(np.abs(Freal[:, :6]) > 1e-6, np.abs(Freal[:, :6]), 1.0),
                out=np.zeros_like(Err_abs),
                where=True
            )

            _plot_vector_series(
                tFR, np.abs(Err_abs),
                title="Wrench Absolute Errors",
                xlabel="Time [s]", ylabel="|Error| [SI]",
                labels=LABELS_ERROR_ABS,
                filename=os.path.join(outdir, "error_abs_log.pdf"),
                colors=ERROR_COLORS, linestyles=ERROR_LINESTYLES_ABS, yscale='log'
            )
            _plot_vector_series(
                tFR, np.abs(Err_rel),
                title="Wrench Relative Errors",
                xlabel="Time [s]", ylabel="|Relative Error| [-]",
                labels=LABELS_ERROR_REL,
                filename=os.path.join(outdir, "error_rel_log.pdf"),
                colors=ERROR_COLORS, linestyles=ERROR_LINESTYLES_REL, yscale='log'
            )