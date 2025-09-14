import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as patches
from matplotlib.patches import FancyArrow

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
    r"$\Delta F_x$", r"$\Delta F_y$", r"$\Delta F_z$",
    r"$\Delta M_x$", r"$\Delta M_y$", r"$\Delta M_z$"
]
LABELS_ERROR_REL = [
    r"$\Delta F_x$ rel", r"$\Delta F_y$ rel", r"$\Delta F_z$ rel",
    r"$\Delta M_x$ rel", r"$\Delta M_y$ rel", r"$\Delta M_z$ rel"
]

# -----------------------------
# Utilities
# -----------------------------
def _as_np(x):
    """Convert to np.ndarray robustly."""
    if x is None:
        return None
    try:
        return np.asarray(x)
    except Exception:
        return np.array(list(x))

def _ensure_2d(series):
    """
    Ensure shape (T, D) from series:
      - list of scalars -> (T, 1)
      - list of vectors -> (T, D)
      - np.array 1D -> (T, 1)
      - np.array 2D -> (T, D)
    """
    if series is None:
        return None
    arr = _as_np(series)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr

def _resolve_time(time, length, dt=None):
    """
    Return time vector:
      - If 'time' provided, validate and return.
      - Else, use 'dt' (if provided).
      - Else, use indices [0,1,2,...].
    """
    if time is not None:
        t = _as_np(time).astype(float)
        if len(t) != length:
            raise ValueError(
                f"Time vector length {len(t)} != data length {length}"
            )
        return t
    if dt is None:
        return np.arange(length, dtype=float)
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
    """
    Convert quaternion (x,y,z,w) to Euler ZYX -> (yaw,pitch,roll).
    """
    q = _quat_normalize(q_xyzw)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    if degrees:
        k = 180.0 / np.pi
        return yaw * k, pitch * k, roll * k
    return yaw, pitch, roll

def _quat_series_to_euler_zyx(quat_series, degrees=True):
    """
    quat_series: (T,4) [x,y,z,w]
    Returns (T,3): [yaw,pitch,roll]
    """
    Q = _ensure_2d(quat_series)
    if Q is None:
        return None
    if Q.shape[1] != 4:
        raise ValueError(f"Expected (T,4) quaternions, got {Q.shape}")
    YPR = np.zeros((Q.shape[0], 3))
    for i in range(Q.shape[0]):
        YPR[i] = np.array(_quat_to_euler_zyx(Q[i], degrees=degrees))
    return YPR

# -----------------------------
# Base Plotting Functions
# -----------------------------
def _plot_vector_series(t, Y, title, xlabel, ylabel, labels, filename, yscale=None):
    """
    Plot multi-dimensional series Y (T, D).
    """
    Y = _ensure_2d(Y)
    if Y is None or Y.size == 0:
        return
    T, D = Y.shape
    if len(t) != T:
        if Y.T.shape[0] == len(t):
            Y = Y.T
            T, D = Y.shape
        else:
            raise ValueError(f"len(t)={len(t)} vs Y.shape={Y.shape}")

    if labels is None:
        labels = [f"Dim {i}" for i in range(D)]
    elif isinstance(labels, str):
        labels = [labels]
    if len(labels) < D:
        labels = labels + [f"Dim {i}" for i in range(len(labels), D)]
    elif len(labels) > D:
        labels = labels[:D]

    _safe_makedirs_for(filename)
    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    color_cycle = itertools.cycle([
        STYLE_GUIDE['colors']['blue'],
        STYLE_GUIDE['colors']['green'],
        STYLE_GUIDE['colors']['orange'],
        STYLE_GUIDE['colors']['red'],
        STYLE_GUIDE['colors']['purple']
    ])

    for i in range(D):
        ax.plot(
            t, Y[:, i], label=labels[i],
            color=next(color_cycle),
            linewidth=STYLE_GUIDE['lines']['width']
        )

    ax.set_xlabel(xlabel, fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel(ylabel, fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(title, fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])

    if D > 1 or (D == 1 and labels[0] is not None):
        ax.legend(loc='best', fontsize=STYLE_GUIDE['fonts']['legend'])
    if yscale:
        ax.set_yscale(yscale)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

# -----------------------------
# Contact logs: forces and moments
# -----------------------------
def _extract_wrench_component(contact_log, start_idx):
    """
    Extract wrench components (force or moment) from contact_log.
    """
    N = len(contact_log)
    component = np.zeros((4, N, 3))
    for i, frame in enumerate(contact_log):
        for j in range(min(4, len(frame))):
            item = frame[j]
            try:
                vec = np.asarray(item, dtype=float)
            except Exception:
                continue
            if vec.ndim == 1 and vec.size >= start_idx + 3:
                component[j, i, :] = vec[start_idx:start_idx+3]
    return component

def _extract_forces_from_contact_log(contact_log):
    """
    Each frame has up to 4 items.
      - [f_world(3,), pos(3,)]
      - [wrench(6,), ...] -> take [:3]
      - [force(3,), ...]
    Returns (4, N, 3).
    """
    return _extract_wrench_component(contact_log, 0)

def _extract_moments_from_contact_log(contact_log):
    """
    Each frame has up to 4 items.
      - [wrench(6,), ...] -> take [3:6]
      - If only forces, return zeros.
    Returns (4, N, 3).
    """
    return _extract_wrench_component(contact_log, 3)

def _apply_time_window(t, arr_like, start_time=0.0):
    """
    Apply temporal window t >= start_time to t and array.
    Handles:
      - (N, D)
      - (4, N, 3) (contact logs)
    """
    t = np.asarray(t).reshape(-1)
    mask = t >= float(start_time)
    if arr_like is None:
        return t[mask], None
    arr = np.asarray(arr_like)
    if arr.ndim == 2 and arr.shape[0] == t.size:
        return t[mask], arr[mask]
    if arr.ndim == 3 and arr.shape[1] == t.size:
        return t[mask], arr[:, mask, :]
    return t, arr_like

# -----------------------------
# Plot foot contact wrenches
# -----------------------------
def _plot_foot_contact_wrench(time_arr, wrench_data, component_type,
                              foot_name, outdir, plot_from_zero=True):
    """
    Plot contact forces or moments with consistent style.
    """
    start_time = 0.0 if plot_from_zero else 2.0
    t, W = _apply_time_window(time_arr, wrench_data, start_time=start_time)

    is_force = component_type.lower() == 'force'
    unit = "[N]" if is_force else "[N·m]"
    labels = LABELS_FORCE if is_force else LABELS_MOMENT

    fig = plt.figure(figsize=STYLE_GUIDE['figure']['size_tall'])
    gs = fig.add_gridspec(4, 2, width_ratios=[4, 1], hspace=0.35)
    fig.suptitle(
        f"Contact {component_type}s – {foot_name} Foot",
        fontsize=STYLE_GUIDE['fonts']['suptitle'],
        color=STYLE_GUIDE['colors']['text']
    )

    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    color_cycle_base = [
        STYLE_GUIDE['colors']['blue'],
        STYLE_GUIDE['colors']['green'],
        STYLE_GUIDE['colors']['orange']
    ]

    for i in range(4):
        for c in range(3):
            axs[i].plot(
                t, W[i, :, c], label=labels[c],
                lw=STYLE_GUIDE['lines']['width'],
                color=color_cycle_base[c]
            )
        axs[i].set_ylabel(
            f"{component_type} [{i + 1}] {unit}",
            fontsize=STYLE_GUIDE['fonts']['label']
        )
        if i == 0:
            axs[i].legend(
                loc='upper right',
                fontsize=STYLE_GUIDE['fonts']['legend']
            )
        axs[i].grid(
            True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid']
        )
        axs[i].tick_params(
            axis='both', which='major',
            labelsize=STYLE_GUIDE['fonts']['ticks']
        )
        axs[i].set_xlim(t[0], t[-1])
    axs[-1].set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])

    # Foot schematic
    ax_schematic = fig.add_subplot(gs[:, 1])
    hx, hy = 0.1, 0.04
    rect = patches.Rectangle(
        (-hy, -hx), 2*hy, 2*hx,
        linewidth=1,
        edgecolor=STYLE_GUIDE['colors']['text'],
        facecolor=STYLE_GUIDE['colors']['line_light']
    )
    ax_schematic.add_patch(rect)

    # Right foot default
    poc_positions = [(-hy, -hx), (-hy, hx), (hy, hx), (hy, -hx)]
    if foot_name == "Left":
        poc_positions = [(hy, -hx), (hy, hx), (-hy, hx), (-hy, -hx)]

    for i, (x, y) in enumerate(poc_positions):
        ax_schematic.plot(x, y, 'o', color=STYLE_GUIDE['colors']['blue'])
        ax_schematic.text(
            x + 0.020, y + (0.015 if y >= 0 else -0.015),
            f'{i+1}', fontsize=12,
            color=STYLE_GUIDE['colors']['text'],
            va='center'
        )

    arrow = FancyArrow(
        0, 0, 0, 0.15,
        width=0.005, head_width=0.015, head_length=0.015,
        color=STYLE_GUIDE['colors']['blue']
    )
    ax_schematic.add_patch(arrow)
    ax_schematic.text(
        0.02, 0.18, '+X (Front)',
        color=STYLE_GUIDE['colors']['text'],
        fontsize=10, ha='center'
    )
    ax_schematic.set_xlim(-0.08, 0.08)
    ax_schematic.set_ylim(-0.15, 0.22)
    ax_schematic.set_aspect('equal')
    ax_schematic.set_title(
        "Points of Contact\n(Top View)",
        fontsize=STYLE_GUIDE['fonts']['label']
    )
    ax_schematic.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(outdir, f'contact_{component_type.lower()}_{foot_name.lower()}.pdf')
    )
    plt.close(fig)

# -----------------------------
# Total GRF/GRM and per-foot plots
# -----------------------------
def _plot_total_grf_xyz(time_arr, forces_left, forces_right, outdir, total_mass_hint=35.1151):
    """
    Sum forces from 4 contact points per foot -> (N,3),
    then sum feet -> (N,3).
    """
    t = np.asarray(time_arr).reshape(-1)
    F_L = forces_left.sum(axis=0)   # (N,3)
    F_R = forces_right.sum(axis=0)  # (N,3)
    F_T = F_L + F_R                 # (N,3)

    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    ax.plot(t, F_T[:, 0], label=r"$F_x$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, F_T[:, 1], label=r"$F_y$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, F_T[:, 2], label=r"$F_z$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['orange'])

    weight = float(total_mass_hint) * 9.81
    ax.plot(t, np.full_like(t, weight), STYLE_GUIDE['lines']['grid_style'],
            label="Weight", linewidth=1.2, color=STYLE_GUIDE['colors']['red'])

    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Force [N]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title("Total Ground Reaction Force",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend'])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "grf_total_xyz.pdf"))
    plt.close(fig)

def _plot_total_grm_xyz(time_arr, moments_left, moments_right, outdir):
    """
    Sum moments from 4 contact points per foot -> (N,3),
    then sum feet -> (N,3).
    """
    t = np.asarray(time_arr).reshape(-1)
    M_L = moments_left.sum(axis=0)   # (N,3)
    M_R = moments_right.sum(axis=0)  # (N,3)
    M_T = M_L + M_R                  # (N,3)

    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    ax.plot(t, M_T[:, 0], label=r"$M_x$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, M_T[:, 1], label=r"$M_y$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, M_T[:, 2], label=r"$M_z$", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['orange'])

    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Moment [N·m]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title("Total Ground Reaction Moment",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend'])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "grm_total_xyz.pdf"))
    plt.close(fig)

def _plot_grf_xyz_one_foot(time_arr, forces_foot, foot_name, outdir, total_mass_hint=35.1151):
    """
    Sum forces of 4 contact points of one foot -> (N,3) and plot.
    """
    t = np.asarray(time_arr).reshape(-1)
    F_FOOT = forces_foot.sum(axis=0)  # (N,3)

    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    ax.plot(t, F_FOOT[:, 0], label=rf"$F_x$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, F_FOOT[:, 1], label=rf"$F_y$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, F_FOOT[:, 2], label=rf"$F_z$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['orange'])

    weight = float(total_mass_hint) * 9.81
    ax.plot(t, np.full_like(t, weight/2.0), STYLE_GUIDE['lines']['grid_style'],
            label="Weight/2", linewidth=1.2, color=STYLE_GUIDE['colors']['red'])

    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Force [N]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Ground Reaction Force – {foot_name} Foot",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend'])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"grf_{foot_name.lower()}_xyz.pdf"))
    plt.close(fig)

def _plot_grm_xyz_one_foot(time_arr, moments_foot, foot_name, outdir):
    """
    Sum moments of 4 contact points of one foot -> (N,3) and plot.
    """
    t = np.asarray(time_arr).reshape(-1)
    M_FOOT = moments_foot.sum(axis=0)  # (N,3)

    fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size_wide'])

    ax.plot(t, M_FOOT[:, 0], label=rf"$M_x$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['blue'])
    ax.plot(t, M_FOOT[:, 1], label=rf"$M_y$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['green'])
    ax.plot(t, M_FOOT[:, 2], label=rf"$M_z$ {foot_name}", lw=STYLE_GUIDE['lines']['width'],
            color=STYLE_GUIDE['colors']['orange'])

    ax.set_xlabel("Time (s)", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_ylabel("Moment [N·m]", fontsize=STYLE_GUIDE['fonts']['label'])
    ax.set_title(f"Ground Reaction Moment – {foot_name} Foot",
                 fontsize=STYLE_GUIDE['fonts']['title'],
                 color=STYLE_GUIDE['colors']['text'])
    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'],
            color=STYLE_GUIDE['colors']['grid'])
    ax.tick_params(axis='both', which='major',
                   labelsize=STYLE_GUIDE['fonts']['ticks'])
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=STYLE_GUIDE['fonts']['legend'])
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, f"grm_{foot_name.lower()}_xyz.pdf"))
    plt.close(fig)

# -----------------------------
# Public API
# -----------------------------
def make_plots(
    outdir="plots/general",
    *,
    time=None,     # (T,)
    dt=None,
    tau_ext=None,  # (T, nv) or (T,)
    cf_left_log=None,
    cf_right_log=None,
    pos_left_log=None,   # (T,3)
    pos_right_log=None,  # (T,3)
    ori_left_log=None,   # (T,4) quaternion [x,y,z,w]
    ori_right_log=None,  # (T,4)
    f_ext_log=None,
    f_real_log=None,
    f_A_r_log=None,
    f_D_r_log=None,
    plot_from_zero=True,
    **_ignore,
):
    """
    Generate plots for given logs.
    Args:
        plot_from_zero (bool): If True, start at t=0. If False, start at t>=2s.
    """
    os.makedirs(outdir, exist_ok=True)
    print("--- Generating all plots...")

    # === tau_ext: norm + components ===
    if tau_ext is not None and len(tau_ext) > 0:
        tau_ext_arr = _ensure_2d(tau_ext)  # (T, D)
        T = tau_ext_arr.shape[0]
        t = _resolve_time(time, T, dt)

        # Norm
        try:
            tau_norm = np.linalg.norm(tau_ext_arr, axis=1)
            _plot_vector_series(
                t, tau_norm.reshape(-1, 1),
                title=f"External Torque Norm ({TAU_EXT})",
                xlabel="Time [s]",
                ylabel=r"$\|\tau_{\mathrm{ext}}\|$ [SI]",
                labels=[r"$\|\tau_{ext}\|$"],
                filename=os.path.join(outdir, "tau_ext_norm.pdf"),
            )
        except Exception as e:
            print(f"[plots] Error plotting τ_ext norm: {e}")

        # Full components
        _plot_vector_series(
            t, tau_ext_arr,
            title=f"External Torque Components ({TAU_EXT})",
            xlabel="Time [s]",
            ylabel="Torque [N·m]",
            labels=None,
            filename=os.path.join(outdir, "tau_ext_components.pdf"),
        )

        # First 3 entries as forces
        if tau_ext_arr.shape[1] >= 3:
            _plot_vector_series(
                t, tau_ext_arr[:, :3],
                title=f"{TAU_EXT} – Forces",
                xlabel="Time [s]",
                ylabel="Force [N]",
                labels=LABELS_FORCE,
                filename=os.path.join(outdir, "tau_ext_forces_first3.pdf"),
            )
        # Next 3 as moments
        if tau_ext_arr.shape[1] >= 6:
            _plot_vector_series(
                t, tau_ext_arr[:, 3:6],
                title=f"{TAU_EXT} – Moments",
                xlabel="Time [s]",
                ylabel="Moment [N·m]",
                labels=LABELS_MOMENT,
                filename=os.path.join(outdir, "tau_ext_moments_next3.pdf"),
            )

    # === Contact forces & moments per foot + totals ===
    if cf_left_log is not None and cf_right_log is not None:
        T_forces = len(cf_left_log)
        t_forces = _resolve_time(time, T_forces, dt)

        # Forces
        cf_left = _extract_forces_from_contact_log(cf_left_log)
        cf_right = _extract_forces_from_contact_log(cf_right_log)

        _plot_foot_contact_wrench(t_forces, cf_left, "Force", "Left", outdir, plot_from_zero)
        _plot_foot_contact_wrench(t_forces, cf_right, "Force", "Right", outdir, plot_from_zero)
        _plot_total_grf_xyz(t_forces, cf_left, cf_right, outdir)
        _plot_grf_xyz_one_foot(t_forces, cf_left, "Left", outdir)
        _plot_grf_xyz_one_foot(t_forces, cf_right, "Right", outdir)

        # Moments
        cm_left = _extract_moments_from_contact_log(cf_left_log)
        cm_right = _extract_moments_from_contact_log(cf_right_log)

        _plot_foot_contact_wrench(t_forces, cm_left, "Moment", "Left", outdir, plot_from_zero)
        _plot_foot_contact_wrench(t_forces, cm_right, "Moment", "Right", outdir, plot_from_zero)
        _plot_total_grm_xyz(t_forces, cm_left, cm_right, outdir)
        _plot_grm_xyz_one_foot(t_forces, cm_left, "Left", outdir)
        _plot_grm_xyz_one_foot(t_forces, cm_right, "Right", outdir)

    # === Foot pose plots ===
    def _plot_pose_series(t, pos_log, ori_log, foot_name):
        # Position
        P = _ensure_2d(pos_log)  # (T,3)
        if P is not None and P.shape[1] == 3:
            _plot_vector_series(
                t, P,
                title=f"{foot_name} Foot Position",
                xlabel="Time (s)", ylabel="Position [m]",
                labels=[r"$x$", r"$y$", r"$z$"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_pos.pdf"),
            )

        # Quaternion
        Q = _ensure_2d(ori_log)  # (T,4)
        if Q is not None and Q.shape[1] == 4:
            _plot_vector_series(
                t, Q,
                title=f"{foot_name} Foot Quaternion",
                xlabel="Time [s]", ylabel="Component",
                labels=[r"$q_x$", r"$q_y$", r"$q_z$", r"$q_w$"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_quat.pdf"),
            )
            # Euler ZYX
            YPR = _quat_series_to_euler_zyx(Q, degrees=True)  # (T,3)
            if YPR is not None:
                _plot_vector_series(
                    t, YPR,
                    title=f"{foot_name} Foot Orientation (Euler ZYX)",
                    xlabel="Time [s]", ylabel="Angle [deg]",
                    labels=[r"Yaw (Z)", r"Pitch (Y)", r"Roll (X)"],
                    filename=os.path.join(outdir, f"{foot_name.lower()}_foot_euler_zyx.pdf"),
                )

    if pos_left_log is not None or ori_left_log is not None:
        T_L = len(pos_left_log) if pos_left_log is not None else len(ori_left_log)
        tL = _resolve_time(time, T_L, dt)
        _plot_pose_series(tL, pos_left_log, ori_left_log, "Left")

    if pos_right_log is not None or ori_right_log is not None:
        T_R = len(pos_right_log) if pos_right_log is not None else len(ori_right_log)
        tR = _resolve_time(time, T_R, dt)
        _plot_pose_series(tR, pos_right_log, ori_right_log, "Right")

    # === External Wrench Logs ===
    if f_ext_log is not None:
        Fext = _ensure_2d(f_ext_log)  # (T,6)
        if Fext is not None and Fext.shape[1] >= 6:
            T_F = Fext.shape[0]
            tF = _resolve_time(time, T_F, dt)

            _plot_vector_series(
                tF, Fext[:, :3],
                title="External Wrench – Forces",
                xlabel="Time [s]", ylabel="Force [N]",
                labels=LABELS_FORCE,
                filename=os.path.join(outdir, "f_ext_forces.pdf"),
            )
            _plot_vector_series(
                tF, Fext[:, 3:6],
                title="External Wrench – Moments",
                xlabel="Time [s]", ylabel="Moment [N·m]",
                labels=LABELS_MOMENT,
                filename=os.path.join(outdir, "f_ext_moments.pdf"),
            )

    if f_A_r_log is not None:
        FAR = _ensure_2d(f_A_r_log)
        if FAR is not None and FAR.shape[1] >= 6:
            T_A = FAR.shape[0]
            tA = _resolve_time(time, T_A, dt)
            _plot_vector_series(
                tA, FAR[:, :3],
                title="Archimedes Forces",
                xlabel="Time [s]", ylabel="Force [N]",
                labels=LABELS_FORCE,
                filename=os.path.join(outdir, "archimedes_forces.pdf"),
            )
            _plot_vector_series(
                tA, FAR[:, 3:6],
                title="Archimedes Moments",
                xlabel="Time [s]", ylabel="Moment [N·m]",
                labels=LABELS_MOMENT,
                filename=os.path.join(outdir, "archimedes_moments.pdf"),
            )

    if f_D_r_log is not None:
        FDR = _ensure_2d(f_D_r_log)
        if FDR is not None and FDR.shape[1] >= 6:
            T_D = FDR.shape[0]
            tD = _resolve_time(time, T_D, dt)
            _plot_vector_series(
                tD, FDR[:, :3],
                title="Drag Forces",
                xlabel="Time [s]", ylabel="Force [N]",
                labels=LABELS_FORCE,
                filename=os.path.join(outdir, "drag_forces.pdf"),
            )
            _plot_vector_series(
                tD, FDR[:, 3:6],
                title="Drag Moments",
                xlabel="Time [s]", ylabel="Moment [N·m]",
                labels=LABELS_MOMENT,
                filename=os.path.join(outdir, "drag_moments.pdf"),
            )

    # === Compare estimated vs real external wrench ===
    if f_real_log is not None and f_ext_log is not None:
        Fext = _ensure_2d(f_ext_log)
        Freal = _ensure_2d(f_real_log)
        if Fext is not None and Freal is not None and \
           Fext.shape[0] == Freal.shape[0] and Fext.shape[1] >= 6 and Freal.shape[1] >= 6:

            T_FR = Fext.shape[0]
            tFR = _resolve_time(time, T_FR, dt)

            # Forces comparison
            _plot_vector_series(
                tFR, np.column_stack((Freal[:, :3], Fext[:, :3])),
                title="Forces: Real vs Estimated",
                xlabel="Time [s]", ylabel="Force [N]",
                labels=[r"$F_x^{real}$", r"$F_y^{real}$", r"$F_z^{real}$",
                        r"$F_x^{ext}$", r"$F_y^{ext}$", r"$F_z^{ext}$"],
                filename=os.path.join(outdir, "forces_real_vs_estimated.pdf"),
            )

            # Moments comparison
            _plot_vector_series(
                tFR, np.column_stack((Freal[:, 3:6], Fext[:, 3:6])),
                title="Moments: Real vs Estimated",
                xlabel="Time [s]", ylabel="Moment [N·m]",
                labels=[r"$M_x^{real}$", r"$M_y^{real}$", r"$M_z^{real}$",
                        r"$M_x^{ext}$", r"$M_y^{ext}$", r"$M_z^{ext}$"],
                filename=os.path.join(outdir, "moments_real_vs_estimated.pdf"),
            )

            # Errors
            Err_abs = Freal[:, :6] - Fext[:, :6]
            Err_rel = np.divide(
                Err_abs,
                np.where(np.abs(Freal[:, :6]) > 1e-6, np.abs(Freal[:, :6]), 1.0),
                out=np.zeros_like(Err_abs), where=True
            )

            _plot_vector_series(
                tFR, Err_abs,
                title="Absolute Errors (Real - Estimated)",
                xlabel="Time [s]", ylabel="Error [SI]",
                labels=LABELS_ERROR_ABS,
                filename=os.path.join(outdir, "error_abs.pdf"),
            )
            _plot_vector_series(
                tFR, Err_rel,
                title="Relative Errors (Real - Estimated)",
                xlabel="Time [s]", ylabel="Relative Error",
                labels=LABELS_ERROR_REL,
                filename=os.path.join(outdir, "error_rel.pdf"),
            )
