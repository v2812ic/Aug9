import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilidades internas
# -----------------------------
def _as_np(x):
    """Convierte a np.ndarray de forma tolerante."""
    if x is None:
        return None
    try:
        return np.asarray(x)
    except Exception:
        return np.array(list(x))


def _ensure_2d(series):
    """
    Asegura shape (T, D) a partir de una serie que puede ser:
      - lista de escalares -> (T, 1)
      - lista de vectores  -> (T, D)
      - np.array 1D        -> (T, 1)
      - np.array 2D        -> (T, D)
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
    Devuelve el vector de tiempo:
      - Si 'time' viene, lo valida y devuelve.
      - Si no, usa 'dt' (si viene) para generar [0, dt, 2dt, ...].
      - Si no viene ni time ni dt, usa índices [0, 1, 2, ...].
    """
    if time is not None:
        t = _as_np(time).astype(float)
        if len(t) != length:
            raise ValueError(f"El vector 'time' tiene len={len(t)} y la serie len={length}.")
        return t
    if dt is None:
        return np.arange(length, dtype=float)
    return np.arange(length, dtype=float) * float(dt)


def _safe_makedirs_for(filename):
    d = os.path.dirname(filename) or "."
    os.makedirs(d, exist_ok=True)


# -----------------------------
# Quaterniones / Orientación
# -----------------------------
def _quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0.0] = 1.0
    return q / n

def _quat_to_euler_zyx(q_xyzw, degrees=True):
    """
    Convierte un cuaternión (x, y, z, w) a Euler ZYX -> (yaw, pitch, roll).
    Devuelve radianes por defecto, grados si degrees=True.
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
    quat_series: (T,4) con [x,y,z,w]
    Devuelve (T,3): [yaw, pitch, roll]
    """
    Q = _ensure_2d(quat_series)
    if Q is None: return None
    if Q.shape[1] != 4:
        raise ValueError(f"Se esperaba quat en forma (T,4) [x,y,z,w], recibido {Q.shape}")
    YPR = np.zeros((Q.shape[0], 3))
    for i in range(Q.shape[0]):
        YPR[i] = np.array(_quat_to_euler_zyx(Q[i], degrees=degrees))
    return YPR


# -----------------------------
# Plotting básicos
# -----------------------------
def _plot_vector_series(t, Y, title, xlabel, ylabel, labels, filename, yscale = None):
    """
    Dibuja múltiples series columnares Y (T, D) en la misma figura.
    - t: (T,)
    - Y: (T, D)
    """
    Y = _ensure_2d(Y)
    if Y is None or Y.size == 0:
        return
    T, D = Y.shape
    if len(t) != T:
        # intentar corregir si viene traspuesta
        if Y.T.shape[0] == len(t):
            Y = Y.T
            T, D = Y.shape
        else:
            raise ValueError(f"Dimensiones no coherentes: len(t)={len(t)} vs Y.shape={Y.shape}")

    if labels is None:
        labels = [f"{ylabel}[{i}]" for i in range(D)]
    elif isinstance(labels, str):
        labels = [labels]
    if len(labels) < D:
        labels = labels + [f"{ylabel}[{i}]" for i in range(len(labels), D)]
    elif len(labels) > D:
        labels = labels[:D]

    _safe_makedirs_for(filename)
    plt.figure()

    if yscale:
        plt.yscale(yscale)

    for i in range(D):
        plt.plot(t, Y[:, i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if D > 1:
        plt.legend(loc='upper left', fontsize='small')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    plt.close()


# -----------------------------
# Contact logs: fuerzas y momentos
# -----------------------------
def _extract_forces_from_contact_log(contact_log):
    """
    contact_log por frame: hasta 4 items.
      A) [ [f_world(3,), pos(3,)], ... ]
      B) [ wrench(6,), ... ] -> fuerza = wrench[:3]
      C) [ force(3,), ... ]
    Devuelve: forces (4, N, 3)
    """
    N = len(contact_log)
    forces = np.zeros((4, N, 3))
    for i, frame in enumerate(contact_log):
        for j in range(min(4, len(frame))):
            item = frame[j]
            # A) [f, pos]
            if isinstance(item, (list, tuple)) and len(item) == 2:
                f = np.asarray(item[0], dtype=float)
                if f.size >= 3:
                    forces[j, i, :] = f[:3]
                continue
            # B/C) vector
            vec = np.asarray(item, dtype=float)
            if vec.ndim == 1 and vec.size >= 3:
                forces[j, i, :] = vec[:3]  # si wrench, esto es fuerza
    return forces

def _extract_moments_from_contact_log(contact_log):
    """
    Igual que el de fuerzas, pero intenta recuperar momentos:
      - Si item es wrench(6,), toma item[3:6] como [Mx,My,Mz]
      - Si item es [f,pos] o solo fuerza(3,), deja 0 (no hay momento)
    Devuelve: moments (4, N, 3)
    """
    N = len(contact_log)
    moms = np.zeros((4, N, 3))
    for i, frame in enumerate(contact_log):
        for j in range(min(4, len(frame))):
            item = frame[j]
            vec = None
            if isinstance(item, (list, tuple)) and len(item) == 2:
                # [f_world, pos] -> sin momento -> 0
                continue
            try:
                vec = np.asarray(item, dtype=float)
            except Exception:
                continue
            if vec.ndim == 1 and vec.size >= 6:
                moms[j, i, :] = vec[3:6]  # tail del wrench
    return moms

def _plot_foot_contact_forces(time_arr, forces, foot_name, outdir):
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrow

    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(4, 2, width_ratios=[4, 1], hspace=0.35)

    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    comp_labels = ["Fx", "Fy", "Fz"]
    for i in range(4):
        for c in range(3):
            axs[i].plot(time_arr, forces[i, :, c], label=comp_labels[c], lw=1.6)
        axs[i].set_ylabel(f"Force [{i + 1}] (N)")
        axs[i].set_ylim(-50, 100)
        if i == 0:
            axs[i].legend(loc='upper right', fontsize='small')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].minorticks_on()
        axs[i].grid(which='minor', linestyle=':', alpha=0.3, linewidth=0.5)
    axs[-1].set_xlabel("Time (s)")

    # ESQUEMA DEL PIE
    ax_schematic = fig.add_subplot(gs[:, 1])
    hx, hy = 0.1, 0.04  # half-extents
    rect = patches.Rectangle((-hy, -hx), 2*hy, 2*hx,
                             linewidth=1, edgecolor='black',
                             facecolor='lightgray', alpha=0.3)
    ax_schematic.add_patch(rect)

    if foot_name == "Left":
        poc_positions = [(hy, -hx), (hy, hx), (-hy, hx), (-hy, -hx)]
    else:
        poc_positions = [(-hy, -hx), (-hy, hx), (hy, hx), (hy, -hx)]

    for i, (x, y) in enumerate(poc_positions):
        ax_schematic.plot(x, y, 'o', color='blue')
        ax_schematic.text(x + 0.005, y + 0.005, f'{i+1}',
                          fontsize=12, color='black')

    from matplotlib.patches import FancyArrow
    arrow = FancyArrow(0, 0, 0, 0.15, width=0.005,
                       head_width=0.015, head_length=0.015, color='blue')
    ax_schematic.add_patch(arrow)
    ax_schematic.text(0.02, 0.18, '+X (front)', color='k', fontsize=10, ha='center')

    ax_schematic.set_xlim(-0.08, 0.08)
    ax_schematic.set_ylim(-0.15, 0.22)
    ax_schematic.set_aspect('equal')
    ax_schematic.set_title("Points of contact\n(top view)")
    ax_schematic.axis('off')

    fig.suptitle(f"Contact forces - {foot_name} foot", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(outdir, f'foto_contact_forces_{foot_name.lower()}.png'), dpi=300)
    plt.close(fig)

def _plot_foot_contact_moments(time_arr, moments, foot_name, outdir):
    """
    Igual que fuerzas pero para momentos [Mx, My, Mz] en cada punto.
    """
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrow

    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(4, 2, width_ratios=[4, 1], hspace=0.35)

    axs = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    comp_labels = ["Mx", "My", "Mz"]
    for i in range(4):
        for c in range(3):
            axs[i].plot(time_arr, moments[i, :, c], label=comp_labels[c], lw=1.6)
        axs[i].set_ylabel(f"Moment [{i + 1}] (N·m)")
        axs[i].set_ylim(-30, 30)  # ajusta si lo necesitas
        if i == 0:
            axs[i].legend(loc='upper right', fontsize='small')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].minorticks_on()
        axs[i].grid(which='minor', linestyle=':', alpha=0.3, linewidth=0.5)
    axs[-1].set_xlabel("Time (s)")

    # Esquema pie
    ax_schematic = fig.add_subplot(gs[:, 1])
    hx, hy = 0.1, 0.04
    rect = patches.Rectangle((-hy, -hx), 2*hy, 2*hx,
                             linewidth=1, edgecolor='black',
                             facecolor='lightgray', alpha=0.3)
    ax_schematic.add_patch(rect)
    if foot_name == "Left":
        poc_positions = [(hy, -hx), (hy, hx), (-hy, hx), (-hy, -hx)]
    else:
        poc_positions = [(-hy, -hx), (-hy, hx), (hy, hx), (hy, -hx)]
    for i, (x, y) in enumerate(poc_positions):
        ax_schematic.plot(x, y, 'o', color='purple')
        ax_schematic.text(x + 0.005, y + 0.005, f'{i+1}', fontsize=12, color='black')

    arrow = FancyArrow(0, 0, 0, 0.15, width=0.005,
                       head_width=0.015, head_length=0.015, color='blue')
    ax_schematic.add_patch(arrow)
    ax_schematic.text(0.02, 0.18, '+X (front)', color='k', fontsize=10, ha='center')

    ax_schematic.set_xlim(-0.08, 0.08)
    ax_schematic.set_ylim(-0.15, 0.22)
    ax_schematic.set_aspect('equal')
    ax_schematic.set_title("Points of contact\n(top view)")
    ax_schematic.axis('off')

    fig.suptitle(f"Contact moments - {foot_name} foot", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(outdir, f'foto_contact_moments_{foot_name.lower()}.png'), dpi=300)
    plt.close(fig)

def _plot_total_grf_xyz(time_arr, forces_left, forces_right, outdir, total_mass_hint=35.1151):
    """
    Suma fuerzas de 4 puntos por pie -> (N,3) y luego suma pies -> (N,3).
    """
    t = np.asarray(time_arr).reshape(-1)
    F_L = forces_left.sum(axis=0)   # (N,3)
    F_R = forces_right.sum(axis=0)  # (N,3)
    F_T = F_L + F_R                 # (N,3)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, F_T[:, 0], label="Fx_total", lw=1.8)
    ax.plot(t, F_T[:, 1], label="Fy_total", lw=1.8)
    ax.plot(t, F_T[:, 2], label="Fz_total", lw=1.8)

    weight = float(total_mass_hint) * 9.81
    ax.plot(t, np.full_like(t, weight), "--", label="Weight", linewidth=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title("Total Ground Reaction – Fx, Fy, Fz")
    ax.set_ylim(-100, 400)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "grf_total_xyz.png"), dpi=300)
    plt.close(fig)

def _plot_total_grm_xyz(time_arr, moments_left, moments_right, outdir):
    """
    Suma momentos de 4 puntos por pie -> (N,3) y luego suma pies -> (N,3).
    """
    t = np.asarray(time_arr).reshape(-1)
    M_L = moments_left.sum(axis=0)   # (N,3)
    M_R = moments_right.sum(axis=0)  # (N,3)
    M_T = M_L + M_R                  # (N,3)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, M_T[:, 0], label="Mx_total", lw=1.8)
    ax.plot(t, M_T[:, 1], label="My_total", lw=1.8)
    ax.plot(t, M_T[:, 2], label="Mz_total", lw=1.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Moment (N·m)")
    ax.set_title("Total Ground Reaction Moment – Mx, My, Mz")
    ax.set_ylim(-50, 50)  # ajusta si lo necesitas
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", alpha=0.3, linewidth=0.5)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "grm_total_xyz.png"), dpi=300)
    plt.close(fig)


# -----------------------------
# API pública
# -----------------------------
def make_plots(
    outdir="plots",
    *,
    time=None,     # (T,)
    dt=None,
    tau_ext=None,  # (T, nv) o (T,)
    cf_left_log=None,
    cf_right_log=None,
    pos_left_log=None,   # (T,3) o lista de [x,y,z]
    pos_right_log=None,  # (T,3)
    ori_left_log=None,   # (T,4) cuat [x,y,z,w]
    ori_right_log=None,  # (T,4)
    f_ext_log = None,
    f_real_log = None,
    f_A_r_log=None,   # <-- NUEVO
    f_D_r_log=None,
    **_ignore,
):
    """
    Genera figuras SOLO con lo que pases por kwargs.
    """
    os.makedirs(outdir, exist_ok=True)

    # === tau_ext: norma + componentes ===
    if tau_ext is not None and len(tau_ext) > 0:
        tau_ext_arr = _ensure_2d(tau_ext)  # (T, D)
        T = tau_ext_arr.shape[0]
        t = _resolve_time(time, T, dt)

        # Norma
        try:
            tau_norm = np.linalg.norm(tau_ext_arr, axis=1)
            _plot_vector_series(
                t, tau_norm.reshape(-1, 1),
                title="Norma de t_ext",
                xlabel="Tiempo [s]",
                ylabel="||t_ext||",
                labels=["||t_ext||"],
                filename=os.path.join(outdir, "tau_ext_norm.png"),
            )
        except Exception as e:
            print(f"[plots] Error graficando norma de t_ext: {e}")

        # Componentes
        _plot_vector_series(
            t, tau_ext_arr,
            title="Componentes de t_ext",
            xlabel="Tiempo [s]",
            ylabel="t_ext",
            labels=None,
            filename=os.path.join(outdir, "tau_ext_components.png"),
        )

    # === Contact forces & moments por pie + totales ===
    if cf_left_log is not None and cf_right_log is not None:
        T_forces = len(cf_left_log)
        t_forces = _resolve_time(time, T_forces, dt)

        # Fuerzas (4,N,3)
        cf_left  = _extract_forces_from_contact_log(cf_left_log)
        cf_right = _extract_forces_from_contact_log(cf_right_log)

        _plot_foot_contact_forces(t_forces, cf_left,  "Left",  outdir)
        _plot_foot_contact_forces(t_forces, cf_right, "Right", outdir)
        _plot_total_grf_xyz(t_forces, cf_left, cf_right, outdir)

        # Momentos (4,N,3) — si el log no trae wrench 6D, quedará en 0
        cm_left  = _extract_moments_from_contact_log(cf_left_log)
        cm_right = _extract_moments_from_contact_log(cf_right_log)

        _plot_foot_contact_moments(t_forces, cm_left,  "Left",  outdir)
        _plot_foot_contact_moments(t_forces, cm_right, "Right", outdir)
        _plot_total_grm_xyz(t_forces, cm_left, cm_right, outdir)

    # === Posición y orientación de cada pie ===
    def _plot_pose_series(t, pos_log, ori_log, foot_name):
        # Posición
        P = _ensure_2d(pos_log)  # (T,3)
        if P is not None and P.shape[1] == 3:
            _plot_vector_series(
                t, P,
                title=f"{foot_name} foot position",
                xlabel="Time (s)", ylabel="pos (m)",
                labels=["x", "y", "z"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_pos.png"),
            )

        # Orientación – componentes del cuaternión
        Q = _ensure_2d(ori_log)  # (T,4)
        if Q is not None and Q.shape[1] == 4:
            _plot_vector_series(
                t, Q,
                title=f"{foot_name} foot quaternion [x y z w]",
                xlabel="Time (s)", ylabel="quat",
                labels=["qx", "qy", "qz", "qw"],
                filename=os.path.join(outdir, f"{foot_name.lower()}_foot_quat.png"),
            )
            # Euler ZYX (yaw, pitch, roll) para leer orientación fácilmente
            YPR = _quat_series_to_euler_zyx(Q, degrees=True)  # (T,3)
            if YPR is not None:
                _plot_vector_series(
                    t, YPR,
                    title=f"{foot_name} foot orientation (Euler ZYX)",
                    xlabel="Time (s)", ylabel="deg",
                    labels=["yaw(Z)", "pitch(Y)", "roll(X)"],
                    filename=os.path.join(outdir, f"{foot_name.lower()}_foot_euler_zyx.png"),
                )

    # Si hay logs de pos/ori, calcula t coherente y pinta
    if pos_left_log is not None or ori_left_log is not None:
        T_L = len(pos_left_log) if pos_left_log is not None else len(ori_left_log)
        tL  = _resolve_time(time, T_L, dt)
        _plot_pose_series(tL, pos_left_log, ori_left_log, "Left")

    if pos_right_log is not None or ori_right_log is not None:
        T_R = len(pos_right_log) if pos_right_log is not None else len(ori_right_log)
        tR  = _resolve_time(time, T_R, dt)
        _plot_pose_series(tR, pos_right_log, ori_right_log, "Right")
    
    if f_ext_log is not None:

        WL = np.array(np.asarray(f_ext_log)[:, 0])
        WR = np.array(np.asarray(f_ext_log)[:, 1])
        WP = np.array(np.asarray(f_ext_log)[:, 2])

        WT = WL + WR + WP
        _plot_vector_series(t, WT, title = "Computed external wrench", xlabel = "Time (s)", ylabel = "Force/Torque", labels = ["Fx (N)", "Fy(N)", "Fz(N)", "Mx(Nm)", "My(Nm)", "Mz(Nm)"], filename = os.path.join(outdir, "External_wrench.png"))
    
    if f_real_log is not None:
        T = len(f_real_log)
        t = _resolve_time(time, T, dt)
        labels_wrench = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]
        _plot_vector_series(
            t, f_real_log, 
            title="Total Hydrodynamic Wrench on CoM", 
            xlabel="Time (s)", 
            ylabel="Force / Torque", 
            labels=labels_wrench, 
            filename=os.path.join(outdir, "hydro_wrench_total.png")
        )

    if f_A_r_log is not None:
        T = len(f_A_r_log)
        t = _resolve_time(time, T, dt)
        labels_wrench = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]
        _plot_vector_series(
            t, f_A_r_log, 
            title="Archimedes Buoyancy Wrench on CoM", 
            xlabel="Time (s)", 
            ylabel="Force / Torque", 
            labels=labels_wrench, 
            filename=os.path.join(outdir, "hydro_wrench_archimedes.png")
        )

    if f_D_r_log is not None:
        T = len(f_D_r_log)
        t = _resolve_time(time, T, dt)
        labels_wrench = ["Fx (N)", "Fy (N)", "Fz (N)", "Mx (Nm)", "My (Nm)", "Mz (Nm)"]
        _plot_vector_series(
            t, f_D_r_log, 
            title="Hydrodynamic Drag Wrench on CoM", 
            xlabel="Time (s)", 
            ylabel="Force / Torque", 
            labels=labels_wrench, 
            filename=os.path.join(outdir, "hydro_wrench_drag.png")
        )

    if f_ext_log is not None and f_real_log is not None:
        # Asegurarse de que los logs tienen la misma longitud
        if len(f_ext_log) == len(f_real_log):
            T = len(f_real_log)
            t = _resolve_time(time, T, dt)

            # Sumar los componentes del wrench calculado (datos completos)
            WL = np.array(np.asarray(f_ext_log)[:, 0])
            WR = np.array(np.asarray(f_ext_log)[:, 1])
            WP = np.array(np.asarray(f_ext_log)[:, 2])
            WT_computed = WL + WR + WP

            # Convertir el wrench real a un array de numpy (datos completos)
            f_real_arr = _ensure_2d(f_real_log)

            # --- FILTRADO DE DATOS PARA t > 1s ---
            mask = t > 1
            if not np.any(mask):
                print("[plots] Aviso: No hay datos para t > 1s. No se puede graficar el error filtrado.")
            else:
                t_filt = t[mask]
                WT_computed_filt = WT_computed[mask]
                f_real_arr_filt = f_real_arr[mask]
                # ------------------------------------

                # Calcular el error relativo sobre los datos filtrados
                abs_error = np.abs(WT_computed_filt - f_real_arr_filt)
                relative_error = abs_error / (np.abs(f_real_arr_filt) + 1e-8)
                
                # Etiquetas para el gráfico
                labels_error = ["ΔFx rel", "ΔFy rel", "ΔFz rel", "ΔMx rel", "ΔMy rel", "ΔMz rel"]
                
                _plot_vector_series(
                    t_filt,  # <--- Usar tiempo filtrado
                    relative_error,
                    title="Error Relativo: Wrench Calculado vs. Real (para t > 1s)", # <--- Título actualizado
                    xlabel="Tiempo (s)",
                    ylabel="Error Relativo (adimensional)",
                    labels=labels_error,
                    filename=os.path.join(outdir, "wrench_error_relative_log_t1.png"), # <--- Nombre de archivo nuevo
                    yscale='log'
                )
        else:
            print("[plots] Aviso: f_ext_log y f_real_log tienen longitudes diferentes. No se puede graficar el error.")
