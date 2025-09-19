import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import re
import numpy as np

STYLE_GUIDE = {
    'colors': {
        'blue': '#4169E1',
        'green': '#3CB371',
        'orange': '#FF8C00',
        'red': '#DC143C',
        'purple': '#BA55D3',
        'text': '#333333',
        'grid': '#CCCCCC'
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
        'size': (12, 8),
        'dpi': 300
    }
}

def _load_offsets(path="experiment_data/offsets.txt"):
    if not os.path.exists(path):
        return None
    try:
        M = pd.read_csv(path, header=None, delim_whitespace=True)
        if M.shape[1] < 4:
            print(f"[plotter] Warning: {path} debe tener 4 columnas [ox oy oz t]. Ignorando offsets.")
            return None
        M = M.iloc[:, :4]
        M.columns = ["ox", "oy", "oz", "t"]
        M = M.replace([np.inf, -np.inf], np.nan).dropna()
        if M.empty:
            return None
        M = M.drop_duplicates(subset="t", keep="first").sort_values("t")
        if len(M) < 2:
            print(f"[plotter] Warning: {path} no tiene suficientes muestras para interpolar.")
            return None
        return M
    except Exception as e:
        print(f"[plotter] Warning leyendo offsets: {e}")
        return None

def _is_com_xy_task(task_name: str) -> bool:
    low = task_name.lower()
    return ("com" in low) and ("xy" in low or "_x_" in low or "_y_" in low or low.endswith("_x_task") or low.endswith("_y_task") or "pos_task" in low)

def plotTasks(plot_from_time: float = 0.0):
    print("--- Starting task plotting process ---")
    base_data_dir = "experiment_data/tasks"
    base_plot_dir = "plots/tasks"
    os.makedirs(base_plot_dir, exist_ok=True)

    if not os.path.isdir(base_data_dir):
        print(f"Error: Data directory '{base_data_dir}' not found.")
        return

    try:
        task_list = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
        if not task_list:
            print(f"No task directories found in '{base_data_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Directory '{base_data_dir}' does not exist.")
        return

    default_label_map = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    plot_color_keys = ['blue', 'green', 'orange', 'red', 'purple']
    plot_colors = [STYLE_GUIDE['colors'][k] for k in plot_color_keys]

    def get_plot_labels(df, task_name_str):
        if df is None:
            return default_label_map
        data_cols = [c for c in df.columns if c != 't']
        if len(data_cols) == 1:
            low = task_name_str.lower()
            if low.endswith(('_x_task', '_x_pos_task', '_x_ori_task')):
                return ['x']
            if low.endswith(('_y_task', '_y_pos_task', '_y_ori_task')):
                return ['y']
            if low.endswith(('_z_task', '_z_pos_task', '_z_ori_task')):
                return ['z']
        return default_label_map

    offsets_df_global = _load_offsets("experiment_data/offsets.txt")

    for task_name in task_list:
        print(f"Processing task: {task_name}...")
        task_data_dir = os.path.join(base_data_dir, task_name)
        paths = {
            "pos_err": os.path.join(task_data_dir, "pos_err.csv"),
            "vel_err": os.path.join(task_data_dir, "vel_err.csv"),
            "des_pos": os.path.join(task_data_dir, "des_pos.csv"),
            "des_vel": os.path.join(task_data_dir, "des_vel.csv"),
            "des_acc": os.path.join(task_data_dir, "des_acc.csv"),
        }

        try:
            dfs_raw = {k: (pd.read_csv(v) if os.path.exists(v) else None) for k, v in paths.items()}
            if all(v is None for v in dfs_raw.values()):
                print(f" -> Warning: No data files found for task '{task_name}'. Skipping.")
                continue

            def filter_by_time(df):
                if df is None or 't' not in df.columns:
                    return df
                return df[df['t'] >= plot_from_time].reset_index(drop=True)

            dfs = {k: filter_by_time(v) for k, v in dfs_raw.items()}

            if task_name == "pelvis_ori_task":
                #pos_missing = (dfs.get("pos_err") is None) or (dfs["pos_err"] is not None and dfs["pos_err"].empty)
                vel_present = (dfs.get("vel_err") is not None) and (not dfs["vel_err"].empty)

                if vel_present:
                    vdf = dfs["vel_err"].copy()
                    if "t" in vdf.columns:
                        t = vdf["t"].to_numpy()
                        data_cols = [c for c in vdf.columns if c != "t"]

                        # posición = 0 antes de t0; integrar por trapecio desde t0 en adelante
                        t0 = 3
                        mask = t >= t0

                        pos_df = pd.DataFrame({"t": t})
                        for c in data_cols:
                            v = vdf[c].to_numpy()
                            pos = np.zeros_like(v, dtype=float)

                            if np.any(mask):
                                t_idx = t[mask]
                                v_idx = v[mask]

                                if t_idx.size >= 2:
                                    dt = np.diff(t_idx)
                                    avg = 0.5 * (v_idx[:-1] + v_idx[1:])
                                    csum = np.concatenate(([0.0], np.cumsum(avg * dt)))  # integral acumulada con trapecio
                                    pos[mask] = csum
                                else:
                                    # Si sólo hay un punto ≥ t0, la integral sigue siendo 0
                                    pos[mask] = 0.0

                            pos_df[c] = pos

                        # Sustituimos/creamos pos_err sintético para esta tarea
                        dfs["pos_err"] = pos_df

            if not any(v is not None and not v.empty for v in dfs.values()):
                print(f" -> Warning: No samples with t >= {plot_from_time} for task '{task_name}'. Skipping.")
                continue

            label_maps = {k: get_plot_labels(dfs[k], task_name) for k in dfs.keys()}

            # ---------- FIGURA DE ERRORES ----------
            error_panels = []
            if dfs["pos_err"] is not None and not dfs["pos_err"].empty:
                error_panels.append(("pos_err", "Position Error", "Error [SI]"))
            if dfs["vel_err"] is not None and not dfs["vel_err"].empty:
                error_panels.append(("vel_err", "Velocity Error", "Error [SI]"))

            if error_panels:
                nrows = len(error_panels)
                fig_err, axes_err = plt.subplots(nrows, 1, figsize=STYLE_GUIDE['figure']['size'], sharex=True)
                if nrows == 1:
                    axes_err = [axes_err]

                title_name = task_name.replace("_", " ").title().replace("Com", "CoM").replace("Xy", "XY")
                fig_err.suptitle(title_name, fontsize=STYLE_GUIDE['fonts']['suptitle'], color=STYLE_GUIDE['colors']['text'])

                def plot_df(ax, df, label_map):
                    color_cycle = itertools.cycle(plot_colors)
                    for col in df.columns:
                        if col == 't':
                            continue
                        label = col
                        m = re.search(r'\d+$', col)
                        if m:
                            idx = int(m.group(0))
                            if idx < len(label_map):
                                label = label_map[idx]
                        ax.plot(df['t'].to_numpy(), df[col].to_numpy(),
                                label=label, color=next(color_cycle),
                                linewidth=STYLE_GUIDE['lines']['width'])

                for ax, (key, ttl, ylab) in zip(axes_err, error_panels):
                    plot_df(ax, dfs[key], label_maps[key])
                    ax.set_title(ttl, fontsize=STYLE_GUIDE['fonts']['title'])
                    ax.set_ylabel(ylab, fontsize=STYLE_GUIDE['fonts']['label'])
                    ax.legend(fontsize=STYLE_GUIDE['fonts']['legend'], loc='upper right')
                    ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
                    ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])

                axes_err[-1].set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
                t_max_errors = max(df['t'].max() for (k, _, _) in error_panels for df in [dfs[k]] if df is not None and not df.empty)
                for ax in axes_err:
                    ax.set_xlim(plot_from_time, t_max_errors)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_err = os.path.join(base_plot_dir, f"{task_name}_errors.pdf")
                plt.savefig(out_err, dpi=STYLE_GUIDE['figure']['dpi'])
                plt.close(fig_err)
                print(f" -> Errors plot saved to: {out_err}")
            else:
                print(f" -> No error data to plot for '{task_name}'.")

            # ---------- FIGURA DE DESEADOS ----------
            desired_panels = []
            if dfs["des_pos"] is not None and not dfs["des_pos"].empty:
                desired_panels.append(("des_pos", "Desired Position", "Value [SI]"))
            if dfs["des_vel"] is not None and not dfs["des_vel"].empty:
                desired_panels.append(("des_vel", "Desired Velocity", "Value [SI]"))
            if dfs["des_acc"] is not None and not dfs["des_acc"].empty:
                desired_panels.append(("des_acc", "Desired Acceleration", "Value [SI]"))

            is_com_xy = _is_com_xy_task(task_name)
            have_offsets = offsets_df_global is not None

            if desired_panels:
                if is_com_xy and have_offsets:
                    desired_panels = [("offsets", "Offsets (ox, oy)", "Offset [SI]")] + desired_panels

                nrows = len(desired_panels)
                fig_des, axes_des = plt.subplots(nrows, 1, figsize=STYLE_GUIDE['figure']['size'], sharex=True)
                if nrows == 1:
                    axes_des = [axes_des]

                title_name = task_name.replace("_", " ").title().replace("Com", "CoM").replace("Xy", "XY")
                fig_des.suptitle(title_name, fontsize=STYLE_GUIDE['fonts']['suptitle'], color=STYLE_GUIDE['colors']['text'])

                def plot_df(ax, df, label_map):
                    color_cycle = itertools.cycle(plot_colors)
                    for col in df.columns:
                        if col == 't':
                            continue
                        label = col
                        m = re.search(r'\d+$', col)
                        if m:
                            idx = int(m.group(0))
                            if idx < len(label_map):
                                label = label_map[idx]
                        ax.plot(df['t'].to_numpy(), df[col].to_numpy(),
                                label=label, color=next(color_cycle),
                                linewidth=STYLE_GUIDE['lines']['width'])

                # t_max para desired (incluyendo offsets si existen)
                t_max_candidates = []
                for key, _, _ in desired_panels:
                    if key == "offsets" and have_offsets:
                        t_off = offsets_df_global["t"].to_numpy()
                        if t_off.size:
                            t_max_candidates.append(t_off[t_off >= plot_from_time].max(initial=plot_from_time))
                    else:
                        dfk = dfs.get(key, None)
                        if dfk is not None and not dfk.empty:
                            t_max_candidates.append(dfk["t"].max())
                t_max_desired = max(t_max_candidates) if t_max_candidates else plot_from_time

                ax_idx = 0
                for (key, ttl, ylab) in desired_panels:
                    ax = axes_des[ax_idx]
                    if key == "offsets":
                        if have_offsets:
                            if dfs["des_pos"] is not None and not dfs["des_pos"].empty:
                                tq = dfs["des_pos"]["t"].to_numpy()
                            else:
                                tq = offsets_df_global["t"].to_numpy()
                                tq = tq[tq >= plot_from_time]
                            toff = offsets_df_global["t"].to_numpy()
                            if tq.size and toff.size:
                                ox = np.interp(tq, toff, offsets_df_global["ox"].to_numpy(),
                                               left=offsets_df_global["ox"].iloc[0],
                                               right=offsets_df_global["ox"].iloc[-1])
                                oy = np.interp(tq, toff, offsets_df_global["oy"].to_numpy(),
                                               left=offsets_df_global["oy"].iloc[0],
                                               right=offsets_df_global["oy"].iloc[-1])
                                ax.plot(tq, ox, label=r"$o_x$", linewidth=STYLE_GUIDE['lines']['width'])
                                ax.plot(tq, oy, label=r"$o_y$", linewidth=STYLE_GUIDE['lines']['width'])
                        ax.set_title(ttl, fontsize=STYLE_GUIDE['fonts']['title'])
                        ax.set_ylabel(ylab, fontsize=STYLE_GUIDE['fonts']['label'])
                        ax.legend(fontsize=STYLE_GUIDE['fonts']['legend'], loc='upper right')
                        ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
                        ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
                    else:
                        if key == "des_pos" and is_com_xy and have_offsets:
                            df_here = dfs[key]
                            if df_here is not None and not df_here.empty:
                                tq = df_here["t"].to_numpy()
                                toff = offsets_df_global["t"].to_numpy()
                                ox = np.interp(tq, toff, offsets_df_global["ox"].to_numpy(),
                                               left=offsets_df_global["ox"].iloc[0],
                                               right=offsets_df_global["ox"].iloc[-1])
                                oy = np.interp(tq, toff, offsets_df_global["oy"].to_numpy(),
                                               left=offsets_df_global["oy"].iloc[0],
                                               right=offsets_df_global["oy"].iloc[-1])

                                data_cols = [c for c in df_here.columns if c != 't']
                                color_cycle = itertools.cycle(plot_colors)

                                if len(data_cols) >= 1:
                                    c = next(color_cycle)
                                    ax.plot(tq, df_here[data_cols[0]].to_numpy(),
                                            label=r"$\mathrm{des}_{x}$", color=c,
                                            linewidth=STYLE_GUIDE['lines']['width'])
                                    ax.plot(tq, df_here[data_cols[0]].to_numpy() - ox,
                                            label=r"$\mathrm{des}_{x}-o_x$", linestyle='--', color=c,
                                            linewidth=STYLE_GUIDE['lines']['width'])
                                if len(data_cols) >= 2:
                                    c = next(color_cycle)
                                    ax.plot(tq, df_here[data_cols[1]].to_numpy(),
                                            label=r"$\mathrm{des}_{y}$", color=c,
                                            linewidth=STYLE_GUIDE['lines']['width'])
                                    ax.plot(tq, df_here[data_cols[1]].to_numpy() - oy,
                                            label=r"$\mathrm{des}_{y}-o_y$", linestyle='--', color=c,
                                            linewidth=STYLE_GUIDE['lines']['width'])
                                if len(data_cols) >= 3:
                                    c = next(color_cycle)
                                    ax.plot(tq, df_here[data_cols[2]].to_numpy(),
                                            label=r"$\mathrm{des}_{z}$", color=c,
                                            linewidth=STYLE_GUIDE['lines']['width'])

                                ax.set_title(ttl, fontsize=STYLE_GUIDE['fonts']['title'])
                                ax.set_ylabel(ylab, fontsize=STYLE_GUIDE['fonts']['label'])
                                ax.legend(fontsize=STYLE_GUIDE['fonts']['legend'], loc='upper right')
                                ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
                                ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
                        else:
                            plot_df(ax, dfs[key], label_maps[key])
                            ax.set_title(ttl, fontsize=STYLE_GUIDE['fonts']['title'])
                            ax.set_ylabel(ylab, fontsize=STYLE_GUIDE['fonts']['label'])
                            ax.legend(fontsize=STYLE_GUIDE['fonts']['legend'], loc='upper right')
                            ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
                            ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])

                    ax_idx += 1

                axes_des[-1].set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
                for ax in axes_des:
                    ax.set_xlim(plot_from_time, t_max_desired)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_des = os.path.join(base_plot_dir, f"{task_name}_desired.pdf")
                plt.savefig(out_des, dpi=STYLE_GUIDE['figure']['dpi'])
                plt.close(fig_des)
                print(f" -> Desired plot saved to: {out_des}")
            else:
                print(f" -> No desired data to plot for '{task_name}'.")

        except Exception as e:
            print(f" -> An unexpected error occurred while processing task '{task_name}': {e}")

    print("\n--- Plotting process completed. ---")
