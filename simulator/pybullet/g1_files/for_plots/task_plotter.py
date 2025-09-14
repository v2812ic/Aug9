import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import re # Importar el módulo de expresiones regulares

# -----------------------------
# Style Guide (Based on LaTeX tcolorbox)
# -----------------------------
STYLE_GUIDE = {
    'colors': {
        # Paleta de colores más vivos y distinguibles
        'blue': '#4169E1',      # RoyalBlue
        'green': '#3CB371',     # MediumSeaGreen
        'orange': '#FF8C00',     # DarkOrange
        'red': '#DC143C',       # Crimson
        'purple': '#BA55D3',    # MediumOrchid
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

def plotTasks():
    """
    Searches for task data in 'experiment_data/tasks', plots them,
    and saves the graphics in 'plots/tasks'.
    """
    print("--- Starting task plotting process ---")
    base_data_dir = "experiment_data/tasks"
    base_plot_dir = "plots/tasks"
    os.makedirs(base_plot_dir, exist_ok=True)

    if not os.path.isdir(base_data_dir):
        print(f"Error: Data directory '{base_data_dir}' not found.")
        print("Please ensure your C++ program has generated the data before running this script.")
        return

    try:
        task_list = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
        if not task_list:
            print(f"No task directories found in '{base_data_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: Directory '{base_data_dir}' does not exist.")
        return

    # Mapa para las etiquetas de la leyenda por defecto
    default_label_map = ['x', 'y', 'z', 'rx', 'ry', 'rz']

    for task_name in task_list:
        print(f"Processing task: {task_name}...")
        task_data_dir = os.path.join(base_data_dir, task_name)
        pos_err_path = os.path.join(task_data_dir, "pos_err.csv")
        vel_err_path = os.path.join(task_data_dir, "vel_err.csv")

        try:
            has_pos = os.path.exists(pos_err_path)
            has_vel = os.path.exists(vel_err_path)
            if not has_pos and not has_vel:
                print(f" -> Warning: No data files found for task '{task_name}'. Skipping.")
                continue

            df_pos_err = pd.read_csv(pos_err_path) if has_pos else None
            df_vel_err = pd.read_csv(vel_err_path) if has_vel else None

            fig, axes = plt.subplots(2, 1, figsize=STYLE_GUIDE['figure']['size'], sharex=True)
            
            title_name = task_name.replace("_", " ").title()
            title_name = title_name.replace("Com", "CoM").replace("Xy", "XY")
            
            fig.suptitle(f'Task Errors: {title_name}', fontsize=STYLE_GUIDE['fonts']['suptitle'], color=STYLE_GUIDE['colors']['text'])

            # --- LÓGICA DE ETIQUETADO INTELIGENTE ---
            def get_plot_labels(df, task_name_str):
                if df is None:
                    return default_label_map
                
                data_cols = [c for c in df.columns if c != 't']
                if len(data_cols) == 1: # Si es una tarea de 1D
                    if task_name_str.lower().endswith(('_x_task', '_x_pos_task', '_x_ori_task')):
                        return ['x']
                    if task_name_str.lower().endswith(('_y_task', '_y_pos_task', '_y_ori_task')):
                        return ['y']
                    if task_name_str.lower().endswith(('_z_task', '_z_pos_task', '_z_ori_task')):
                        return ['z']
                return default_label_map # Si no, usar el mapa por defecto

            pos_label_map = get_plot_labels(df_pos_err, task_name)
            vel_label_map = get_plot_labels(df_vel_err, task_name)
            
            # Graficar error de posición
            if df_pos_err is not None:
                color_cycle = itertools.cycle(STYLE_GUIDE['colors'].values())
                for col in df_pos_err.columns:
                    if col != 't':
                        label = col
                        match = re.search(r'\d+$', col)
                        if match:
                            col_index = int(match.group(0))
                            if col_index < len(pos_label_map):
                                label = pos_label_map[col_index]
                        
                        axes[0].plot(df_pos_err['t'].to_numpy(), df_pos_err[col].to_numpy(), label=label, color=next(color_cycle), linewidth=STYLE_GUIDE['lines']['width'])
            
            axes[0].set_title("Position Error", fontsize=STYLE_GUIDE['fonts']['title'])
            axes[0].set_ylabel("Error [SI]", fontsize=STYLE_GUIDE['fonts']['label'])
            axes[0].legend(fontsize=STYLE_GUIDE['fonts']['legend'])
            axes[0].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
            axes[0].tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])

            # Graficar error de velocidad
            if df_vel_err is not None:
                color_cycle = itertools.cycle(STYLE_GUIDE['colors'].values())
                for col in df_vel_err.columns:
                    if col != 't':
                        label = col
                        match = re.search(r'\d+$', col)
                        if match:
                            col_index = int(match.group(0))
                            if col_index < len(vel_label_map):
                                label = vel_label_map[col_index]

                        axes[1].plot(df_vel_err['t'].to_numpy(), df_vel_err[col].to_numpy(), label=label, color=next(color_cycle), linewidth=STYLE_GUIDE['lines']['width'])

            axes[1].set_title("Velocity Error", fontsize=STYLE_GUIDE['fonts']['title'])
            axes[1].set_ylabel("Error [SI]", fontsize=STYLE_GUIDE['fonts']['label'])
            axes[1].set_xlabel("Time [s]", fontsize=STYLE_GUIDE['fonts']['label'])
            axes[1].legend(fontsize=STYLE_GUIDE['fonts']['legend'])
            axes[1].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
            axes[1].tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            output_path = os.path.join(base_plot_dir, f"{task_name}_errors.pdf")
            plt.savefig(output_path)
            plt.close(fig)
            print(f" -> Plot saved to: {output_path}")

        except Exception as e:
            print(f" -> An unexpected error occurred while processing task '{task_name}': {e}")

    print("\n--- Plotting process completed. ---")