import os
import pandas as pd
import matplotlib.pyplot as plt

def plotTasks():
    """
    Busca datos de tareas en 'experiment_data/tasks', los grafica y guarda los
    gráficos en 'plots/tasks'. Asume que los datos ya han sido generados.
    """
    print("--- Iniciando el proceso de graficado ---")
    base_data_dir = "experiment_data/tasks"
    base_plot_dir = "plots/tasks"

    # Asegurarse de que el directorio de salida para los gráficos exista
    os.makedirs(base_plot_dir, exist_ok=True)

    # Verificar si el directorio de datos de entrada existe
    if not os.path.isdir(base_data_dir):
        print(f"Error: El directorio de datos '{base_data_dir}' no fue encontrado.")
        print("Asegúrate de que tu programa en C++ haya generado los datos antes de ejecutar este script.")
        return

    # Obtener la lista de tareas
    try:
        task_list = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
        if not task_list:
            print(f"No se encontraron directorios de tareas en '{base_data_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: El directorio '{base_data_dir}' no existe.")
        return

    # Iterar sobre cada tarea para crear su gráfico
    for task_name in task_list:
        print(f"Procesando tarea: {task_name}...")
        task_data_dir = os.path.join(base_data_dir, task_name)
        
        pos_err_path = os.path.join(task_data_dir, "pos_err.csv")
        vel_err_path = os.path.join(task_data_dir, "vel_err.csv")

        try:
            df_pos_err = pd.read_csv(pos_err_path)
            df_vel_err = pd.read_csv(vel_err_path)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            title_name = task_name.replace("_", " ").title()
            fig.suptitle(f'Errores de la Tarea: {title_name}', fontsize=16)

            # --- ARREGLO AQUÍ ---
            # Graficar error de posición convirtiendo a NumPy
            for col in df_pos_err.columns:
                if col != 't':
                    axes[0].plot(df_pos_err['t'].to_numpy(), df_pos_err[col].to_numpy(), label=col)
            
            axes[0].set_title("Error de Posición")
            axes[0].set_ylabel("Error")
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)

            # --- ARREGLO AQUÍ ---
            # Graficar error de velocidad convirtiendo a NumPy
            for col in df_vel_err.columns:
                if col != 't':
                    axes[1].plot(df_vel_err['t'].to_numpy(), df_vel_err[col].to_numpy(), label=col)

            axes[1].set_title("Error de Velocidad")
            axes[1].set_ylabel("Error")
            axes[1].set_xlabel("Tiempo (s)")
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            output_path = os.path.join(base_plot_dir, f"{task_name}.pdf")
            plt.savefig(output_path)
            plt.close(fig)
            print(f" -> Gráfico guardado en: {output_path}")

        except FileNotFoundError as e:
            print(f" -> Advertencia: No se encontró el archivo {e.filename} para la tarea '{task_name}'.")
        except Exception as e:
            print(f" -> Error inesperado procesando la tarea '{task_name}': {e}")

    print("\n--- Proceso de graficado completado. ---")
