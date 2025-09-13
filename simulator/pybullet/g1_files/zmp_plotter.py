import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import os

def plot_zmp(time, cf_left_log, cf_right_log, pos_left_log, ori_left_log, pos_right_log, ori_right_log, outdir="plots/zmp"):
    """
    Calcula y grafica el ZMP para cada pie.
    - Suma los wrenches de los múltiples puntos de contacto para obtener un wrench total en el tobillo.
    - Asume que este wrench total ya está en el FRAME LOCAL del pie.
    - Utiliza la geometría del pie corregida.
    """
    print("--- Generando gráficos de ZMP (versión definitiva) ---")

    # === 1. Suma de wrenches de contacto para obtener un wrench total por pie ===
    def _get_total_wrench_log(log):
        """
        Suma los wrenches de los puntos de contacto para cada instante de tiempo.
        Entrada: [[wrench1, wrench2,...], [wrench1,...], [], ...]
        Salida: np.array de shape (T, 6)
        """
        total_wrenches = []
        for frame_wrenches in log:
            if frame_wrenches and len(frame_wrenches) > 0:
                # Suma todos los arrays de wrench en la lista del frame actual
                total_wrench = np.sum(np.asarray(frame_wrenches), axis=0)
                total_wrenches.append(total_wrench)
            else:
                # Si no hay contacto (lista vacía), el wrench total es cero
                total_wrenches.append(np.zeros(6))
        return np.array(total_wrenches)

    time = np.array(time)
    # cf_left_local y cf_right_local ahora son matrices (T, 6) con el wrench total en el tobillo
    cf_left_local = _get_total_wrench_log(cf_left_log)
    cf_right_local = _get_total_wrench_log(cf_right_log)

    # === 2. Geometría del pie (corregida) ===
    foot_half_length = 0.07
    foot_half_width = 0.02
    # Origen del link está 0.02m HACIA ATRÁS del centro geométrico
    foot_center_offset_x = 0.02 

    front_x = foot_center_offset_x + foot_half_length
    back_x = foot_center_offset_x - foot_half_length
    
    foot_corners_local = np.array([
        [front_x,  foot_half_width, 0], [front_x, -foot_half_width, 0],
        [back_x,  -foot_half_width, 0], [back_x,   foot_half_width, 0],
        [front_x,  foot_half_width, 0]
    ])

    zmp_left_local_coords, zmp_right_local_coords = [], []

    for i in range(len(time)):
        # Pie Izquierdo (wrench ya es local)
        fz_l = cf_left_local[i, 2]
        if fz_l > 1e-3:
            px_l = -cf_left_local[i, 4] / fz_l  # -tau_y / Fz
            py_l =  cf_left_local[i, 3] / fz_l  #  tau_x / Fz
        else:
            px_l, py_l = 0, 0
        zmp_left_local_coords.append([px_l, py_l])
        
        # Pie Derecho (wrench ya es local)
        fz_r = cf_right_local[i, 2]
        if fz_r > 1e-3:
            px_r = -cf_right_local[i, 4] / fz_r
            py_r =  cf_right_local[i, 3] / fz_r
        else:
            px_r, py_r = 0, 0
        zmp_right_local_coords.append([px_r, py_r])

    zmp_left_local_coords = np.array(zmp_left_local_coords)
    zmp_right_local_coords = np.array(zmp_right_local_coords)

    # === 4. Graficado ===
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle('Distribución del ZMP en el Frame Local del Pie', fontsize=16)

    # Pie Izquierdo
    axes[0].plot(foot_corners_local[:, 0], foot_corners_local[:, 1], 'k-', label='Límites del Pie')
    axes[0].scatter(zmp_left_local_coords[:, 0], zmp_left_local_coords[:, 1], s=5, alpha=0.5, label='ZMP Local')
    axes[0].set_title('Pie Izquierdo')
    axes[0].set_xlabel('Eje X Local (Adelante) [m]')
    axes[0].set_ylabel('Eje Y Local (Izquierda) [m]')
    axes[0].axis('equal'); axes[0].grid(True); axes[0].legend()

    # Pie Derecho
    axes[1].plot(foot_corners_local[:, 0], foot_corners_local[:, 1], 'k-')
    axes[1].scatter(zmp_right_local_coords[:, 0], zmp_right_local_coords[:, 1], s=5, alpha=0.5, c='r')
    axes[1].set_title('Pie Derecho')
    axes[1].set_xlabel('Eje X Local (Adelante) [m]')
    axes[1].axis('equal'); axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{outdir}/zmp_local_distribution.pdf")
    plt.close(fig)

    print(f" -> Gráfico de ZMP Local (definitivo) guardado en '{outdir}/zmp_local_distribution.pdf'.")