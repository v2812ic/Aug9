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
        # 'trajectory_width': 1.0, # Not needed anymore as lines are removed
        # 'trajectory_alpha': 0.5  # Not needed anymore as lines are removed
    },
    'figure': {
        'size_zmp': (14, 7), # A bit wider for the colorbar
        'size_timeline': (10, 3)
    },
    'colormap': 'viridis' # Colormap for time gradient
}

def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])

def _sum_wrenches_per_timestep(list_of_lists):
    return np.array([np.sum(np.asarray(L), axis=0) if L else np.zeros(6) for L in list_of_lists])

def plot_zmp(time, cf_left_log, cf_right_log,
             pos_left_log, ori_left_log, pos_right_log, ori_right_log,
             outdir="plots/zmp"):
    print("--- Generating ZMP plots (Time-Encoded Version) ---")
    os.makedirs(outdir, exist_ok=True)
    
    time_arr = np.array(time)
    cfL_W = _sum_wrenches_per_timestep(cf_left_log)
    cfR_W = _sum_wrenches_per_timestep(cf_right_log)

    foot_poly = np.array([[0.09, 0.02], [0.09, -0.02], [-0.05, -0.02], [-0.05, 0.02], [0.09, 0.02]])

    T = len(time_arr)
    zmpL, zmpR = np.zeros((T,2)), np.zeros((T,2))
    for i in range(T):
        R_LW = _quat_to_R(np.array(ori_left_log[i])).T
        R_RW = _quat_to_R(np.array(ori_right_log[i])).T
        
        fL_L, mL_L = R_LW @ cfL_W[i, :3], R_LW @ cfL_W[i, 3:6]
        fR_R, mR_R = R_RW @ cfR_W[i, :3], R_RW @ cfR_W[i, 3:6]
        
        if abs(fL_L[2]) > 1e-3: zmpL[i] = [-mL_L[1]/fL_L[2], mL_L[0]/fL_L[2]]
        if abs(fR_R[2]) > 1e-3: zmpR[i] = [-mR_R[1]/fR_R[2], mR_R[0]/fR_R[2]]

    # ZMP Distribution Plot with Time Gradient
    fig, axes = plt.subplots(1, 2, figsize=STYLE_GUIDE['figure']['size_zmp'], sharey=True)
    fig.suptitle('ZMP Trajectory in Local Foot Frame', fontsize=STYLE_GUIDE['fonts']['suptitle'])

    # Calcular límites de los ejes para asegurar que el pie se vea entero
    x_lim_min = min(foot_poly[:, 0].min(), zmpL[:,0].min(), zmpR[:,0].min()) - 0.01
    x_lim_max = max(foot_poly[:, 0].max(), zmpL[:,0].max(), zmpR[:,0].max()) + 0.01
    y_lim_min = min(foot_poly[:, 1].min(), zmpL[:,1].min(), zmpR[:,1].min()) - 0.01
    y_lim_max = max(foot_poly[:, 1].max(), zmpL[:,1].max(), zmpR[:,1].max()) + 0.01

    # Plot para el Pie Izquierdo
    axes[0].plot(foot_poly[:,0], foot_poly[:,1], color=STYLE_GUIDE['colors']['polygon'], label='Support Polygon')
    # Eliminamos la línea que unía los puntos: axes[0].plot(zmpL[:, 0], zmpL[:, 1], ...)
    scL = axes[0].scatter(zmpL[:,0], zmpL[:,1], s=15, c=time_arr, cmap=STYLE_GUIDE['colormap'], vmin=time_arr[0], vmax=time_arr[-1])
    axes[0].set_title('Left Foot', fontsize=STYLE_GUIDE['fonts']['title'])
    axes[0].set_xlabel('Local X [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[0].set_ylabel('Local Y [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'])
    # Ajustamos la posición de la leyenda para evitar superposición
    axes[0].legend(loc='upper left', fontsize=STYLE_GUIDE['fonts']['legend'])
    axes[0].set_xlim(x_lim_min, x_lim_max)
    axes[0].set_ylim(y_lim_min, y_lim_max)

    # Plot para el Pie Derecho
    axes[1].plot(foot_poly[:,0], foot_poly[:,1], color=STYLE_GUIDE['colors']['polygon'])
    # Eliminamos la línea que unía los puntos: axes[1].plot(zmpR[:, 0], zmpR[:, 1], ...)
    scR = axes[1].scatter(zmpR[:,0], zmpR[:,1], s=15, c=time_arr, cmap=STYLE_GUIDE['colormap'], vmin=time_arr[0], vmax=time_arr[-1])
    axes[1].set_title('Right Foot', fontsize=STYLE_GUIDE['fonts']['title'])
    axes[1].set_xlabel('Local X [m]', fontsize=STYLE_GUIDE['fonts']['label'])
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'])
    axes[1].set_xlim(x_lim_min, x_lim_max)
    axes[1].set_ylim(y_lim_min, y_lim_max)

    # Añadir la barra de color
    # Usamos cax para un control más fino y evitamos que se superponga
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(scR, cax=cbar_ax, label='Time [s]')
    
    for ax in axes: ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
    
    plt.tight_layout(rect=[0,0,0.9,0.95]) # Ajustar rect para dejar espacio a la colorbar
    fig.savefig(f"{outdir}/zmp_trajectory_local.pdf")
    plt.close(fig)

    # Stability Timeline Plot (sin cambios respecto a tu versión anterior)
    fig2, ax2 = plt.subplots(figsize=STYLE_GUIDE['figure']['size_timeline'])
    # Nota: Aquí no hay 'insideL' ni 'insideR' definidos, si necesitas esta gráfica,
    # deberías volver a calcular 'insideL' e 'insideR'
    # o eliminar esta parte si ya no es relevante con el enfoque de trayectoria.
    # Por ahora, la he comentado para evitar un error, ya que 'insideL' no existe.
    # Si quieres que se muestre, tendrás que definir 'insideL' y 'insideR' como antes.
    # ax2.plot(time, insideL.astype(int), label='Left Foot Stable', color=STYLE_GUIDE['colors']['blue'])
    # ax2.plot(time, insideR.astype(int), label='Right Foot Stable', color=STYLE_GUIDE['colors']['orange'])
    # ax2.set_yticks([0,1], ['Outside','Inside'])
    # ax2.set_xlabel('Time (s)', fontsize=STYLE_GUIDE['fonts']['label'])
    # ax2.set_title('ZMP Stability Timeline', fontsize=STYLE_GUIDE['fonts']['title'])
    # ax2.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style']); ax2.legend(fontsize=STYLE_GUIDE['fonts']['legend'])
    # ax2.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
    # fig2.tight_layout()
    # fig2.savefig(f"{outdir}/zmp_stability_timeline.pdf")
    # plt.close(fig2)

    print(f" -> ZMP plots saved in '{outdir}'.")