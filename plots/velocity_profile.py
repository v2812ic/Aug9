import os
import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_profile_basic(outdir="plots", filename="perfil_velocidad_continuo.png"):
    """
    Grafica el perfil de velocidad con el MISMO estilo que el resto de tus figuras:
      - Matplotlib puro (sin estilos externos)
      - grid=True
      - sin colores forzados
      - guarda PNG en outdir
    """

    # --- Definición de funciones ---
    def U_log_layer(z):
        # Usa log natural (np.log) como en tu script original.
        # Si prefieres base 10, cambia a: return 0.15 * np.log10(10 * z) + 0.231
        return 0.15 * np.log(10 * z) + 0.231

    def U_sub_layer(z):
        return 2.574 * z

    # --- Parámetros y datos ---
    z_transition = 0.058275  # transición indicada en tu script

    # Segmentos ascendentes (evita z=0 para el log)
    z1 = np.linspace(1e-6, z_transition, 200, endpoint=True)  # subcapa
    z2 = np.linspace(z_transition, 1.5, 1000, endpoint=True)   # capa logarítmica

    U1 = U_sub_layer(z1)
    U2 = U_log_layer(z2)

    z_full = np.concatenate((z1, z2))
    U_full = np.concatenate((U1, U2))

    # --- Graficado con estilo básico consistente ---
    fig, ax = plt.subplots()
    ax.plot(U_full, z_full)

    ax.set_title('Velocity profile')
    ax.set_xlabel('Velocity [m/s]')
    ax.set_ylabel('Height [m]')
    ax.grid(True)   
    plt.xlim(0, 0.7)

    # Guardado
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)

# Uso:
plot_velocity_profile_basic()
