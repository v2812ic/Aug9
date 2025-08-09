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
        # fallback por si viene un iterable raro
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
        # Intentar colapsar últimas dims si viniera algo como (T, D, 1)
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
    d = os.path.dirname(filename)
    if d == "":
        d = "."
    os.makedirs(d, exist_ok=True)


# -----------------------------
# Plotting
# -----------------------------
def _plot_vector_series(t, Y, title, xlabel, ylabel, labels, filename):
    """
    Dibuja múltiples series columnares Y (T, D) en la misma figura.
    - t: (T,)
    - Y: (T, D) o (T,) -> se convierte a (T, D)
    - labels: lista de strings o None/str (se ajusta al número de columnas)
    """
    Y = _ensure_2d(Y)
    if Y is None or Y.size == 0:
        return
    T, D = Y.shape
    if len(t) != T:
        # Si viene traspuesta (D, T) intentamos corregir
        if Y.T.shape[0] == len(t):
            Y = Y.T
            T, D = Y.shape
        else:
            raise ValueError(f"Dimensiones no coherentes: len(t)={len(t)} vs Y.shape={Y.shape}")

    # Normalizar labels
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
    for i in range(D):
        plt.plot(t, Y[:, i], label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if D > 1:
        plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=150)
    plt.close()


# -----------------------------
# API pública
# -----------------------------
def make_plots(
    outdir="plots",
    *,
    time=None,     # opcional: vector de tiempo (T,)
    dt=None,       # opcional: si no pasás 'time', se infiere usando 'dt'
    tau_ext=None,  # lista/array (T,) o (T, nv) o lista de vectores -> se plotea norma + componentes
    # podés agregar más kwargs sin romper (se ignoran)
    **_ignore,
):
    """
    Genera figuras SOLO con lo que pases por kwargs.
    Ejemplo mínimo:
        make_plots(outdir="plots", time=time_log, tau_ext=tau_ext_log)

    Si no pasás 'time', pero sí 'dt', el eje temporal se infiere con la longitud de tau_ext.
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
                title="Norma de τ_ext",
                xlabel="Tiempo [s]",
                ylabel="||τ_ext||",
                labels=["||τ_ext||"],
                filename=os.path.join(outdir, "tau_ext_norm.png"),
            )
        except Exception as e:
            print(f"[plots] Error graficando norma de τ_ext: {e}")

        # Componentes
        _plot_vector_series(
            t, tau_ext_arr,
            title="Componentes de τ_ext",
            xlabel="Tiempo [s]",
            ylabel="τ_ext",
            labels=None,  # autogenera τ_ext[i]
            filename=os.path.join(outdir, "tau_ext_components.png"),
        )

    print(f"[plots] Figuras guardadas en: {os.path.abspath(outdir)}")
