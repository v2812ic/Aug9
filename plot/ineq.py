# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===============================================================
# Unified Style (bigger fonts; no legend/title in the final plot)
# ===============================================================
STYLE_GUIDE = {
    'colors': {
        'blue':   '#4169E1',   # RoyalBlue
        'green':  '#3CB371',
        'orange': '#FF8C00',
        'red':    '#DC143C',
        'purple': '#BA55D3',
        'text':   '#333333',
        'grid':   '#CCCCCC'
    },
    'fonts': {
        'label':  16,   # bigger axis labels
        'ticks':  12,   # bigger ticks
    },
    'lines': {
        'width':      2.0,
        'grid_style': '--'
    },
    'figure': {
        'size': (10, 10)
    }
}

# ===============================================================
# Parameters
# ===============================================================
# Foot geometry (half-lengths of the support rectangle)
X = 0.06    # half-length (m)
Y = 0.02   # half-width  (m)

# Friction
mu = 0.7

# Optional upper bound for fz (set to None if you don't want it)
fz_max = 1500.0

# Robot / LIPM parameters
m   = 35.11   # mass (kg)
g   = 9.81    # gravity (m/s^2)
z_c = 0.6    # CoM height (m)
b_sq = z_c / g

# Nominal fz (body weight)
fz_nominal = m * g

# Reference CoP position in the local foot frame (no y asymmetry)
y_cop = 0.0
x_cop = 0.0
z_cop = 0.6

# OFFSET: vector d = O' -> O (moves O' to geometric center O)
dx, dy, dz = 0.02, 0.0, 0.0

# Search range for (fx, fy) and resolution
N = 100
fx_lim = 150.0
fy_lim = 150.0
fx_vals = np.linspace(-fx_lim, fx_lim, N)
fy_vals = np.linspace(-fy_lim, fy_lim, N)

# ===============================================================
# Utilities
# ===============================================================
def _safe_makedirs_for(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ===============================================================
# Contact Wrench Cone (rectangle with friction and moment bounds)
# ===============================================================
def build_U(X, Y, mu, use_fz_max=True):
    """
    Matrix U such that U w + u >= 0 defines the foot's CWC.
    18 linear inequalities (standard linearization at edges/vertices).
    """
    U = np.zeros((18, 6))

    # 1) fz >= 0
    U[0, 2] = 1.0

    # 2) |fx| <= mu * fz
    U[1, [0, 2]] = [ 1.0,  mu]
    U[2, [0, 2]] = [-1.0,  mu]

    # 3) |fy| <= mu * fz
    U[3, [1, 2]] = [ 1.0,  mu]
    U[4, [1, 2]] = [-1.0,  mu]

    # 4) |mx| <= Y * fz
    U[5,  [2, 3]] = [ Y,  1.0]
    U[6,  [2, 3]] = [ Y, -1.0]

    # 5) |my| <= X * fz
    U[7,  [2, 4]] = [ X,  1.0]
    U[8,  [2, 4]] = [ X, -1.0]

    # 6) tz bounds (vertex linearization, conservative)
    rows = np.array([
        [ Y,  X, (X+Y)*mu, -mu, -mu,  1.0],
        [ Y, -X, (X+Y)*mu, -mu,  mu,  1.0],
        [-Y,  X, (X+Y)*mu,  mu, -mu,  1.0],
        [-Y, -X, (X+Y)*mu,  mu,  mu,  1.0],
        [-Y, -X, (X+Y)*mu, -mu, -mu, -1.0],
        [-Y,  X, (X+Y)*mu, -mu,  mu, -1.0],
        [ Y, -X, (X+Y)*mu,  mu, -mu, -1.0],
        [ Y,  X, (X+Y)*mu,  mu,  mu, -1.0],
    ])
    U[9:17, :] = rows

    # 7) fz <= fz_max (optional)
    if use_fz_max:
        U[17, 2] = -1.0

    return U

def build_u(fz_max=None):
    u = np.zeros(18)
    if fz_max is not None:
        u[17] = fz_max
    return u

U = build_U(X, Y, mu, use_fz_max=(fz_max is not None))
u = build_u(fz_max)

# ===============================================================
# Offset transform:  w_O = T w_O'
# ===============================================================
def build_T(dx, dy, dz):
    I3 = np.eye(3)
    L = np.array([[   0,  dz, -dy],
                  [ -dz,   0,  dx],
                  [  dy, -dx,   0]])
    return np.block([[I3, np.zeros((3,3))],
                     [L,  I3]])

T = build_T(dx, dy, dz)

# ===============================================================
# Feasibility (single foot) with the new wrench model
# ===============================================================
def feasible_mask_single_foot(fx_vals, fy_vals, m, g, b_sq, x_cop, y_cop, z_cop):
    FX, FY = np.meshgrid(fx_vals, fy_vals)
    ok = np.ones_like(FX, dtype=bool)
    fz_base = -m * g

    for i in range(FX.shape[0]):
        for j in range(FX.shape[1]):
            fx = FX[i, j]
            fy = FY[i, j]

            # Entire wrench is supported by a single foot
            f_x = -fx/2
            f_y = -fy/2
            f_z = -fz_base/2
            
            f = np.array([f_x, f_y, f_z])
            p = -np.array([x_cop, y_cop, z_cop])

            p[:2] = p[:2] - (b_sq/m * f[:2])
            
            mom = -np.cross(f, p)
            
            #print(mom)
            
            w_prime  = np.concatenate((f, mom))
            
            #print(w_prime)
            s = U @ T @ w_prime + u
            ok[i, j] = np.all(s >= -1e-9)

    return ok

# ===============================================================
# Compute feasibility for a single foot
# ===============================================================
OK = feasible_mask_single_foot(fx_vals, fy_vals, m, g, b_sq, x_cop, y_cop, z_cop)

# ===============================================================
# Axis intersections
# ===============================================================
idx_fx_zero = np.argmin(np.abs(fx_vals))
idx_fy_zero = np.argmin(np.abs(fy_vals))

def get_axis_limits(axis_data, values):
    idx = np.where(axis_data)[0]
    if len(idx) == 0:
        return (None, None)
    return values[idx.min()], values[idx.max()]

limits = {
    "Foot": {
        "fx": get_axis_limits(OK[idx_fy_zero, :], fx_vals),
        "fy": get_axis_limits(OK[:, idx_fx_zero], fy_vals)
    }
}

def fmt_lim_pair(p):
    if p[0] is None or p[1] is None:
        return "[N/A, N/A]"
    return f"[{p[0]:+.2f}, {p[1]:+.2f}]"

# Console output in English
print("--- REACTION FORCE LIMITS ON AXES (N) ---")
fx_min, fx_max = limits["Foot"]["fx"]
fy_min, fy_max = limits["Foot"]["fy"]
print("Single Foot:")
print(f"  Fx-axis (with Fy=0): {fmt_lim_pair((fx_min, fx_max))}")
print(f"  Fy-axis (with Fx=0): {fmt_lim_pair((fy_min, fy_max))}")
print("-----------------------------------------")

# ===============================================================
# Vector plot (no legend, no title)
# ===============================================================
fig, ax = plt.subplots(figsize=STYLE_GUIDE['figure']['size'])

# 1) Feasible region (translucent blue)
ax.contourf(
    fx_vals, fy_vals, OK,
    levels=[0.5, 1.5],
    colors=[STYLE_GUIDE['colors']['blue']],
    alpha=0.25
)

# 2) Boundary contour (blue line)
ax.contour(
    fx_vals, fy_vals, OK,
    levels=[0.5],
    colors=[STYLE_GUIDE['colors']['blue']],
    linewidths=STYLE_GUIDE['lines']['width']
)


# Axes/style (English labels, LaTeX symbols)
ax.set_xlabel(r'$F_x$ [N]', fontsize=STYLE_GUIDE['fonts']['label'])
ax.set_ylabel(r'$F_y$ [N]', fontsize=STYLE_GUIDE['fonts']['label'])
ax.set_aspect('equal', 'box')
ax.grid(True, linestyle=STYLE_GUIDE['lines']['grid_style'], color=STYLE_GUIDE['colors']['grid'])
ax.tick_params(axis='both', which='major', labelsize=STYLE_GUIDE['fonts']['ticks'])
ax.axhline(0, color='black', linewidth=0.7)
ax.axvline(0, color='black', linewidth=0.7)

# Save vector PDF and show
out_path = "plots/cwc/cwc_Fx_Fy_single.pdf"
_safe_makedirs_for(out_path)
fig.tight_layout()
fig.savefig(out_path)   # vector (PDF)
print(f"Figure saved to: {out_path}")

plt.show()             # display
