import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data
from typing import Sequence

# -----------------------------------------------------------------------------
#                    FUNCIONES DE AYUDA (Sin cambios)
# -----------------------------------------------------------------------------

def _frame_id_from_link_idx(model, g1_humanoid, link_idx: int) -> int:
    if link_idx == -1:
        root_name = pb.getBodyInfo(g1_humanoid)[0].decode('utf-8')
        return model.getFrameId(root_name)
    else:
        link_name = pb.getJointInfo(g1_humanoid, link_idx)[12].decode('utf-8')
        return model.getFrameId(link_name)

def _wrench_translate(wrench: np.ndarray, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """
    Traslada una wrench (vector de 6) desde p_from hasta p_to.
    M_to = M_from + (p_from - p_to) x F
    """
    F = wrench[:3]
    M_from = wrench[3:]
    p_from = np.asarray(p_from).reshape(3)
    p_to = np.asarray(p_to).reshape(3)
    
    M_to = M_from + np.cross(p_from - p_to, F)
    return np.hstack([F, M_to])

def _get_link_origin_world(model, data, g1_humanoid, link_idx):
    """Posición WORLD del origen del frame de Pinocchio para ese link."""
    fid = _frame_id_from_link_idx(model, g1_humanoid, link_idx)
    return np.asarray(data.oMf[fid].translation).reshape(3)

def _get_link_com_world(g1_humanoid, link_idx):
    """Función de ayuda para obtener el CoM de un link si lo necesitas."""
    if link_idx == -1:
        base_pos, base_orn = pb.getBasePositionAndOrientation(g1_humanoid)
        R = np.array(pb.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
        local_inertial_pos = np.asarray(pb.getDynamicsInfo(g1_humanoid, -1)[3], dtype=float)
        return np.asarray(base_pos) + R @ local_inertial_pos
    else:
        ls = pb.getLinkState(g1_humanoid, link_idx, computeForwardKinematics=True)
        return np.asarray(ls[0], dtype=float)

# -----------------------------------------------------------------------------
#          FUNCIÓN COMPLETA Y DEFINITIVA
# -----------------------------------------------------------------------------

def solve_force(
    link_idxs: Sequence[int],
    p_link_com_world: np.ndarray,
    tau_ext: np.ndarray,
    g1_humanoid: int,
    model: pin.Model,
    data: pin.Data,
    q_pin: np.ndarray
) -> np.ndarray:
    """
    Estima las wrenches para varios links usando la matriz de jacobianos apilados,
    traslada cada wrench al mismo punto p_link_com_world y devuelve la suma.

    Args:
        link_idxs (Sequence[int]): Índices de los links a analizar (-1 permitido para la base).
        p_link_com_world (np.ndarray): Punto común (x,y,z) en WORLD al que trasladar todas las wrenches.
        tau_ext (np.ndarray): Vector de torques externos residuales (nv,).
        g1_humanoid, model, data, q_pin: Objetos de simulación y estado.

    Returns:
        np.ndarray: Wrench total [Fx, Fy, Fz, Mx, My, Mz] en el punto común p_link_com_world.
    """
    # -- Preparación de entradas --
    link_idxs = list(link_idxs)
    if len(link_idxs) == 0:
        return np.zeros(6)

    tau_ext = np.asarray(tau_ext, dtype=float).reshape(-1)
    p_common = np.asarray(p_link_com_world, dtype=float).reshape(3)

    # -- Cinemática --
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    # -- Construir jacobianos apilados y orígenes de frame --
    J_blocks = []             # cada bloque es (nv x 6)
    frame_origins = []        # posiciones (3,) del origen del frame de cada link en WORLD

    for idx in link_idxs:
        fid = _frame_id_from_link_idx(model, g1_humanoid, idx)

        # Jacobiano 6D en LOCAL, transpuesto -> (nv x 6) para cumplir tau = J @ w
        J6_worldT = pin.computeFrameJacobian(
            model, data, q_pin, fid, pin.ReferenceFrame.LOCAL
        ).T
        J_blocks.append(J6_worldT)

        frame_origins.append(np.asarray(data.oMf[fid].translation).reshape(3))

    # Apilar horizontalmente: J_stack (nv x 6m)
    J_stack = np.hstack(J_blocks)

    # -- Resolver todas las wrenches a la vez --
    # tau_ext ≈ J_stack @ w_stack, donde w_stack = [w1; w2; ...; wm] (6m,)
    w_stack = np.linalg.pinv(J_stack) @ tau_ext
    # (opcional, regularizada):
    # lam = 1e-8
    # JT = J_stack.T
    # w_stack = JT @ np.linalg.solve(J_stack @ JT + lam * np.eye(J_stack.shape[0]), tau_ext)

    # -- Trasladar cada wrench al punto común y sumar --
    total_wrench = np.zeros(6)
    m = len(link_idxs)
    for i in range(m):
        w_i_at_frame = w_stack[6*i : 6*(i+1)]           # (6,)
        p_from = frame_origins[i]                       # origen del frame i (WORLD)

        # trasladar desde el origen del frame del link i al punto común
        w_i_at_common = _wrench_translate(
            wrench=w_i_at_frame,
            p_from=p_from,
            p_to=p_common
        )

        total_wrench += w_i_at_common

    return total_wrench