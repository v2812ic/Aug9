import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data

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

def solve_force(link_idx: int, 
                                p_link_com_world: np.ndarray, 
                                tau_ext: np.ndarray, 
                                g1_humanoid: int, 
                                model: pin.Model, 
                                data: pin.Data, 
                                q_pin: np.ndarray) -> np.ndarray:
    """
    Estima la wrench para un único link y la devuelve trasladada al CoM de ese link.

    Args:
        link_idx (int): El índice del link a analizar (-1 para la base).
        p_link_com_world (np.ndarray): La posición (x,y,z) del CoM del link en coordenadas WORLD.
        tau_ext (np.ndarray): El vector de torques externos residuales.
        g1_humanoid, model, data, q_pin: Objetos de simulación y estado.

    Returns:
        np.ndarray: La wrench [Fx,Fy,Fz,Mx,My,Mz] estimada en el punto p_link_com_world.
    """
    # -- Paso 1: Estimar la wrench en el origen del frame del link --
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    fid = _frame_id_from_link_idx(model, g1_humanoid, link_idx)
    J6_worldT = pin.computeFrameJacobian(model, data, q_pin, fid, pin.ReferenceFrame.LOCAL).T

    tau_ext = np.asarray(tau_ext, dtype=float).reshape(-1)
    
    # Resolver la estimación inicial usando la pseudoinversa
    wrench_at_frame_origin = np.linalg.pinv(J6_worldT) @ tau_ext
    
    # -- Paso 2: Trasladar la wrench al CoM del link proporcionado --
    
    # Punto de partida: el origen del frame donde se calculó la wrench
    p_frame_origin = np.asarray(data.oMf[fid].translation).reshape(3)
    
    # Punto de destino: el CoM que nos han pasado como argumento
    p_link_com = np.asarray(p_link_com_world).reshape(3)

    final_wrench_at_com = _wrench_translate(
        wrench=wrench_at_frame_origin,
        p_from=p_frame_origin,
        p_to=p_link_com
    )
    
    return final_wrench_at_com