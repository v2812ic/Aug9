import numpy as np
import pinocchio as pin
import pybullet as pb
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
    Ambos puntos y la wrench deben estar en el mismo frame (WORLD).
    M_to = M_from + (p_from - p_to) x F
    """
    F = wrench[:3]
    M_from = wrench[3:]
    p_from = np.asarray(p_from).reshape(3)
    p_to = np.asarray(p_to).reshape(3)
    
    #print("F:", F.transpose())
    #print("M:", M_from.transpose())
    #print("p:", (p_to - p_from).transpose())
    # --- CORRECCIÓN CRÍTICA DEL SIGNO ---
    # La fórmula correcta para trasladar el momento usa un signo de suma.
    M_to = M_from + np.cross(p_from - p_to, F)
    
    return np.hstack([F, M_to])

# -----------------------------------------------------------------------------
#          FUNCIÓN PRINCIPAL CORREGIDA Y MEJORADA
# -----------------------------------------------------------------------------

def solve_force(
    link_idxs: Sequence[int],
    p_com_world: np.ndarray,
    tau_ext: np.ndarray,
    g1_humanoid: int,
    model: pin.Model,
    data: pin.Data,
    q_pin: np.ndarray
) -> tuple[np.ndarray, dict[int, np.ndarray]]: # <--- CAMBIO: Ahora devuelve una tupla
    """
    Estima las wrenches locales, las convierte al frame del mundo, las traslada 
    al CoM global del robot y devuelve la WRENCH TOTAL y las CONTRIBUCIONES INDIVIDUALES.
    """
    # ... (Preparación y cálculo de Jacobianos igual que antes) ...
    link_idxs = list(link_idxs)
    if not link_idxs:
        return np.zeros(6), {}

    tau_ext = np.asarray(tau_ext, dtype=float).reshape(-1)
    p_common = np.asarray(p_com_world, dtype=float).reshape(3)

    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    J_blocks = []
    frames_data = []

    for idx in link_idxs:
        fid = _frame_id_from_link_idx(model, g1_humanoid, idx)
        J6_local_T = pin.computeFrameJacobian(
            model, data, q_pin, fid, pin.ReferenceFrame.LOCAL
        ).T
        J_blocks.append(J6_local_T)
        frames_data.append({
            "origin_world": np.asarray(data.oMf[fid].translation).reshape(3),
            "rotation_world": data.oMf[fid].rotation
        })

    J_stack = np.hstack(J_blocks)
    w_stack_local = np.linalg.pinv(J_stack) @ tau_ext

    # -- 4. Convertir, trasladar, sumar Y GUARDAR CONTRIBUCIONES --
    total_wrench_at_com = np.zeros(6)
    
    # --- CAMBIO: Guardamos las contribuciones individuales ---
    individual_wrenches_at_com = {}
    
    num_links = len(link_idxs)
    
    for i in range(num_links):
        link_id = link_idxs[i] # Obtenemos el ID del link actual
        w_i_local = w_stack_local[6*i : 6*(i+1)]
        
        R_world_from_local = frames_data[i]["rotation_world"]
        F_world = R_world_from_local @ w_i_local[:3]
        M_world = R_world_from_local @ w_i_local[3:]
        w_i_world_at_frame_origin = np.hstack([F_world, M_world])
        
        p_from_world = frames_data[i]["origin_world"]
        
        w_i_at_common = _wrench_translate(
            wrench=w_i_world_at_frame_origin,
            p_from=p_from_world,
            p_to=p_common
        )
        
        # --- CAMBIO: Guardamos la wrench de este link en el diccionario ---
        individual_wrenches_at_com[link_id] = w_i_at_common
        
        total_wrench_at_com += w_i_at_common

    # --- CAMBIO: Devolvemos ambos resultados ---
    return total_wrench_at_com, individual_wrenches_at_com