# --------------------------------------------------------
# MANAGER DE FUERZAS OBSOLETO; DEJO LO DE ARQUIMEDES POR CURIOSIDAD PERO NO SE VA A USAR
# --------------------------------------------------------

import numpy as np
import pybullet as pb
import math
import os
import trimesh
import pinocchio as pin
from util.python_utils import util
from config.g1.sim.pybullet.ihwbc.pybullet_params import Config
from concurrent.futures import ThreadPoolExecutor

# ===== TEST DE APLICACION DE FUERZAS CONTROLADAS. NO SE CORRESPONDEN A LAS DEL FLUIDO =====

# -----------------------------
# Helpers: Rotations & Frames
# -----------------------------
def _quat_rotate(q_xyzw, v):
    """Rota el vector v por el cuaternión q (xyzw). Devuelve R(q)*v."""
    R = np.array(pb.getMatrixFromQuaternion(q_xyzw)).reshape(3, 3)
    return R @ np.asarray(v, dtype=float)

def get_link_pose_world(robot, link_idx):
    """
    Devuelve (pos, orn) del frame del link en WORLD.
      - link_idx == -1: base flotante (root)
      - link_idx >= 0 : link articulado (getLinkState)
    """
    if link_idx == -1:
        return pb.getBasePositionAndOrientation(robot)
    ls = pb.getLinkState(robot, link_idx, computeForwardKinematics=True)
    # linkWorldPosition y linkWorldOrientation (frame del link)
    return ls[4], ls[5]

def get_com_world(robot, link_idx):
    """
    Devuelve la posición del CoM en WORLD del link/base.
      - link_idx == -1: base flotante (usa inertialPos local de la base)
      - link_idx >= 0 : ls[0] ya es el CoM mundial del link
    """
    if link_idx == -1:
        base_pos, base_orn = get_link_pose_world(robot, -1)
        local_inertial_pos = pb.getDynamicsInfo(robot, -1)[3]  # CoM local en frame de la base
        return np.asarray(base_pos) + _quat_rotate(base_orn, local_inertial_pos)
    ls = pb.getLinkState(robot, link_idx, computeForwardKinematics=True)
    return np.asarray(ls[0])  # linkWorldPosition (CoM)

def get_com_local_in_link_frame(robot, link_idx):
    """Posición local del CoM en el frame del link/base (inertial pos)."""
    return np.asarray(pb.getDynamicsInfo(robot, link_idx)[3], dtype=float)

def link_local_point_to_world(robot, link_idx, p_local):
    """Convierte un punto dado en el frame del link a coordenadas WORLD."""
    p_local = np.asarray(p_local, dtype=float)
    pos_w, orn_w = get_link_pose_world(robot, link_idx)
    return np.asarray(pos_w) + _quat_rotate(orn_w, p_local)

# -----------------------------
# Wrench mapping to CoM
# -----------------------------
def wrench_about_com_from_application_point(robot, link_idx, F_world, M_world, p_app_world):
    """
    Dado F y M aplicados en p_app_world, devuelve el wrench equivalente en el CoM:
      M_com = M_world + (p_app_world - p_com_world) x F_world
    """
    F = np.asarray(F_world, dtype=float)
    M = np.asarray(M_world, dtype=float)
    p_app = np.asarray(p_app_world, dtype=float)
    p_com = get_com_world(robot, link_idx)
    r = p_app - p_com
    M_com = M + np.cross(r, F)
    return F, M_com

# -----------------------------
# Aplicación del wrench

def apply_wrench(robot, link_idx, F_world, M_world, p_app, frame="WORLD"):
    """
    Aplica F y M (siempre en WORLD) en el punto p_app (interpretado en 'frame').
    Esta versión es robusta y corrige la inconsistencia de los flags de PyBullet.
    """
    F_world_list = np.asarray(F_world, float).tolist()
    M_world_list = np.asarray(M_world, float).tolist()
    p_app_list = np.asarray(p_app, float).tolist()

    # PyBullet interpreta el vector de FUERZA y TORQUE según el último flag.
    # La convención es pasarle vectores en WORLD y dejar que él haga la
    # transformación si el PUNTO DE APLICACIÓN es local.
    
    if frame.upper() == "WORLD":
        # Vector de fuerza en WORLD, punto de aplicación en WORLD.
        pb.applyExternalForce(robot, link_idx, F_world_list, p_app_list, pb.WORLD_FRAME)
        pb.applyExternalTorque(robot, link_idx, M_world_list, pb.WORLD_FRAME)
    
    elif frame.upper() == "LINK":
        # A PyBullet se le pasa el PUNTO en LINK, pero el VECTOR de fuerza sigue
        # siendo en WORLD. Es una peculiaridad de la API. Para aplicar una
        # fuerza local, tendríamos que rotarla primero.
        # Para evitar ambigüedad, forzamos que la fuerza siempre sea WORLD.
        
        # Obtenemos la pose del link para convertir el punto local a global.
        pos_w, orn_w = get_link_pose_world(robot, link_idx)
        p_app_world = np.asarray(pos_w) + _quat_rotate(orn_w, p_app)
        
        # Aplicamos la fuerza WORLD en el punto ya convertido a WORLD.
        pb.applyExternalForce(robot, link_idx, F_world_list, p_app_world.tolist(), pb.WORLD_FRAME)
        pb.applyExternalTorque(robot, link_idx, M_world_list, pb.WORLD_FRAME)
    else:
        raise ValueError("frame must be 'WORLD' or 'LINK'")


# -----------------------------
# API principal para tu sim
# -----------------------------
def apply_external_forces(
    robot, t, *,
    link_idx=3,
    F_target=np.array([0.0, 25.0, 0.0]),
    M_target=np.array([0.0, 0.0, 0.0]),
    p_app=None,
    p_app_frame="WORLD",
    t_on=3.0,
    ramp=10,
    # NUEVO: para referir todo al CoM GLOBAL
    model=None, data=None, q_pin=None,
    com = None,
    return_wrench_about="COM_GLOBAL",
):
    """
    Aplica la fuerza/par en p_app y devuelve SIEMPRE (F, M) respecto al CoM GLOBAL.
    """
    F_target = np.asarray(F_target, dtype=float)
    M_target = np.asarray(M_target, dtype=float)
    F = np.zeros(3); M = np.zeros(3)

    if t > t_on:
        s = (2.0/np.pi)*np.arctan(ramp*(t - t_on))
        F = F_target * s
        M = M_target * s

        #print(F.transpose(), " N")

        # p_app por defecto
        if p_app is None:
            if p_app_frame.upper() == "LINK":
                p_app = np.zeros(3)
            else:
                p_app = np.asarray(get_link_pose_world(robot, link_idx)[0], dtype=float)

        # Aplica en simulación
        apply_wrench(robot, link_idx, F_world=F, M_world=M, p_app=p_app, frame=p_app_frame)

        # p_app en WORLD
        p_app_world = (link_local_point_to_world(robot, link_idx, p_app)
                       if p_app_frame.upper() == "LINK" else np.asarray(p_app, float))

        # === NUEVO: Momento respecto al CoM GLOBAL ===
        # Asegura cinemática al día
        
        if com is not None:
            pin.forwardKinematics(model, data, q_pin)
            pin.updateFramePlacements(model, data)
            p_com_global = np.asarray(pin.centerOfMass(model, data, q_pin)).reshape(3,)
        else:
            p_com_global = com
    
        M_global = M + np.cross(p_app_world - p_com_global, F)

        return np.concatenate([F, M_global])

    # Antes de t_on: todo cero
    return np.concatenate([F, M])


    