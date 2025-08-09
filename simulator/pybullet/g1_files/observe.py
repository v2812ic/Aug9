import numpy as np
import pybullet as pb

def observe(robot, model, bullet_to_pino, dt,
                        v_prev=None, rot_prev=None):
    # ---------- base pose -------------------------------------------------
    pos_w, orn_w = pb.getBasePositionAndOrientation(robot)
    quat_xyzw = np.array(orn_w)                 # PyBullet xyzw
    q_pin = np.zeros(model.nq)
    q_pin[:3]  = pos_w                          # x y z   (traslación primero)
    q_pin[3:7] = quat_xyzw                      # qx qy qz qw

    # ---------- velocities ------------------------------------------------
    w_w, v_w = pb.getBaseVelocity(robot)
    R_wb = np.array(pb.getMatrixFromQuaternion(orn_w)).reshape(3,3)
    w_b = R_wb.T @ w_w                          # world → base
    v_b = R_wb.T @ v_w
    v_pin = np.zeros(model.nv)
    v_pin[:6] = np.hstack([w_b, v_b])

    # ---------- joints ----------------------------------------------------
    jstate = pb.getJointStates(robot, bullet_to_pino)
    qj = np.array([s[0] for s in jstate])
    vj = np.array([s[1] for s in jstate])
    try:
        aj = np.array([s[6] for s in jstate])
    except IndexError:
        aj = None

    q_pin[7:]    = qj
    v_pin[6:]    = vj

    # ---------- accelerations --------------------------------------------
    a_pin = np.zeros(model.nv)
    if aj is not None:
        a_pin[6:] = aj                            # joints
    if v_prev is not None:
        a_pin[:6] = (v_pin[:6] - v_prev[:6]) / dt
        if aj is None:
            a_pin[6:] = (v_pin[6:] - v_prev[6:]) / dt
    return q_pin, v_pin, a_pin