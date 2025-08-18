import numpy as np
import pybullet as pb

def group_tripod(W_world, body_id, link_idx_vec,
                       left_ids, right_ids, pelvis_ids, ref_W=None):
    """
    W_world: Nx6 en MUNDO (Fx,Fy,Fz,Mx,My,Mz) por link (puede incluir -1 para la base).
    ref_W: punto (3,) en mundo al que quieres referir los momentos (p.ej. CoM). Si None, no traslada.
    Devuelve 3x6 (L, R, Pelvis) en mundo, con momentos referidos a ref_W si se da.
    """
    idx_map = {link_idx_vec[i]: i for i in range(len(link_idx_vec))}

    def link_pos_world(link_idx):
        if link_idx == -1:
            pW, _ = pb.getBasePositionAndOrientation(body_id)
            return np.array(pW)
        else:
            ls = pb.getLinkState(body_id, link_idx, computeForwardKinematics=1)
            return np.array(ls[4])  # worldLinkFramePosition

    def sum_group(ids):
        Fsum = np.zeros(3)
        Msum = np.zeros(3)
        for lid in ids:
            if lid not in idx_map:
                continue
            r = idx_map[lid]
            Fw = W_world[r, :3]          # ya en mundo
            Mw = W_world[r, 3:6]         # ya en mundo, pero ojo al punto de referencia
            if ref_W is not None:
                pW = link_pos_world(lid)  # origen del frame del link en mundo
                Mw = Mw + np.cross(pW - ref_W, Fw)  # trasladar momento al ref_W
            Fsum += Fw
            Msum += Mw
        return np.hstack([Fsum, Msum])

    WL = sum_group(left_ids)
    WR = sum_group(right_ids)
    WP = sum_group(pelvis_ids)
    return np.vstack([WL, WR, WP])