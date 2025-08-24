import numpy as np
import pybullet as pb
import math
from config.g1.sim.pybullet.ihwbc.pybullet_params import Config

# ===== CILINDROS DEL ROBOT -SECCIONES =====
class Section:
    def __init__(self, link_idx, p0_local, p1_local, radio):
        self.link_idx = link_idx
        self.p0_local = np.array(p0_local)
        self.p1_local = np.array(p1_local)
        self.radio = radio

# ===== OBJETO DEL AGUA =====
class Water:
    def __init__(self):
        self.v = Config.v_water
        self.h = Config.h_water
        self.rho = Config.rho
        self.theta = Config.theta

        self.v_vec = np.array([self.v*math.cos(self.theta), self.v*math.sin(self.theta), 0.0])

         # Incluir parametros random
    
    # TODO: rutinas que actualicen su velocidad y posicion
    def get_v_vec_h(self, z):
        pass

    def update_params(self): # Las llamadas de esto requeriran parametros de Config, directamente pasados desde alli
        pass

# ===== MANAGER DE FUERZAS
class SectionManager:
    def __init__(self, link_id_dict, shin_params, thigh_params, pelvis_params, interpolator):
            
            self.link_id_dict = link_id_dict
            self.sections = []
            self.interpolator = interpolator

            # ===== Espinilla =====
            shu, shd = shin_params['half_up'], shin_params['half_dn']
            spos     = shin_params['pos_off']
            sr       = shin_params['radius']
            for side in ("left","right"):
                idx = link_id_dict[f"{side}_knee_link"]
                p0 = (spos[0], spos[1], spos[2] - shd)
                p1 = (spos[0], spos[1], spos[2] + shu)
                self.sections.append(Section(idx, p0, p1, sr))

            # ===== Muslo =====
            thu, thd = thigh_params['half_up'], thigh_params['half_dn']
            tpos     = thigh_params['pos_off']
            tr       = thigh_params['radius']
            for side in ("left","right"):
                idx = link_id_dict[f"{side}_hip_yaw_link"]
                y_off = tpos[1] if side=="left" else -tpos[1]
                p0 = (tpos[0], y_off, tpos[2] - thd)
                p1 = (tpos[0], y_off, tpos[2] + thu)
                self.sections.append(Section(idx, p0, p1, tr))

            # ===== Pelvis =====
            pu, pd = pelvis_params['half_up'], pelvis_params['half_dn']
            ppos    = pelvis_params['pos_off']
            pr      = pelvis_params['radius']
            idx     = link_id_dict.get("pelvis", -1)
            p0 = (ppos[0], ppos[1], ppos[2] - pd)
            p1 = (ppos[0], ppos[1], ppos[2] + pu)
            self.sections.append(Section(idx, p0, p1, pr))


    def get_link_frame(self, robot_id, link_idx):
        if link_idx == -1:
            return pb.getBasePositionAndOrientation(robot_id)
        
        # link_idx != -1
        st = pb.getLinkState(robot_id, link_idx, computeForwardKinematics=True) 
        return st[0], st[1]
    
        
    def _get_submerged_properties(self, p0w, p1w, water):
        z0, z1 = p0w[2], p1w[2]

        #  Totalmente fuera del agua
        if z0 >= water.h and z1 >= water.h:
            return 0, None
        
        #  Totalmente sumergido
        if z0 < water.h and z1 < water.h:
            L_sub = np.linalg.norm(p1w - p0w)
            C_sub = (p0w + p1w) /2
            return L_sub, C_sub
    
        #  Parcialmente sumergido
        p_deep, p_shallow = (p0w, p1w) if z0 < z1 else (p1w, p0w)

        interp_frac = (water.h - p_deep[2]) / (p_shallow[2] - p_deep[2])
        p_surface = p_deep + interp_frac*(p_shallow - p_deep)

        L_sub = np.linalg.norm(p_surface - p_deep)
        C_sub = (p_surface + p_deep) / 2
        return L_sub, C_sub
    
    def apply_archimedes(self, robot, robot_com, water, g = 9.81):
        F_arch_total = np.zeros(3)
        tau_arch_total = np.zeros(3)

        for section in self.sections:
            
            lp, lo = self.get_link_frame(robot, section.link_idx)
            p0w, _ = pb.multiplyTransforms(lp, lo, section.p0_local.tolist(), [0,0,0,1])
            p1w, _ = pb.multiplyTransforms(lp, lo, section.p1_local.tolist(), [0,0,0,1])

            p0w, p1w = np.array(p0w), np.array(p1w)

            L_sub, C_sub = self._get_submerged_properties(p0w, p1w, water)

            if L_sub > 0:
                V_sub = math.pi * section.radio * section.radio * L_sub
                F_arch = np.array([0.0, 0.0, water.rho * g * V_sub])
                pb.applyExternalForce(robot, section.link_idx, F_arch, C_sub, pb.WORLD_FRAME)

                F_arch_total += F_arch
                r_vec = C_sub - robot_com
                tau_arch_total += np.cross(r_vec, F_arch)

        return F_arch_total, tau_arch_total
    
    def apply_drag(self, robot, robot_com, water):
        F_drag_total = np.zeros(3)
        tau_drag_total = np.zeros(3)

        return F_drag_total, tau_drag_total
    
    # Podríamos poner una función que calculara las dos y las sumara pero me puede interesar guardarme el log de las dos por separado


    
