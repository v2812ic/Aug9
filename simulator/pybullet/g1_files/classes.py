import numpy as np
import pybullet as pb
import math
from config.g1.sim.pybullet.ihwbc.pybullet_params import Config


# A final de semana tengo que tener cómo funciona el WBC, todo menos el último capítulo, y tendré un mes para redactar y funarme la última parte. No da tiempo xd. Si quiero funarlo en dos semanas...
# Redacción WBC (4H) xd, va a ser mucho más que eso

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
        self.h = Config.h_water
        self.rho = Config.rho
        self.nu = Config.nu

        self.v_base = Config.v_water
        self.theta_base = Config.theta
        self.v_vec_base = np.array([self.v_base*math.cos(self.theta_base), self.v_base*math.sin(self.theta_base), 0.0])

        self.random_amplitude = Config.random_amplitude
        self.random_freq = Config.random_frequency

         # Incluir parametros random
    
    # TODO: rutinas que actualicen su velocidad y posicion
    def get_v_vec(self, z):
        return self.v_vec_base

    def update_params(self): # Las llamadas de esto requeriran parametros de Config, directamente pasados desde alli
        pass

# ===== MANAGER DE FUERZAS ===== 
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
    
        
    def get_submerged_properties(self, p0w, p1w, water):
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
    
    def apply_archimedes(self, robot, robot_com, t, water, g = 9.81):
        F_arch_total = np.zeros(3)
        tau_arch_total = np.zeros(3)

        for section in self.sections:
            
            lp, lo = self.get_link_frame(robot, section.link_idx)
            p0w, _ = pb.multiplyTransforms(lp, lo, section.p0_local.tolist(), [0,0,0,1])
            p1w, _ = pb.multiplyTransforms(lp, lo, section.p1_local.tolist(), [0,0,0,1])

            p0w, p1w = np.array(p0w), np.array(p1w)

            L_sub, C_sub = self.get_submerged_properties(p0w, p1w, water)

            if L_sub > 0:
                V_sub = math.pi * section.radio * section.radio * L_sub
                F_arch = np.array([0.0, 0.0, water.rho * g * V_sub]) * (1 - 2/(1 + math.exp((t-Config.initForce)*Config.smooth_forces)))
                print((1 - 2/(1 + math.exp((t-Config.initForce)*Config.smooth_forces))))
                pb.applyExternalForce(robot, section.link_idx, F_arch, C_sub, pb.WORLD_FRAME)

                F_arch_total += F_arch
                r_vec = C_sub - robot_com
                tau_arch_total += np.cross(r_vec, F_arch)

        return F_arch_total, tau_arch_total
    
    def apply_drag(self, robot, robot_com, t, water, num_slices=1):

        F_drag_total = np.zeros(3)
        tau_drag_total = np.zeros(3)

        for section in self.sections:
            # --- OBTENER GEOMETRÍA DEL LINK Y PARTE SUMERGIDA ---
            lp, lo = self.get_link_frame(robot, section.link_idx)
            p0w, _ = pb.multiplyTransforms(lp, lo, section.p0_local.tolist(), [0,0,0,1])
            p1w, _ = pb.multiplyTransforms(lp, lo, section.p1_local.tolist(), [0,0,0,1])
            p0w, p1w = np.array(p0w), np.array(p1w)

            # Determinar los extremos del segmento sumergido. No es necesario el C_sub para las slices asi que omitimos el uso de la otra función
            p_deep, p_shallow = (p0w, p1w) if p0w[2] < p1w[2] else (p1w, p0w)
            if p_shallow[2] < 0: continue # Totalmente fuera del agua 
            
            h_water = water.h
            submerged_start = p_deep
            
            if p_deep[2] >= h_water: continue # Totalmente fuera del agua
            
            submerged_end = p_shallow
            if p_shallow[2] > h_water:
                interp_frac = (h_water - p_deep[2]) / (p_shallow[2] - p_deep[2])
                submerged_end = p_deep + interp_frac * (p_shallow - p_deep)
            
            submerged_axis = submerged_end - submerged_start
            L_sub = np.linalg.norm(submerged_axis)
            if L_sub < 1e-6: continue
            
            # --- BUCLE DE INTEGRACIÓN POR LONCHAS ---
            slice_length = L_sub / num_slices
            slice_axis = submerged_axis / L_sub

            for i in range(num_slices):
                # Calcular el centro de la loncha actual
                slice_center = submerged_start + slice_axis * (slice_length * (i + 0.5))
                z = slice_center[2]

                # OBTENER VELOCIDADES LOCALES
                # Asumimos que la velocidad lineal del link es constante en toda la sección
                if section.link_idx == -1: lin_vel, _ = pb.getBaseVelocity(robot)
                else: lin_vel = pb.getLinkState(robot, section.link_idx, computeLinkVelocity=True)[6]
                
                v_water_local = water.get_v_vec(z)
                v_rel = v_water_local - np.array(lin_vel)
                
                # CALCULAR FUERZA DE ARRASTRE PARA ESTA LONCHA
                axis = p1w - p0w # Eje del cilindro completo, no de la loncha
                axis /= np.linalg.norm(axis)

                v_parallel = np.dot(v_rel, axis) * axis
                v_normal = v_rel - v_parallel
                v_normal_mag = np.linalg.norm(v_normal)

                if v_normal_mag < 1e-6: continue

                # Parámetros para el interpolador
                phi_c_rad = np.arccos(np.clip(np.dot(axis, v_rel / np.linalg.norm(v_rel)), -1.0, 1.0))
                yaw_angle_deg = 90.0 - np.rad2deg(phi_c_rad)
                Re_normal = v_normal_mag * (2 * section.radio) / water.nu
                
                cd_normal = self.interpolator.get_cd_from_distribution(abs(yaw_angle_deg), Re_normal)

                F_drag_slice = cd_normal * 0.5 * (section.radio * 2) * water.rho * v_normal * v_normal_mag * slice_length
                
                # APLICAR LA FUERZA DE LA LONCHA Y ACUMULAR EL WRENCH
                if np.linalg.norm(F_drag_slice) > 0:
                    F_drag_slice = F_drag_slice * (1 - 2/(1 + math.exp((t-Config.initForce)*Config.smooth_forces)))
                    
                    pb.applyExternalForce(robot, section.link_idx, F_drag_slice, slice_center, pb.WORLD_FRAME)
                    
                    F_drag_total += F_drag_slice
                    r_vec = slice_center - robot_com
                    tau_drag_total += np.cross(r_vec, F_drag_slice)

        return F_drag_total, tau_drag_total


    
    # Podríamos poner una función que calculara las dos y las sumara pero me puede interesar guardarme el log de las dos por separado


    
