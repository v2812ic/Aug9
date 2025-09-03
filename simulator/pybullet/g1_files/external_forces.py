# --------------------------------------------------------
# MANAGER DE FUERZAS OBSOLETO; DEJO LO DE ARQUIMEDES POR CURIOSIDAD PERO NO SE VA A USAR
# --------------------------------------------------------

import numpy as np
import pybullet as pb
import math
import os
import trimesh

from util.python_utils import util
from config.g1.sim.pybullet.ihwbc.pybullet_params import Config
from concurrent.futures import ThreadPoolExecutor

# ===== TEST DE APLICACION DE FUERZAS CONTROLADAS. NO SE CORRESPONDEN A LAS DEL FLUIDO =====

def apply_external_forces(robot, t):

    F = [0, 0, 0]

    if t > 2:

        F = [-10, 0, 0]
        pb.applyExternalForce(robot, -1, F, [0.0, 0.0, 0.0], pb.WORLD_FRAME)
    
    return F


# ===== CLASE DE CADA SECCION DEL ROBOT =====
class Section:

    # === Init: guarda toda la información de la sección cilíndrica que aproxima al link EN LOCAL ===
    def __init__(self, link_idx, p0_local, p1_local, radio):
        self.link_idx = link_idx
        self.p0_local  = np.array(p0_local)
        self.p1_local  = np.array(p1_local)
        self.radio     = radio
        #self.sample_pts = None
        #self.mesh_volume = None

    # === Sample: genera una nube de N puntos en la malla de la sección (Arquímedes -> sample_points -> process shape -> sample)
    def sample(mesh, bbox_min, bbox_max, N):

        samples = []

        while len(samples) < N:
            pt = np.random.rand(1, 3) * (bbox_max - bbox_min) + bbox_min
            if mesh.contains(pt):
                samples.append(pt)

        return np.concatenate(samples)

    # === Process shape: calcula el volumen del link y llama a la función de sampling de puntos (Arquímedes -> sample_points -> process shape)
    def process_shape(self, shape, base_dir, N):
        _, link_idx, geomType, _, mesh_file_raw, local_pos, local_orn, _ = shape
        mesh_file = mesh_file_raw.decode('utf-8') if isinstance(mesh_file_raw, bytes) else mesh_file_raw
        if geomType != pb.GEOM_MESH or not mesh_file:
            return None

        # Resolver posibles rutas al mesh
        candidates = [
            mesh_file,
            os.path.join(os.getcwd(), mesh_file),
            os.path.join(base_dir, mesh_file),
        ]
        path = next((c for c in candidates if isinstance(c, str) and os.path.exists(c)), None)
        if path is None:
            print(f"Warning: mesh not found for link {link_idx}, tried: {candidates}")
            return None

        # Cargar mesh y calcular volumen
        try:
            mesh = trimesh.load(path, force='mesh')
        except Exception as e:
            print(f"Error loading mesh {path}: {e}")
            return None

        volume = mesh.volume
        bbox_min, bbox_max = mesh.bounds

        # Muestreo de puntos
        pts_local = self.sample(mesh, bbox_min, bbox_max, N)
        if pts_local is None:
            print(f"Warning: no interior points sampled for link {link_idx}")
            return None

        return volume, pts_local

    # Ojo a que esto no está bien montado porque esto llama a todas las shapes del robot, no solo a las que nos interesan
    
    # === Sample points: asigna la lista de puntos y volumen al elemento section (Arquímedes -> sample)
    def sample_points(self, robot, urdf_path = "../../robot_model/g1/meshes", N = Config.N_sample):
        if os.path.isdir(urdf_path):
            base_dir = urdf_path
        else:
            base_dir = os.path.dirname(urdf_path)
        shapes = pb.getVisualShapeData(robot)

        # Paralelizar el procesamiento de links
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_shape, shape, base_dir, N) for shape in shapes]
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
        
        for volume, pts_local in results:
            self.mesh_volume = volume
            self.pts_local = pts_local

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



# ===== SUPER CLASE QUE GESTIONA CADA UNA DE LAS SECCIONES DEL ROBOT =====

class SectionManager:
    def __init__(self, link_id_dict, shin_params, thigh_params, pelvis_params):
        self.link_id_dict = link_id_dict
        self.sections = []

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

    def compute_axes_and_angles(self, robot_id, sec, water):

        # extremos en WORLD
        lp, lo = self._get_link_frame(robot_id, sec.link_idx)
        p0w, _ = pb.multiplyTransforms(lp, lo, sec.p0_local.tolist(), [0,0,0,1])
        p1w, _ = pb.multiplyTransforms(lp, lo, sec.p1_local.tolist(), [0,0,0,1])
        p0, p1 = np.array(p0w), np.array(p1w)

        # eje del cilindro
        axis = p1 - p0
        L = np.linalg.norm(axis)
        if L < 1e-6:
            return None, None, None, None
        axis /= L

        # ángulo con la corriente
        vdir = water.v / (np.linalg.norm(water.v) + 1e-12)
        phi_curr = math.acos(np.clip(np.dot(axis, vdir), -1.0, 1.0))

        # ángulo con la vertical Z
        z_axis = np.array([0,0,1])
        phi_vert = math.acos(np.clip(np.dot(axis, z_axis), -1.0, 1.0))

        return p0, p1, axis, phi_curr, phi_vert
    


    # ===== CALCULO DE FUERZAS EXTERNAS (Diferenciamos Arquímedes y Arrastre)
    
    def apply_archimedes(self, robot, water, g = 9.81):

        F_arch = np.zeros(3)

        for section in self.sections:

            if section.link_idx == -1:
                link_pos, link_orn = pb.getBasePositionAndOrientation(robot)
            else:
                link_pos, link_orn = pb.getLinkState(robot, section.link_idx)[0:2]
            
            R_link = util.quat_to_rot(np.array(link_orn))
            T_link = np.eye(4)
            T_link[:3, :3] = R_link
            T_link[:3, 3] = link_pos

            # ===== Pone los puntos del link en mundo para controlar su altura real ======
            pts_world = (R_link @ section.pts_local.T).T + link_pos

            # ===== Filtra los sumergidos ======
            mask_sub = pts_world[:, 2] < water.h
            k_sub = mask_sub.sum()
            if k_sub == 0:
                continue

            # ===== Si hay ppuntos, calcula el centroide sumergido
            V_tot = section.mesh_volume
            frac = k_sub / pts_world.shape[0]

            V_sub = V_tot * frac
            C_sub = pts_world[mask_sub].mean(axis = 0)

            # ===== Fuerza Arquimedes =====
            Fz = water.rho * g * V_sub
            F = np.array([0.0, 0.0, Fz])
            pb.applyExternalForce(robot, section.link_idx, F.tolist(), C_sub.tolist(), pb.WORLD_FRAME)

            F_arch += F

        return F_arch
    
    def compute_cd(phic, phiv):
        # placeholder del coeficiente de arrastre, seguramente ni lo haga así
        return 1.1

    def apply_drag(self, robot, water, cd = compute_cd):

        F_drag = np.zeros(3)

        # TODO: Enchufar aleatoriedad en las fuerzas

        for section in self.sections:

            result = self.compute_axes_and_angles(robot, section, water.v_vec)
            if result[0] is None:
                continue
            p0, p1, axis, phi_c, phi_v = result

            # Velocidad relativa al agua
            lin = np.array(pb.getLinkState(
                robot, section.link_idx,
                computeLinkVelocity=True
            )[6]) if section.link_idx != -1 else pb.getBaseVelocity(robot)[0]
            vrel = water.v_vec- lin
            mag  = np.linalg.norm(vrel)
            if mag < 1e-6:
                continue

            # Fracción sumergida
            


    # ===== APPLY HYDRODYNAMICS: Aplica las dos fuerzas a la vez y devuelve el wrench total para comprobar ======
    
    def apply_hydrodynamics(self, robot, water):

        F_arch = self.apply_archimedes(robot, water)
        F_drag = self.apply_drag(robot, water)

        return F_arch + F_drag

    
    