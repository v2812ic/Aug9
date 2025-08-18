# sim_main.py
import os
import sys
import shutil
import numpy as np
import pybullet as pb
import cv2
import pinocchio as pin
import yaml

# Rutas (para pybind y utilidades del proyecto)
cwd = os.getcwd()
sys.path.extend([cwd, cwd + "/build/lib"])
urdf_path = cwd + "/robot_model/g1/g1_29dof_lock_waist.urdf"

from config.g1.sim.pybullet.ihwbc.pybullet_params import Config
from util.python_utils import pybullet_util
import g1_interface_py

# ===== Import del coeficiente de rozamiento
yaml_path = cwd + "/config/g1/sim/pybullet/ihwbc/pnc.yaml"
with open(yaml_path, 'r') as f:
    config_yaml = yaml.safe_load(f)
mu_ = config_yaml["wbc"]["contact"]["mu"]
mu_ = 0.5

# ===== Segmentación del main =====
from g1_files.sensors import get_sensor_data_from_pybullet, compute_base_joint_debug
from g1_files.control import apply_control_input_to_pybullet, handle_keyboard_events
from g1_files.init_pose import set_init_config_pybullet_robot
from g1_files.cleanup import install_signal_handler

# ====== Observador ======
from g1_files.observe import observe
from g1_files.contact import get_contact_wrenches
from g1_files.tau_ext import get_tau_ext
from g1_files.solver_forces import solve_force
from g1_files.forces_to_world import group_tripod

# ====== Plots (se generan en Ctrl+C) ======
from g1_files.plots import make_plots

# Aplicación de fuerzas externas transitorias
from g1_files.external_forces import apply_external_forces

# Mirar el uso de actuadores
from g1_files.printTorques import printTorques, plotTorques
flag_torque_limit = False


def main():
    # ====== Visualización / Sim ======
    pb.connect(pb.GUI)
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.resetDebugVisualizerCamera(
        cameraDistance=1.5, cameraYaw=120, cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5],
    )
    pb.setPhysicsEngineParameter(
        fixedTimeStep=Config.CONTROLLER_DT, numSubSteps=Config.N_SUBSTEP
    )
    pb.setGravity(0, 0, -9.81)

    # ====== Carga modelos ======
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
    g1_humanoid = pb.loadURDF(
        urdf_path,
        Config.INITIAL_BASE_JOINT_POS, Config.INITIAL_BASE_JOINT_QUAT,
        useFixedBase=0,
    )
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()

    num_joints = pb.getNumJoints(g1_humanoid)
    pino_joint_names = model.names[2:]  # salteamos universe y root
    name2idx_bullet = {pb.getJointInfo(g1_humanoid, i)[1].decode(): i for i in range(num_joints)}
    bullet_to_pino = [name2idx_bullet[n] for n in pino_joint_names]

    _ground = pb.loadURDF(cwd + "/robot_model/ground/plane.urdf", useFixedBase=1)
    pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    # ====== Lateral friction coefficients ======

    pb.changeDynamics(g1_humanoid, 6, lateralFriction = mu_) #pie izquierdo
    pb.changeDynamics(g1_humanoid, 13, lateralFriction = mu_) #derecho
    #pb.changeDynamics(_ground, -1, lateralFriction=1) # setada a 1 para que mutot = 1 *mu, ya está por defecto

    for i in range(pb.getNumJoints(g1_humanoid)):
        link_name = pb.getJointInfo(g1_humanoid, i)[12].decode('UTF-8')
        dynamics_info = pb.getDynamicsInfo(g1_humanoid, i)
        lateral_friction = dynamics_info[1]
        print(f"Robot Link '{link_name}' (ID: {i}): Friction coefficient = {lateral_friction}")

    ground_dynamics_info = pb.getDynamicsInfo(_ground, -1)
    ground_lateral_friction = ground_dynamics_info[1]
    print(f"\nGround: Friction coefficient = {ground_lateral_friction}")
    print("=======================================\n")

    # ====== Config robot ======
    (
        n_q, n_v, n_a, joint_id_dict, link_id_dict,
        pos_basejoint_to_basecom, rot_basejoint_to_basecom,
    ) = pybullet_util.get_robot_config(
        g1_humanoid,
        Config.INITIAL_BASE_JOINT_POS,
        Config.INITIAL_BASE_JOINT_QUAT,
        Config.PRINT_ROBOT_INFO,
    )
    set_init_config_pybullet_robot(g1_humanoid)
    pybullet_util.set_joint_friction(g1_humanoid, joint_id_dict, 0)
    pybullet_util.set_link_damping(g1_humanoid, link_id_dict, 0.0, 0.0)

    # ====== Interfaces RPC ======
    rpc_g1_interface   = g1_interface_py.G1Interface()
    rpc_g1_sensor_data = g1_interface_py.G1SensorData()
    rpc_g1_command     = g1_interface_py.G1Command()

    # ====== Opciones de simulación ======
    dt = Config.CONTROLLER_DT
    count = 0
    jpg_count = 0

    if Config.MEASURE_COMPUTATION_TIME:
        from pytictoc import TicToc
        timer = TicToc()
        compuation_cal_list = []
    else:
        compuation_cal_list = None

    if Config.VIDEO_RECORD:
        video_dir = "video/g1"
        shutil.rmtree(video_dir, ignore_errors=True)
        os.makedirs(video_dir)
    else:
        video_dir = None

    # ====== Logs y estado ======
    previous_torso_velocity = np.array([0.0, 0.0, 0.0])

    l_contact_volt_noise = 0.001
    r_contact_volt_noise = 0.001
    imu_dvel_bias = np.array([0.0, 0.0, 0.0])
    imu_ang_vel_noise_std_dev = 0.0

    # Si necesitas “quitar” masas de algunos links
    for link_name in ("imu_in_torso", "imu_in_pelvis", "d435_link", "mid360_link"):
        if link_name in link_id_dict:
            pb.changeDynamics(g1_humanoid, link_id_dict[link_name],
                              mass=0.0, localInertiaDiagonal=[0.0, 0.0, 0.0])
            
    

    # ====== Init de variables ======
    v_prev = None
    tau_ext = np.zeros(model.nv)
    tau_j = np.zeros(model.nv)
    f_ext = None

    # ====== Logs configurables ======
    time_log = []
    tau_ext_log = []    
    cf_left_log = []
    cf_right_log = []

    pos_left_log = []
    ori_left_log = []
    pos_right_log = []
    ori_right_log = []



    # ====== Finalización/plots (para Ctrl+C) ======
    def finalize():
        try:
            # Pasá SOLO lo que quieras graficar; lo que no pases, no se dibuja.
            make_plots(
                outdir="plots",
                time=time_log,
                tau_ext=tau_ext_log,
                cf_left_log = cf_left_log,
                cf_right_log = cf_right_log,
                pos_left_log = pos_left_log,
                pos_right_log = pos_right_log,
                ori_left_log = ori_left_log,
                ori_right_log = ori_right_log,
            )
            plotTorques()
            
        except Exception as e:
            print(f"[finalize] Error generando plots: {e}")

    install_signal_handler(
        Config, pybullet_util,
        compuation_cal_list, video_dir,
        on_exit=finalize  
    )

    # ====== Bucle principal ======
    while count*dt < Config.endSimulation or not Config.endSimulation:
        # --- Depuración: estados "ground truth" del base joint ---
        (base_joint_pos, base_joint_quat,
         base_joint_lin_vel, base_joint_ang_vel) = compute_base_joint_debug(
            g1_humanoid, rot_basejoint_to_basecom, pos_basejoint_to_basecom
        )
        rpc_g1_sensor_data.base_joint_pos_      = base_joint_pos
        rpc_g1_sensor_data.base_joint_quat_     = base_joint_quat
        rpc_g1_sensor_data.base_joint_lin_vel_  = base_joint_lin_vel
        rpc_g1_sensor_data.base_joint_ang_vel_  = base_joint_ang_vel

        # --- Teclado (interrupciones mapeadas si las usas) ---
        keys = pb.getKeyboardEvents()
        handle_keyboard_events(keys, rpc_g1_interface, pybullet_util)

        # --- Lectura de sensores ---
        (imu_frame_quat, imu_ang_vel, imu_dvel, joint_pos, joint_vel,
         b_lf_contact, b_rf_contact, l_normal_force, r_normal_force) = \
            get_sensor_data_from_pybullet(g1_humanoid, previous_torso_velocity)

        # --- Muestras de ruido ---
        l_normal_volt_noise = np.random.normal(0, l_contact_volt_noise)
        r_normal_volt_noise = np.random.normal(0, r_contact_volt_noise)
        imu_ang_vel_noise   = np.random.normal(0, imu_ang_vel_noise_std_dev)

        # --- Post-proceso sensores ---
        l_normal_force = pybullet_util.simulate_contact_sensor(l_normal_force)
        r_normal_force = pybullet_util.simulate_contact_sensor(r_normal_force)
        imu_dvel       = pybullet_util.add_sensor_noise(imu_dvel, imu_dvel_bias)
        imu_ang_vel    = pybullet_util.add_sensor_noise(imu_ang_vel, imu_ang_vel_noise)
        l_normal_force = pybullet_util.add_sensor_noise(l_normal_force, l_normal_volt_noise)
        r_normal_force = pybullet_util.add_sensor_noise(r_normal_force, r_normal_volt_noise)

        # --- Poblar estructura RPC ---
        rpc_g1_sensor_data.imu_frame_quat_     = imu_frame_quat
        rpc_g1_sensor_data.imu_ang_vel_        = imu_ang_vel
        rpc_g1_sensor_data.imu_dvel_           = imu_dvel
        rpc_g1_sensor_data.imu_lin_acc_        = imu_dvel / dt
        rpc_g1_sensor_data.joint_pos_          = joint_pos
        rpc_g1_sensor_data.joint_vel_          = joint_vel
        rpc_g1_sensor_data.b_lf_contact_       = b_lf_contact
        rpc_g1_sensor_data.b_rf_contact_       = b_rf_contact
        rpc_g1_sensor_data.lf_contact_normal_  = l_normal_force
        rpc_g1_sensor_data.rf_contact_normal_  = r_normal_force

        # --- Control ---
        if Config.MEASURE_COMPUTATION_TIME:
            timer.tic()
        rpc_g1_interface.GetCommand(rpc_g1_sensor_data, rpc_g1_command)
        if Config.MEASURE_COMPUTATION_TIME:
            compuation_cal_list.append(timer.tocvalue())

        rpc_trq_command = rpc_g1_command.joint_trq_cmd_
        apply_control_input_to_pybullet(g1_humanoid, rpc_trq_command)

        # --- Guardar vel. torso para siguiente dVel ---
        previous_torso_velocity = pybullet_util.get_link_vel(
            g1_humanoid, link_id_dict["imu_in_torso"]
        )[3:6]

        # --- Grabación opcional ---
        if Config.VIDEO_RECORD and (count % Config.RECORD_FREQ == 0):
            camera_data = pb.getDebugVisualizerCamera()
            frame = pybullet_util.get_camera_image_from_debug_camera(
                camera_data, Config.RENDER_WIDTH, Config.RENDER_HEIGHT
            )
            cv2.imwrite(f"{video_dir}/step{jpg_count:06d}.jpg", frame)
            jpg_count += 1

        apply_external_forces(g1_humanoid, count*dt)

        # --- Observador ---
        
        q_pin, v_pin, a_pin = observe(g1_humanoid, model, bullet_to_pino, dt, v_prev)
        tau_c, tau_cf, cf_left, cf_right = get_contact_wrenches(g1_humanoid, _ground, model, data, q_pin)
        v_prev = v_pin.copy()

        tau_j[:] = 0.0
        tau_j[6:] = rpc_trq_command  # asigna torques actuados en el segmento articular

        tau_ext = get_tau_ext(dt, model, data, q_pin, v_pin, tau_c, tau_j)

        #print(f"{tau_ext[:6]}")

        # --- Solver de fuerzas en las piernas
        if not count % Config.solver_frequency:
            com_W = pin.centerOfMass(model, data, q_pin)
            f_ext_local = solve_force(Config.rglrztn_forces, Config.link_idx_vec, tau_ext, g1_humanoid, model, data, q_pin, Config.pinv_forces, Config.printForces)
            #print("globals: ", f_ext_local)
            f_ext = group_tripod(f_ext_local, g1_humanoid, Config.link_idx_vec, Config.left_ids, Config.right_ids, Config.pelvis_ids, ref_W = com_W)

           
            rpc_g1_interface.set_external_force(f_ext)
                
        # Se la pasamos a C++ cuando ha empezado
        if Config.InitObservations < count*dt:
            rpc_g1_interface.set_external_torque(tau_ext)
            rpc_g1_interface.set_external_force(f_ext)
            
        # Informacion varia de la posicion y orientacion del pie
        pos_left, ori_left = pb.getLinkState(g1_humanoid, 6)[0], pb.getLinkState(g1_humanoid, 6)[1]
        pos_right, ori_right = pb.getLinkState(g1_humanoid, 13)[0], pb.getLinkState(g1_humanoid, 13)[1]

        # --- Logs ---
        time_log.append(count * dt)
        tau_ext_log.append(np.array(tau_ext))
        cf_left_log.append(cf_left)
        cf_right_log.append(cf_right)
        
        pos_left_log.append(pos_left)
        ori_left_log.append(ori_left)
        pos_right_log.append(pos_right)
        ori_right_log.append(ori_right)

        # --- Step ---
        pb.stepSimulation()
        count += 1

        if Config.plotTorques:
            printTorques(g1_humanoid, count*dt, rpc_trq_command, flag_torque_limit, verbose = False)
    
    finalize()
        

if __name__ == "__main__":
    main()
