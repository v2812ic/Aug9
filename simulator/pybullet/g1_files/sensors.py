# sensors.py
import numpy as np
import pybullet as pb
from util.python_utils import util, liegroup
from config.g1.sim.pybullet.ihwbc.pybullet_params import G1JointIdx, G1LinkIdx
from util.python_utils import pybullet_util


def get_sensor_data_from_pybullet(robot, previous_torso_velocity):
    """
    Lee estados del robot en PyBullet y devuelve un paquete de sensores.
    Retorna:
      (imu_quat, imu_ang_vel, imu_dvel, joint_pos[37], joint_vel[37],
       b_lf_contact, b_rf_contact, l_normal_force, r_normal_force)
    """
    joint_pos, joint_vel = np.zeros(37), np.zeros(37)

    # IMU
    imu_state = pb.getLinkState(robot, G1LinkIdx.imu_in_torso, 1, 1)
    imu_frame_quat = np.array(imu_state[1])   # orientación (quat)
    imu_ang_vel    = np.array(imu_state[7])   # vel. angular (linkState[7])

    # dVel simulado (usa vel. previa del torso)
    imu_dvel = pybullet_util.simulate_dVel_data(
        robot, G1LinkIdx.imu_in_torso, previous_torso_velocity
    )

    # ===== LF =====
    joint_pos[0] = pb.getJointState(robot, G1JointIdx.left_hip_pitch_joint)[0]
    joint_pos[1] = pb.getJointState(robot, G1JointIdx.left_hip_roll_joint)[0]
    joint_pos[2] = pb.getJointState(robot, G1JointIdx.left_hip_yaw_joint)[0]
    joint_pos[3] = pb.getJointState(robot, G1JointIdx.left_knee_joint)[0]
    joint_pos[4] = pb.getJointState(robot, G1JointIdx.left_ankle_pitch_joint)[0]
    joint_pos[5] = pb.getJointState(robot, G1JointIdx.left_ankle_roll_joint)[0]
    # ===== RF =====
    joint_pos[6]  = pb.getJointState(robot, G1JointIdx.right_hip_pitch_joint)[0]
    joint_pos[7]  = pb.getJointState(robot, G1JointIdx.right_hip_roll_joint)[0]
    joint_pos[8]  = pb.getJointState(robot, G1JointIdx.right_hip_yaw_joint)[0]
    joint_pos[9]  = pb.getJointState(robot, G1JointIdx.right_knee_joint)[0]
    joint_pos[10] = pb.getJointState(robot, G1JointIdx.right_ankle_pitch_joint)[0]
    joint_pos[11] = pb.getJointState(robot, G1JointIdx.right_ankle_roll_joint)[0]
    # ===== torso =====
    joint_pos[12] = pb.getJointState(robot, G1JointIdx.waist_yaw_joint)[0]
    # ===== LH =====
    joint_pos[13] = pb.getJointState(robot, G1JointIdx.left_shoulder_pitch_joint)[0]
    joint_pos[14] = pb.getJointState(robot, G1JointIdx.left_shoulder_roll_joint)[0]
    joint_pos[15] = pb.getJointState(robot, G1JointIdx.left_shoulder_yaw_joint)[0]
    joint_pos[16] = pb.getJointState(robot, G1JointIdx.left_elbow_joint)[0]
    joint_pos[17] = pb.getJointState(robot, G1JointIdx.left_wrist_roll_joint)[0]
    joint_pos[18] = pb.getJointState(robot, G1JointIdx.left_wrist_pitch_joint)[0]
    joint_pos[19] = pb.getJointState(robot, G1JointIdx.left_wrist_yaw_joint)[0]
    # ===== RH =====
    joint_pos[20] = pb.getJointState(robot, G1JointIdx.right_shoulder_pitch_joint)[0]
    joint_pos[21] = pb.getJointState(robot, G1JointIdx.right_shoulder_roll_joint)[0]
    joint_pos[22] = pb.getJointState(robot, G1JointIdx.right_shoulder_yaw_joint)[0]
    joint_pos[23] = pb.getJointState(robot, G1JointIdx.right_elbow_joint)[0]
    joint_pos[24] = pb.getJointState(robot, G1JointIdx.right_wrist_roll_joint)[0]
    joint_pos[25] = pb.getJointState(robot, G1JointIdx.right_wrist_pitch_joint)[0]
    joint_pos[26] = pb.getJointState(robot, G1JointIdx.right_wrist_yaw_joint)[0]

    # Velocidades:
    # ===== LF =====
    joint_vel[0] = pb.getJointState(robot, G1JointIdx.left_hip_pitch_joint)[1]
    joint_vel[1] = pb.getJointState(robot, G1JointIdx.left_hip_roll_joint)[1]
    joint_vel[2] = pb.getJointState(robot, G1JointIdx.left_hip_yaw_joint)[1]
    joint_vel[3] = pb.getJointState(robot, G1JointIdx.left_knee_joint)[1]
    joint_vel[4] = pb.getJointState(robot, G1JointIdx.left_ankle_pitch_joint)[1]
    joint_vel[5] = pb.getJointState(robot, G1JointIdx.left_ankle_roll_joint)[1]
    # ===== RF =====
    joint_vel[6]  = pb.getJointState(robot, G1JointIdx.right_hip_pitch_joint)[1]
    joint_vel[7]  = pb.getJointState(robot, G1JointIdx.right_hip_roll_joint)[1]
    joint_vel[8]  = pb.getJointState(robot, G1JointIdx.right_hip_yaw_joint)[1]
    joint_vel[9]  = pb.getJointState(robot, G1JointIdx.right_knee_joint)[1]
    joint_vel[10] = pb.getJointState(robot, G1JointIdx.right_ankle_pitch_joint)[1]
    joint_vel[11] = pb.getJointState(robot, G1JointIdx.right_ankle_roll_joint)[1]
    # ===== torso =====
    joint_vel[12] = pb.getJointState(robot, G1JointIdx.waist_yaw_joint)[1]
    # ===== LH =====
    joint_vel[13] = pb.getJointState(robot, G1JointIdx.left_shoulder_pitch_joint)[1]
    joint_vel[14] = pb.getJointState(robot, G1JointIdx.left_shoulder_roll_joint)[1]
    joint_vel[15] = pb.getJointState(robot, G1JointIdx.left_shoulder_yaw_joint)[1]
    joint_vel[16] = pb.getJointState(robot, G1JointIdx.left_elbow_joint)[1]
    joint_vel[17] = pb.getJointState(robot, G1JointIdx.left_wrist_roll_joint)[1]
    joint_vel[18] = pb.getJointState(robot, G1JointIdx.left_wrist_pitch_joint)[1]
    joint_vel[19] = pb.getJointState(robot, G1JointIdx.left_wrist_yaw_joint)[1]
    # ===== RH =====
    joint_vel[20] = pb.getJointState(robot, G1JointIdx.right_shoulder_pitch_joint)[1]
    joint_vel[21] = pb.getJointState(robot, G1JointIdx.right_shoulder_roll_joint)[1]
    joint_vel[22] = pb.getJointState(robot, G1JointIdx.right_shoulder_yaw_joint)[1]
    joint_vel[23] = pb.getJointState(robot, G1JointIdx.right_elbow_joint)[1]
    joint_vel[24] = pb.getJointState(robot, G1JointIdx.right_wrist_roll_joint)[1]
    joint_vel[25] = pb.getJointState(robot, G1JointIdx.right_wrist_pitch_joint)[1]
    joint_vel[26] = pb.getJointState(robot, G1JointIdx.right_wrist_yaw_joint)[1]

    # Fuerzas normales de contacto acumuladas
    l_contacts = pb.getContactPoints(bodyA=robot, linkIndexA=G1LinkIdx.l_foot_contact)
    r_contacts = pb.getContactPoints(bodyA=robot, linkIndexA=G1LinkIdx.r_foot_contact)
    _l_normal_force = sum(c[9] for c in l_contacts)
    _r_normal_force = sum(c[9] for c in r_contacts)

    # Flags de contacto (altura del link)
    lz = pb.getLinkState(robot, G1LinkIdx.l_foot_contact, 1, 1)[0][2]
    rz = pb.getLinkState(robot, G1LinkIdx.r_foot_contact, 1, 1)[0][2]
    b_lf_contact = (lz <= 0.01)
    b_rf_contact = (rz <= 0.01)

    return (
        imu_frame_quat, imu_ang_vel, imu_dvel,
        joint_pos, joint_vel,
        b_lf_contact, b_rf_contact,
        _l_normal_force, _r_normal_force,
    )


def compute_base_joint_debug(robot, rot_basejoint_to_basecom, pos_basejoint_to_basecom):
    """
    Calcula estados "ground truth" del base joint (pos, quat, vel lin/ang)
    para depurar el estimador, usando cinemática de la base y transformaciones.
    """
    base_com_pos, base_com_quat = pb.getBasePositionAndOrientation(robot)
    base_com_quat = np.array(base_com_quat)
    rot_world_basecom = util.quat_to_rot(base_com_quat)
    rot_world_basejoint = np.dot(
        rot_world_basecom, rot_basejoint_to_basecom.transpose()
    )
    base_joint_pos = base_com_pos - np.dot(
        rot_world_basejoint, pos_basejoint_to_basecom
    )
    base_joint_quat = util.rot_to_quat(rot_world_basejoint)

    base_com_lin_vel, base_com_ang_vel = pb.getBaseVelocity(robot)

    trans_joint_com = liegroup.RpToTrans(
        rot_basejoint_to_basecom, pos_basejoint_to_basecom
    )
    adjoint_joint_com = liegroup.Adjoint(trans_joint_com)

    twist_basecom_in_world = np.zeros(6)
    twist_basecom_in_world[0:3] = base_com_ang_vel
    twist_basecom_in_world[3:6] = base_com_lin_vel

    augrot_basecom_world = np.zeros((6, 6))
    augrot_basecom_world[0:3, 0:3] = rot_world_basecom.transpose()
    augrot_basecom_world[3:6, 3:6] = rot_world_basecom.transpose()
    twist_basecom_in_basecom = np.dot(augrot_basecom_world, twist_basecom_in_world)

    twist_basejoint_in_basejoint = np.dot(adjoint_joint_com, twist_basecom_in_basecom)

    augrot_world_basejoint = np.zeros((6, 6))
    augrot_world_basejoint[0:3, 0:3] = rot_world_basejoint
    augrot_world_basejoint[3:6, 3:6] = rot_world_basejoint
    twist_basejoint_in_world = np.dot(
        augrot_world_basejoint, twist_basejoint_in_basejoint
    )

    base_joint_ang_vel = twist_basejoint_in_world[0:3]
    base_joint_lin_vel = twist_basejoint_in_world[3:6]

    return base_joint_pos, base_joint_quat, base_joint_lin_vel, base_joint_ang_vel
