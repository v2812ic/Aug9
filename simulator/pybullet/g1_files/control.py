# control.py
import pybullet as pb
from config.g1.sim.pybullet.ihwbc.pybullet_params import G1JointIdx


def apply_control_input_to_pybullet(robot, command):
    """Aplica torques a las 27 DOF activas en el orden esperado."""
    mode = pb.TORQUE_CONTROL

    # ===== LF (0..5) =====
    pb.setJointMotorControl2(robot, G1JointIdx.left_hip_pitch_joint,   mode, force=command[0])
    pb.setJointMotorControl2(robot, G1JointIdx.left_hip_roll_joint,    mode, force=command[1])
    pb.setJointMotorControl2(robot, G1JointIdx.left_hip_yaw_joint,     mode, force=command[2])
    pb.setJointMotorControl2(robot, G1JointIdx.left_knee_joint,        mode, force=command[3])
    pb.setJointMotorControl2(robot, G1JointIdx.left_ankle_pitch_joint, mode, force=command[4])
    pb.setJointMotorControl2(robot, G1JointIdx.left_ankle_roll_joint,  mode, force=command[5])
    # ===== RF (6..11) =====
    pb.setJointMotorControl2(robot, G1JointIdx.right_hip_pitch_joint,  mode, force=command[6])
    pb.setJointMotorControl2(robot, G1JointIdx.right_hip_roll_joint,   mode, force=command[7])
    pb.setJointMotorControl2(robot, G1JointIdx.right_hip_yaw_joint,    mode, force=command[8])
    pb.setJointMotorControl2(robot, G1JointIdx.right_knee_joint,       mode, force=command[9])
    pb.setJointMotorControl2(robot, G1JointIdx.right_ankle_pitch_joint,mode, force=command[10])
    pb.setJointMotorControl2(robot, G1JointIdx.right_ankle_roll_joint, mode, force=command[11])
    # ===== torso (12) =====
    pb.setJointMotorControl2(robot, G1JointIdx.waist_yaw_joint,        mode, force=command[12])
    # ===== LH (13..19) =====
    pb.setJointMotorControl2(robot, G1JointIdx.left_shoulder_pitch_joint, mode, force=command[13])
    pb.setJointMotorControl2(robot, G1JointIdx.left_shoulder_roll_joint,  mode, force=command[14])
    pb.setJointMotorControl2(robot, G1JointIdx.left_shoulder_yaw_joint,   mode, force=command[15])
    pb.setJointMotorControl2(robot, G1JointIdx.left_elbow_joint,          mode, force=command[16])
    pb.setJointMotorControl2(robot, G1JointIdx.left_wrist_roll_joint,     mode, force=command[17])
    pb.setJointMotorControl2(robot, G1JointIdx.left_wrist_pitch_joint,    mode, force=command[18])
    pb.setJointMotorControl2(robot, G1JointIdx.left_wrist_yaw_joint,      mode, force=command[19])
    # ===== RH (20..26) =====
    pb.setJointMotorControl2(robot, G1JointIdx.right_shoulder_pitch_joint, mode, force=command[20])
    pb.setJointMotorControl2(robot, G1JointIdx.right_shoulder_roll_joint,  mode, force=command[21])
    pb.setJointMotorControl2(robot, G1JointIdx.right_shoulder_yaw_joint,   mode, force=command[22])
    pb.setJointMotorControl2(robot, G1JointIdx.right_elbow_joint,          mode, force=command[23])
    pb.setJointMotorControl2(robot, G1JointIdx.right_wrist_roll_joint,     mode, force=command[24])
    pb.setJointMotorControl2(robot, G1JointIdx.right_wrist_pitch_joint,    mode, force=command[25])
    pb.setJointMotorControl2(robot, G1JointIdx.right_wrist_yaw_joint,      mode, force=command[26])


def handle_keyboard_events(keys, rpc_g1_interface, pybullet_util):
    """Mapea teclas numéricas a interrupciones de la interfaz RPC."""
    if pybullet_util.is_key_triggered(keys, "1"):
        rpc_g1_interface.interrupt_.PressOne()
    elif pybullet_util.is_key_triggered(keys, "2"):
        rpc_g1_interface.interrupt_.PressTwo()
    elif pybullet_util.is_key_triggered(keys, "4"):
        rpc_g1_interface.interrupt_.PressFour()
    elif pybullet_util.is_key_triggered(keys, "5"):
        rpc_g1_interface.interrupt_.PressFive()
    elif pybullet_util.is_key_triggered(keys, "6"):
        rpc_g1_interface.interrupt_.PressSix()
    elif pybullet_util.is_key_triggered(keys, "7"):
        rpc_g1_interface.interrupt_.PressSeven()
    elif pybullet_util.is_key_triggered(keys, "8"):
        rpc_g1_interface.interrupt_.PressEight()
    elif pybullet_util.is_key_triggered(keys, "9"):
        rpc_g1_interface.interrupt_.PressNine()
