# init_pose.py
import numpy as np
import pybullet as pb
from config.g1.sim.pybullet.ihwbc.pybullet_params import G1JointIdx


def set_init_config_pybullet_robot(robot):
    """Resetea posiciones articulares a la postura inicial deseada."""
    # Upperbody
    pb.resetJointState(robot, G1JointIdx.left_shoulder_roll_joint,  np.pi / 6, 0.0)
    pb.resetJointState(robot, G1JointIdx.left_elbow_joint,          np.pi / 2, 0.0)
    pb.resetJointState(robot, G1JointIdx.right_shoulder_roll_joint, -np.pi / 6, 0.0)
    pb.resetJointState(robot, G1JointIdx.right_elbow_joint,         np.pi / 2, 0.0)

    # Lowerbody
    hip_yaw_angle = 0
    pb.resetJointState(robot, G1JointIdx.left_hip_roll_joint,   np.radians( hip_yaw_angle), 0.0)
    pb.resetJointState(robot, G1JointIdx.left_hip_pitch_joint, -np.pi / 6, 0.0)
    pb.resetJointState(robot, G1JointIdx.left_knee_joint,       np.pi / 4, 0.0)
    pb.resetJointState(robot, G1JointIdx.left_ankle_pitch_joint,-np.pi / 12, 0.0)
    pb.resetJointState(robot, G1JointIdx.left_ankle_roll_joint, np.radians(-hip_yaw_angle), 0.0)

    pb.resetJointState(robot, G1JointIdx.right_hip_roll_joint,  np.radians(-hip_yaw_angle), 0.0)
    pb.resetJointState(robot, G1JointIdx.right_hip_pitch_joint, -np.pi / 6, 0.0)
    pb.resetJointState(robot, G1JointIdx.right_knee_joint,       np.pi / 4, 0.0)
    pb.resetJointState(robot, G1JointIdx.right_ankle_pitch_joint,-np.pi / 12, 0.0)
    pb.resetJointState(robot, G1JointIdx.right_ankle_roll_joint, np.radians( hip_yaw_angle), 0.0)
