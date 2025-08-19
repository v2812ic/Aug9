import numpy as np
import pybullet as pb


def apply_external_forces(robot, t):

    if t > 2:

        pb.applyExternalForce(robot, -1, [-20, 0.0, 0.0], [0.0, 0.0, 0.0], pb.WORLD_FRAME)

    return
    