import numpy as np
import pybullet as pb


def apply_external_forces(robot, t):

    if t > 3:

        pb.applyExternalForce(robot, -1, [0, 0.0, 0.0], [0.0, 0.0, 0.0], pb.WORLD_FRAME)

    return
    