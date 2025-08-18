import numpy as np
import pybullet as pb
import pinocchio as pin

def get_contact_wrenches(robot, ground, model, data, q_pin):
    qc_tau = np.zeros(model.nv)
    contacts = pb.getContactPoints(bodyA=robot, bodyB=ground)

    tau_cf = []

    cf_left = []
    cf_right = []

    # It's crucial that 'data' reflects the current 'q_pin' for frame kinematics.
    # pin.computeFrameJacobian might update kinematics for the specific frame,
    # but to be absolutely safe or if you need frame poses for other things
    # before the Jacobian call, ensure forward kinematics is run.
    # However, data.oMf[frame_id] will be correctly populated by computeFrameJacobian
    # for the q_pin used in its call.

    for contact in contacts:
        link_idx = contact[3]  # PyBullet link index (-1 for base)
        fN = contact[9] * np.array(contact[7])
        f1 = contact[10] * np.array(contact[11])
        f2 = contact[12] * np.array(contact[13])
        f_world = fN + f1 + f2  # Contact force in world coordinates

        #print(f"link {link_idx}: {f_world} | Pos: {contact[5]}")

        pos_c = np.array(contact[5])  # Contact point on robot in world coordinates

        # Get Pinocchio frame ID for the contacted link
        if link_idx == -1:
            try:
                frame_id = model.getFrameId(pb.getBodyInfo(robot)[0].decode('UTF-8')) # Assuming body name is base link
            except Exception as e:
                print(f"Warning: Could not get frame_id for base link (link_idx: {link_idx}). Error: {e}")
                continue
        else:
            link_name_bytes = pb.getJointInfo(robot, link_idx)[12]
            link_name = link_name_bytes.decode('utf-8')
            try:
                frame_id = model.getFrameId(link_name)
            except RuntimeError as e: # Pinocchio throws RuntimeError if frame not found
                print(f"Warning: Could not get frame_id for link_name '{link_name}' (link_idx: {link_idx}). Error: {e}")
                continue


        # Jacobian of the frame 'frame_id' expressed in the world frame.
        # This call updates data.oMf[frame_id] internally for the given q_pin.
        J6_world = pin.computeFrameJacobian(model, data, q_pin, frame_id, pin.ReferenceFrame.WORLD)

        # print(np.shape(J6_world))

        # Origin of the frame 'frame_id' in world coordinates
        pos_frame_origin_w = data.oMf[frame_id].translation

        # Vector from the frame origin to the contact point, in world coordinates
        r_frame_origin_to_contact_point_w = pos_c - pos_frame_origin_w

        # Moment of the contact force about the frame origin, in world coordinates
        M_at_frame_origin_w = np.cross(r_frame_origin_to_contact_point_w, f_world)

        wrench_w = np.hstack([f_world, M_at_frame_origin_w])

        if link_idx == 6:
            cf_left.append(wrench_w)
        if link_idx == 13:
            cf_right.append(wrench_w)

        qc_ind = J6_world.T @ wrench_w

        tau_cf.append(qc_ind)

        qc_tau += qc_ind

    return qc_tau, tau_cf, cf_left, cf_right